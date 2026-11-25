"""Current MedMCQA-only BF16 LoRA training runtime for Qwen2.5-3B and Qwen3.5-4B.

The two historical training entries are intentionally untouched. This runtime
uses the current model-specific MedMCQA releases, pinned model revisions,
locked hyperparameters, offline-only loading, and resumable optimizer-boundary
checkpoints. Preflight never loads model weights or starts CUDA training.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

from medical_sft.paths import ROOT
CONFIG_DIR = ROOT / "configs" / "medmcqa_only_training"
PREFLIGHT_OUTPUT_ROOT = ROOT / "outputs" / "preflight" / "medmcqa_training"
OUTPUT_ROOT = ROOT / "outputs" / "medmcqa_only_sft_20261109"
MAX_LENGTH = 512
EXPECTED_PHYSICAL_GPU_INDEX = "1"
EXPECTED_GPU_BUS_SUFFIX = "b3:00.0"
MIN_FREE_GPU_MIB = 12 * 1024


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def load_initial_train_scope_ids() -> tuple[set[str], dict[str, Any]]:
    """Load and verify the immutable 173,648-ID Train subset membership list."""
    release = read_json(ROOT / "data_release_manifest.json")
    split = release.get("split_protocol", {})
    artifact = release.get("artifacts", {}).get("train_scope", {})
    relative_path = "data/processed/train/medmcqa_original_train_scope_ids.jsonl"
    frozen_scope_rows = split.get("frozen_train_scope_rows", split.get("initial_train_rows"))
    if (
        frozen_scope_rows != 173_648
        or split.get("train_scope_path") != relative_path
        or artifact.get("path") != relative_path
        or artifact.get("rows") != 173_648
    ):
        raise ValueError("Initial MedMCQA Train-scope manifest identity mismatch")

    path = ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)

    ids: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank Train-scope ID record at line {line_number}")
            record = json.loads(line)
            sample_id = record.get("id") if isinstance(record, dict) else None
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError(f"Invalid Train-scope ID at line {line_number}")
            if sample_id in ids:
                raise ValueError(f"Duplicate Train-scope ID: {sample_id}")
            ids.add(sample_id)

    if len(ids) != 173_648:
        raise ValueError(f"Train-scope ID count mismatch: {len(ids)}")
    return ids, {
        "path": str(path.relative_to(ROOT)),
        "rows": len(ids),
        "bytes": path.stat().st_size,
        "sha256": artifact["sha256"],
    }


def assert_source_ids_in_train_scope(
    source_ids: set[str], train_scope_ids: set[str]
) -> None:
    """Fail closed if any final training record is outside official Train scope."""
    outside_scope = {str(sample_id) for sample_id in source_ids} - train_scope_ids
    if outside_scope:
        raise ValueError(
            "Final training data contains IDs outside the original MedMCQA Train scope: "
            + ", ".join(sorted(outside_scope)[:10])
        )


def load_profile(profile_name: str) -> dict[str, Any]:
    if profile_name not in {"qwen25_3b", "qwen35_4b"}:
        raise ValueError(f"Unknown training profile: {profile_name}")
    config_name = "qwen25_3b_instruct.json" if profile_name == "qwen25_3b" else "qwen35_4b.json"
    path = CONFIG_DIR / config_name
    if not path.is_file():
        raise FileNotFoundError(path)
    profile = read_json(path)
    if profile.get("schema_version") != "medmcqa-only-bf16-lora-v1":
        raise ValueError("Unexpected current training config schema")
    if profile.get("profile") != profile_name:
        raise ValueError("Training profile/config identity mismatch")
    training = profile["training"]
    if profile["max_length"] != MAX_LENGTH or training["truncation"] is not False:
        raise ValueError("Current training protocol requires max_length=512 and truncation=false")
    if training["method"] != "BF16 LoRA" or training["precision"] != "bfloat16":
        raise ValueError("Current training baseline must remain BF16 base weights + LoRA")
    if training["micro_batch_size"] * training["gradient_accumulation_steps"] != training["effective_batch_size"]:
        raise ValueError("Effective batch size does not match micro-batch × accumulation")
    micro_batches = math.ceil(profile["rows"] / training["micro_batch_size"])
    steps = math.ceil(micro_batches / training["gradient_accumulation_steps"]) * training["epochs"]
    warmup = math.ceil(steps * training["warmup_ratio"])
    if training["warmup_steps"] != warmup:
        raise ValueError(f"Warmup lock mismatch: expected {warmup}, got {training['warmup_steps']}")
    if profile["rows"] != 137_928 or profile["data_bytes"] <= 0:
        raise ValueError("Unexpected current MedMCQA-only training release size")
    return profile


def expected_steps(profile: dict[str, Any]) -> int:
    training = profile["training"]
    micro_batches = math.ceil(profile["rows"] / training["micro_batch_size"])
    return math.ceil(micro_batches / training["gradient_accumulation_steps"]) * training["epochs"]


def validate_record(row: dict[str, Any], line_number: int, profile: dict[str, Any]) -> None:
    prefix = f"JSONL line {line_number}"
    if not isinstance(row, dict):
        raise ValueError(f"{prefix}: record must be an object")
    required = {
        "id", "source", "source_id", "quality_class", "question", "options",
        "original_gold", "gold_index_base", "gold_answer", "explanation",
        "messages", "full_token_length", "prompt_token_length",
        "target_token_length", "training_protocol", "provenance",
    }
    missing = required.difference(row)
    if missing:
        raise ValueError(f"{prefix}: missing fields {sorted(missing)}")
    if row["source"] != "MedMCQA":
        raise ValueError(f"{prefix}: final training source must be MedMCQA only")
    source_id = str(row["source_id"])
    if not source_id or row["id"] != f"MedMCQA:{source_id}":
        raise ValueError(f"{prefix}: id/source_id identity mismatch")
    if row["quality_class"] not in {"A", "B"}:
        raise ValueError(f"{prefix}: unexpected training quality class")
    if not isinstance(row["question"], str) or not row["question"].strip():
        raise ValueError(f"{prefix}: question is blank")
    options = row["options"]
    if not isinstance(options, dict) or set(options) != {"A", "B", "C", "D"}:
        raise ValueError(f"{prefix}: MedMCQA options must be exactly A/B/C/D")
    if any(not isinstance(value, str) or not value.strip() for value in options.values()):
        raise ValueError(f"{prefix}: option text is blank")
    gold = row["gold_answer"]
    if gold not in options:
        raise ValueError(f"{prefix}: gold answer is not a supplied option")
    if row["gold_index_base"] != 0 or type(row["original_gold"]) is not int:
        raise ValueError(f"{prefix}: invalid MedMCQA gold index representation")
    if row["original_gold"] not in range(4) or "ABCD"[row["original_gold"]] != gold:
        raise ValueError(f"{prefix}: original gold does not map to gold_answer")
    explanation = row["explanation"]
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError(f"{prefix}: explanation is missing or blank")
    messages = row["messages"]
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{prefix}: expected exactly two chat messages")
    if messages[0].get("role") != "user" or not isinstance(messages[0].get("content"), str) or not messages[0]["content"].strip():
        raise ValueError(f"{prefix}: invalid user prompt")
    if messages[1].get("role") != "assistant":
        raise ValueError(f"{prefix}: invalid assistant target role")
    expected_target = f"Answer: {gold}\nExplanation: {explanation}"
    if messages[1].get("content") != expected_target:
        raise ValueError(f"{prefix}: assistant target differs from Answer + Explanation fields")
    if row["training_protocol"] != profile["training_protocol"]:
        raise ValueError(f"{prefix}: model/revision/tokenization protocol mismatch")
    lengths = (
        row["full_token_length"],
        row["prompt_token_length"],
        row["target_token_length"],
    )
    if any(type(value) is not int for value in lengths):
        raise ValueError(f"{prefix}: token lengths must be integers")
    full_length, prompt_length, target_length = lengths
    if (
        full_length <= 0 or full_length > MAX_LENGTH
        or prompt_length <= 0 or target_length <= 0
        or full_length != prompt_length + target_length
    ):
        raise ValueError(f"{prefix}: invalid token accounting or overlength sample")
    if not isinstance(row["provenance"], dict):
        raise ValueError(f"{prefix}: provenance must be an object")


class JsonlOffsetDataset(Dataset):
    """Index JSONL byte offsets; never materialize the complete training text in RAM."""

    def __init__(self, path: Path, profile: dict[str, Any]):
        self.path = Path(path)
        self.profile = profile
        self.offsets: list[int] = []
        ids: set[str] = set()
        source_ids: set[str] = set()
        max_length = -1
        max_index = 0
        histogram: Counter[str] = Counter()

        with self.path.open("rb") as handle:
            while True:
                offset = handle.tell()
                raw = handle.readline()
                if not raw:
                    break
                line_number = len(self.offsets) + 1
                if not raw.strip():
                    raise ValueError(f"Blank JSONL line at {line_number}")
                try:
                    row = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid UTF-8 JSON at line {line_number}") from exc
                validate_record(row, line_number, profile)
                sample_id = row["id"]
                source_id = str(row["source_id"])
                if sample_id in ids or source_id in source_ids:
                    raise ValueError(f"Duplicate MedMCQA training identity at line {line_number}")
                ids.add(sample_id)
                source_ids.add(source_id)
                self.offsets.append(offset)
                length = row["full_token_length"]
                if length > max_length:
                    max_length, max_index = length, line_number - 1
                bucket = "0-127" if length <= 127 else "128-255" if length <= 255 else "256-383" if length <= 383 else "384-512"
                histogram[bucket] += 1

        if len(self.offsets) != profile["rows"]:
            raise ValueError(f"Training row count mismatch: expected={profile['rows']}, actual={len(self.offsets)}")
        if max_length != profile["max_full_tokens"]:
            raise ValueError("Training data maximum token length differs from the locked preflight")
        train_scope_ids, train_scope_info = load_initial_train_scope_ids()
        assert_source_ids_in_train_scope(source_ids, train_scope_ids)
        from medical_sft.evaluation.medmcqa_evaluation import assert_training_safe
        assert_training_safe(self.path, ids | source_ids)
        self.audit = {
            "path": str(self.path.resolve()),
            "bytes": self.path.stat().st_size,
            "rows": len(self.offsets),
            "unique_ids": len(ids),
            "source_counts": {"MedMCQA": len(self.offsets)},
            "initial_train_scope_membership_check": {
                "status": "PASS",
                "scope_rows": train_scope_info["rows"],
                "training_rows": len(source_ids),
                "final_training_ids_outside_scope": 0,
                "scope_sha256": train_scope_info["sha256"],
            },
            "max_full_tokens": max_length,
            "max_length_row_index": max_index,
            "length_histogram": dict(histogram),
            "evaluation_overlap_check": "PASS",
        }

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, index: int) -> dict[str, Any]:
        with self.path.open("rb") as handle:
            handle.seek(self.offsets[index])
            return json.loads(handle.readline().decode("utf-8"))


def resolve_model_path(profile: dict[str, Any], supplied: str | Path | None = None) -> Path:
    expected = (ROOT / profile["model_path_relative"]).resolve()
    if supplied is not None:
        raw = Path(supplied)
        if raw.is_symlink():
            raise ValueError("Model path must not be a symlink")
        actual = raw.resolve()
        if actual != expected:
            raise ValueError(f"Model path must resolve to the pinned snapshot: {expected}")
    if not expected.is_dir():
        raise FileNotFoundError(f"Pinned full model snapshot is missing: {expected}")
    return expected


def verify_model_snapshot(model_path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    """Model loading performs the asset checks; do not hash model weights."""
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    return {"path": str(model_path)}


def verify_reference_preflight(profile: dict[str, Any]) -> dict[str, Any]:
    report = read_json(ROOT / profile["preflight_report_path"])
    if report.get("status") != "PASS" or report.get("rows") != profile["rows"]:
        raise ValueError("Reference length preflight is incomplete")
    return report


def load_engine(profile: dict[str, Any], model_path: Path):
    if profile["model_family"] == "qwen2_causal_lm":
        from transformers import AutoTokenizer
        engine = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        tokenizer = engine
    elif profile["model_family"] == "qwen3_5_multimodal":
        from transformers import AutoProcessor
        engine = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        tokenizer = getattr(engine, "tokenizer", None)
        if tokenizer is None:
            raise TypeError("Qwen3.5 AutoProcessor does not expose its tokenizer")
    else:
        raise ValueError(f"Unsupported model family: {profile['model_family']}")
    if not callable(getattr(engine, "apply_chat_template", None)):
        raise TypeError(f"Loaded {type(engine).__name__} does not expose apply_chat_template")
    if type(tokenizer).__name__ != profile["tokenizer_class"]:
        raise TypeError(f"Unexpected tokenizer class: {type(tokenizer).__name__}")
    template = getattr(engine, "chat_template", None) or getattr(tokenizer, "chat_template", None)
    if not template:
        raise ValueError("Pinned model has no Chat Template")
    return engine, tokenizer


def apply_template(engine, profile: dict[str, Any], messages: list[dict[str, str]], add_generation_prompt: bool):
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if profile["model_family"] == "qwen3_5_multimodal":
        kwargs["enable_thinking"] = False
        kwargs["processor_kwargs"] = {"padding": False, "truncation": False}
    else:
        kwargs["padding"] = False
        kwargs["truncation"] = False
    return engine.apply_chat_template(messages, **kwargs)


def encode_pair(engine, profile: dict[str, Any], messages: list[dict[str, str]]):
    full = apply_template(engine, profile, messages, False)
    prompt = apply_template(engine, profile, messages[:-1], True)
    full_ids = full["input_ids"][0].long()
    prompt_ids = prompt["input_ids"][0].long()
    if full_ids.numel() <= prompt_ids.numel():
        raise ValueError("Chat Template produced no supervised completion tokens")
    if not torch.equal(full_ids[: prompt_ids.numel()], prompt_ids):
        raise ValueError("Prompt token sequence is not a prefix of the full conversation")
    attention = full.get("attention_mask")
    if attention is None:
        attention = torch.ones_like(full["input_ids"])
    return full_ids, attention[0].long(), prompt_ids


def make_collator(engine, tokenizer, profile: dict[str, Any]):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")

    def collate(samples: list[dict[str, Any]]):
        ids_list, masks_list, labels_list, sample_ids, lengths = [], [], [], [], []
        for sample in samples:
            full_ids, attention, prompt_ids = encode_pair(engine, profile, sample["messages"])
            full_length = full_ids.numel()
            prompt_length = prompt_ids.numel()
            if full_length > MAX_LENGTH or prompt_length >= MAX_LENGTH:
                raise ValueError(f"Sample {sample['id']} exceeds the locked 512-token limit; truncation is forbidden")
            if full_length != sample["full_token_length"] or prompt_length != sample["prompt_token_length"]:
                raise ValueError(f"Sample {sample['id']} token lengths differ from the frozen preflight")
            if full_length - prompt_length != sample["target_token_length"]:
                raise ValueError(f"Sample {sample['id']} target token length differs from the frozen preflight")
            labels = full_ids.clone()
            labels[:prompt_length] = -100
            if not torch.any(labels != -100).item():
                raise ValueError(f"Sample {sample['id']} has no supervised target tokens")
            ids_list.append(full_ids)
            masks_list.append(attention)
            labels_list.append(labels)
            sample_ids.append(sample["id"])
            lengths.append(full_length)
        batch = {
            "input_ids": pad_sequence(ids_list, batch_first=True, padding_value=int(pad_id)),
            "attention_mask": pad_sequence(masks_list, batch_first=True, padding_value=0),
        }
        labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)
        return batch, labels, {"sample_ids": sample_ids, "real_lengths": lengths}

    return collate


def environment_record() -> dict[str, Any]:
    def version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": version("transformers"),
        "peft": version("peft"),
        "accelerate": version("accelerate"),
        "numpy": version("numpy"),
        "cuda_available": torch.cuda.is_available(),
    }


def verify_evaluation_protocol() -> dict[str, Any]:
    """Verify the fixed Train-derived validation split and both final test sets."""
    from medical_sft.evaluation.medmcqa_evaluation import load_evaluation

    release = read_json(ROOT / "data_release_manifest.json")
    split = release.get("split_protocol", {})
    expected_split = {
        "source_split": "official_train",
        "raw_rows": 182_822,
        "pre_split_exclusions": {
            "answer_conflicts": 4,
            "duplicate_questions": 31,
            "total": 35,
        },
        "eligible_rows": 182_787,
        "group_key": "normalized_question",
        "normalization": "strip; collapse consecutive whitespace; casefold",
        "method": "grouped_random_split",
        "validation_fraction_target": 0.05,
        "random_seed": 20261020,
        "random_generator": "random.Random(seed)",
        "validation_rows": 9_139,
        "train_scope_path": "data/processed/train/medmcqa_original_train_scope_ids.jsonl",
        "internal_validation_path": "data/processed/validation/internal_medmcqa/dev_eval.jsonl",
    }
    for key, expected in expected_split.items():
        if split.get(key) != expected:
            raise ValueError(f"MedMCQA Train/validation split protocol mismatch: {key}")
    cleaning_input_rows = expected_split["raw_rows"] - expected_split["validation_rows"]
    if split.get("cleaning_input_rows", cleaning_input_rows) != cleaning_input_rows:
        raise ValueError("MedMCQA cleaning input count mismatch")

    internal_rows, internal_info = load_evaluation("dev", root=ROOT)
    official_rows, official_info = load_evaluation("validation", root=ROOT)
    train_scope_ids, train_scope_info = load_initial_train_scope_ids()
    internal_ids = {str(row["id"]) for row in internal_rows}
    if train_scope_ids & internal_ids:
        raise ValueError("Original MedMCQA Train scope and internal validation IDs overlap")
    if len(internal_rows) != 9_139 or len(official_rows) != 4_183:
        raise ValueError("MedMCQA internal validation / official test row count mismatch")
    if internal_info["input_path"].resolve() != (ROOT / expected_split["internal_validation_path"]).resolve():
        raise ValueError("Internal validation path differs from the frozen split protocol")
    if official_info["input_path"].resolve() != (
        ROOT / release["artifacts"]["reserved"]["path"]
    ).resolve():
        raise ValueError("MedMCQA official Validation test path differs from the release manifest")

    expqa_path = ROOT / "data/evaluation/medexpqa_full/manifest.json"
    expqa = read_json(expqa_path)
    if (
        expqa.get("purpose") != "evaluation_only"
        or expqa.get("total_rows") != 618
        or expqa.get("total_split_rows") != {"train": 430, "validation": 63, "test": 125}
    ):
        raise ValueError("MedExpQA must remain the complete 618-row evaluation-only dataset")

    checked_files = []
    medexpqa_ids: set[str] = set()
    for split_name in ("train", "validation", "test"):
        spec = expqa["splits"][split_name]
        parts = spec["composition"] if split_name == "train" else [spec]
        counted = 0
        for part in parts:
            path = ROOT / "data/evaluation/medexpqa_full" / part["file"]
            if not path.is_file():
                raise ValueError(f"MedExpQA evaluation file missing: {part['file']}")
            rows = 0
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        raise ValueError(f"Blank MedExpQA evaluation row: {part['file']}:{line_number}")
                    record = json.loads(line)
                    sample_id = record.get("id") if isinstance(record, dict) else None
                    if not isinstance(sample_id, (str, int)) or not str(sample_id):
                        raise ValueError(f"Missing MedExpQA evaluation ID: {part['file']}:{line_number}")
                    if str(sample_id) in medexpqa_ids:
                        raise ValueError(f"Duplicate MedExpQA evaluation ID: {sample_id}")
                    medexpqa_ids.add(str(sample_id))
                    rows += 1
            if rows != part["rows"]:
                raise ValueError(f"MedExpQA evaluation row count mismatch: {part['file']}")
            counted += rows
            checked_files.append({
                "path": str(path.relative_to(ROOT)),
                "rows": rows,
                "sha256": part["sha256"],
            })
        if counted != spec["rows"]:
            raise ValueError(f"MedExpQA {split_name} split row count mismatch")

    from medical_sft.evaluation.medexpqa_evaluation import load_full_evaluation
    medexpqa_rows, medexpqa_info = load_full_evaluation()
    if (
        medexpqa_info["total_rows"] != 618
        or medexpqa_info["unique_ids"] != 618
        or medexpqa_info["split_rows"] != {"train": 430, "validation": 63, "test": 125}
        or {str(row["id"]) for row in medexpqa_rows} != medexpqa_ids
    ):
        raise ValueError("MedExpQA full evaluation records do not match the verified release manifest")

    medmcqa_eval_ids = {str(row["id"]) for row in internal_rows + official_rows}
    if medexpqa_ids & medmcqa_eval_ids:
        raise ValueError("MedExpQA and MedMCQA evaluation IDs overlap")

    return {
        "initial_train_scope": {
            **train_scope_info,
            "ids_disjoint_from_internal_validation": True,
        },
        "internal_validation": {
            "path": str(internal_info["input_path"].relative_to(ROOT)),
            "rows": len(internal_rows),
            "seed": 20261020,
            "target_fraction": 0.05,
            "method": "grouped_random_by_normalized_question",
            "role": "internal_validation_only",
        },
        "test_sets": [
            {
                "name": "MedMCQA official Validation",
                "path": str(official_info["input_path"].relative_to(ROOT)),
                "rows": len(official_rows),
                "role": "final_test",
            },
            {
                "name": "MedExpQA full",
                "manifest_path": str(expqa_path.relative_to(ROOT)),
                "rows": expqa["total_rows"],
                "split_rows": expqa["total_split_rows"],
                "unique_ids": len(medexpqa_ids),
                "cross_dataset_id_overlap": 0,
                "role": "final_test_evaluation_only",
                "files": checked_files,
            },
        ],
        "evaluation_source_hash_and_row_count_check": "PASS",
    }


def run_preflight(profile_name: str, supplied_model_path: str | Path | None = None) -> dict[str, Any]:
    profile = load_profile(profile_name)
    model_path = resolve_model_path(profile, supplied_model_path)
    model_validation = verify_model_snapshot(model_path, profile)
    reference = verify_reference_preflight(profile)
    evaluation_protocol = verify_evaluation_protocol()
    data_path = ROOT / profile["data_path"]
    dataset = JsonlOffsetDataset(data_path, profile)
    engine, tokenizer = load_engine(profile, model_path)
    sample_indices = list(dict.fromkeys([
        0,
        len(dataset) // 2,
        dataset.audit["max_length_row_index"],
        len(dataset) - 1,
    ]))
    samples = [dataset[index] for index in sample_indices]
    collator = make_collator(engine, tokenizer, profile)
    batch, labels, info = collator(samples)
    if batch["input_ids"].shape != batch["attention_mask"].shape or labels.shape != batch["input_ids"].shape:
        raise ValueError("CPU DataLoader/collator tensor shape mismatch")
    if info["sample_ids"] != [sample["id"] for sample in samples]:
        raise ValueError("CPU DataLoader/collator changed sample identity or order")
    if int((labels != -100).sum()) != sum(sample["target_token_length"] for sample in samples):
        raise ValueError("Prompt masking or full-target supervision check failed")
    if int(batch["attention_mask"].sum()) != sum(info["real_lengths"]):
        raise ValueError("Padding mask check failed")
    env = environment_record()
    version_matches = env["transformers"] == profile["expected_transformers_version"]
    training = profile["training"]
    return {
        "status": "PASS" if version_matches else "PASS_CPU_PREFLIGHT_VERSION_MISMATCH",
        "ready_to_train": version_matches and env["torch"] == profile.get("expected_torch_version"),
        "profile": profile_name,
        "model": model_validation,
        "model_path": str(model_path),
        "tokenizer_processor_reference": reference,
        "data": dataset.audit,
        "evaluation_protocol": evaluation_protocol,
        "cpu_batch_preflight": {
            "status": "PASS",
            "sample_indices": sample_indices,
            "sample_ids": info["sample_ids"],
            "real_lengths": info["real_lengths"],
            "batch_shape": list(batch["input_ids"].shape),
            "supervised_tokens": int((labels != -100).sum()),
            "padding_mask_sum": int(batch["attention_mask"].sum()),
            "prompt_masking": "PASS",
            "full_target_supervision": "PASS",
            "truncation": False,
        },
        "training": {
            **training,
            "total_optimizer_steps": expected_steps(profile),
            "effective_batch_size": training["micro_batch_size"] * training["gradient_accumulation_steps"],
        },
        "environment": env,
        "required_training_environment": {
            "torch": profile.get("expected_torch_version"),
            "transformers": profile["expected_transformers_version"],
            "CUDA_VISIBLE_DEVICES": "1",
            "physical_gpu": "GPU 1, PCI bus B3:00.0",
            "minimum_free_gpu_mib": MIN_FREE_GPU_MIB,
        },
        "training_started": False,
        "model_weights_loaded": False,
        "gpu_forward_backward_or_update": False,
    }


def write_json(path: Path, value: dict[str, Any], *, exclusive: bool = False) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    if exclusive:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
    else:
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def check_training_environment(profile: dict[str, Any]) -> dict[str, Any]:
    env = environment_record()
    if env["torch"] != profile.get("expected_torch_version"):
        raise RuntimeError(
            f"Training requires torch {profile.get('expected_torch_version')}; current={env['torch']}"
        )
    if env["transformers"] != profile["expected_transformers_version"]:
        raise RuntimeError(
            f"Training requires transformers {profile['expected_transformers_version']}; current={env['transformers']}"
        )
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Training must run with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != EXPECTED_PHYSICAL_GPU_INDEX:
        raise RuntimeError("Only physical GPU 1 is authorized: set CUDA_VISIBLE_DEVICES=1 before Python starts")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Expected exactly one visible CUDA device with BF16 support")
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,pci.bus_id,memory.free", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    selected = None
    for line in query.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            continue
        if fields[0] == EXPECTED_PHYSICAL_GPU_INDEX:
            selected = fields
            break
    if selected is None or not selected[1].lower().endswith(EXPECTED_GPU_BUS_SUFFIX):
        raise RuntimeError("Physical GPU 1 PCI bus identity is not the previously authorized B3:00.0 device")
    free_mib = int(selected[2])
    if free_mib < MIN_FREE_GPU_MIB:
        raise RuntimeError(f"GPU 1 free memory {free_mib} MiB is below the {MIN_FREE_GPU_MIB} MiB training gate")
    return {
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "logical_cuda_devices": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "physical_gpu_index": selected[0],
        "pci_bus_id": selected[1],
        "free_memory_mib_before_load": free_mib,
        "bf16_supported": True,
    }


def load_bf16_lora_model(profile: dict[str, Any], model_path: Path):
    from peft import LoraConfig, get_peft_model
    if profile["model_family"] == "qwen2_causal_lm":
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), dtype=torch.bfloat16, device_map={"": 0},
            local_files_only=True, trust_remote_code=False, use_safetensors=True,
        )
    else:
        from transformers import AutoModelForMultimodalLM
        model = AutoModelForMultimodalLM.from_pretrained(
            str(model_path), dtype=torch.bfloat16, device_map={"": 0},
            local_files_only=True, trust_remote_code=False, use_safetensors=True,
        )
    if type(model).__name__ != profile["model_class"]:
        raise TypeError(f"Unexpected loaded model class: {type(model).__name__}")
    model.requires_grad_(False)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    lora = profile["training"]["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lora["r"], lora_alpha=lora["alpha"], target_modules=lora["target_modules"],
        lora_dropout=lora["dropout"], bias=lora["bias"], task_type=lora["task_type"],
    ))
    model.config.use_cache = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = False
    model.train()
    audit = audit_loaded_model(model, profile)
    return model, audit


def audit_loaded_model(model, profile: dict[str, Any]) -> dict[str, Any]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    base = [(name, parameter) for name, parameter in model.named_parameters()
            if ".lora_A." not in name and ".lora_B." not in name]
    targets = [name for name, module in model.named_modules()
               if hasattr(module, "lora_A") and "default" in module.lora_A]
    if not trainable or not targets:
        raise RuntimeError("PEFT did not attach trainable LoRA parameters")
    if len(targets) != profile["expected_lora_modules"]:
        raise RuntimeError(
            f"LoRA target module count mismatch: expected={profile['expected_lora_modules']}, actual={len(targets)}"
        )
    if any(name.rsplit(".", 1)[-1] not in {"q_proj", "v_proj"} for name in targets):
        raise RuntimeError("Unexpected LoRA target module outside q_proj/v_proj")
    if any(".lora_A." not in name and ".lora_B." not in name for name, _ in trainable):
        raise RuntimeError("A non-LoRA parameter was unexpectedly left trainable")
    if any(parameter.requires_grad for _, parameter in base):
        raise RuntimeError("Base model contains trainable parameters")
    if any(parameter.is_floating_point() and parameter.dtype != torch.bfloat16 for _, parameter in base):
        raise RuntimeError("Base model contains non-BF16 floating-point parameters")
    if any(parameter.device != torch.device("cuda:0") for parameter in model.parameters()):
        raise RuntimeError("Model parameter is not on the authorized logical CUDA 0 device")
    if any(buffer.device != torch.device("cuda:0") for buffer in model.buffers()):
        raise RuntimeError("Model buffer is not on the authorized logical CUDA 0 device")
    if not model.is_gradient_checkpointing or model.config.use_cache:
        raise RuntimeError("Gradient checkpointing/use_cache settings are inconsistent")
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        raise RuntimeError("Quantization is forbidden for the current BF16 LoRA baseline")
    return {
        "status": "PASS",
        "base_model_class": type(model.get_base_model()).__name__,
        "base_frozen": True,
        "base_parameter_count": sum(parameter.numel() for _, parameter in base),
        "trainable_lora_parameter_count": sum(parameter.numel() for _, parameter in trainable),
        "lora_target_module_count": len(targets),
        "lora_targets": targets,
        "all_base_floating_weights_bf16": True,
        "all_parameters_and_buffers_cuda_0": True,
        "gradient_checkpointing": True,
        "use_cache": False,
        "quantization": "none",
    }


def capture_rng() -> dict[str, Any]:
    state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (state[0], state[1].tolist(), state[2], state[3], state[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype="uint32"), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_checkpoint(output: Path, model, optimizer, scheduler, state: dict[str, Any], run_config: dict[str, Any]) -> Path:
    step = state["step"]
    final = output / "checkpoints" / f"step_{step:06d}"
    temporary = final.with_name(final.name + ".partial")
    if final.exists() or temporary.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {final}")
    temporary.mkdir()
    model.save_pretrained(temporary / "adapter", safe_serialization=True)
    rng = capture_rng()
    payload = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": rng,
        "position": state,
        "config": run_config,
    }
    with (temporary / "training_state.pt").open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    write_json(temporary / "boundary.json", {
        "step": step,
        "next_indices": state["order"][state["cursor"]:state["cursor"] + (
            run_config["micro_batch_size"] * run_config["gradient_accumulation_steps"]
        )],
        "config": run_config,
    }, exclusive=True)
    temporary.rename(final)
    restore_rng(rng)
    return final


def restore_checkpoint(path: Path, model, optimizer, scheduler, run_config: dict[str, Any]):
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    path = Path(path)
    payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    if payload.get("config") != run_config:
        raise RuntimeError("Resume checkpoint training config differs from this run")
    weights = load_file(str(path / "adapter" / "adapter_model.safetensors"), device="cpu")
    expected = get_peft_model_state_dict(model)
    if set(weights) != set(expected) or any(weights[key].shape != expected[key].shape for key in weights):
        raise RuntimeError("Resume Adapter keys/shapes differ from the loaded model")
    result = set_peft_model_state_dict(model, weights)
    if getattr(result, "unexpected_keys", []):
        raise RuntimeError("Unexpected Adapter keys while restoring checkpoint")
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    restore_rng(payload["rng"])
    boundary = read_json(path / "boundary.json")
    if boundary["step"] != payload["position"]["step"]:
        raise RuntimeError("Checkpoint step boundary mismatch")
    return payload["position"], {
        "status": "PASS",
        "step": payload["position"]["step"],
        "checkpoint": str(path.resolve()),
        "next_indices": boundary["next_indices"],
    }


def validate_output_path(profile: dict[str, Any], resume_from: str | Path | None = None) -> tuple[Path, Path | None]:
    output = (ROOT / profile["output_dir"]).resolve()
    allowed_root = OUTPUT_ROOT.resolve()
    if not output.is_relative_to(allowed_root):
        raise ValueError("Training output must remain under the current MedMCQA-only output root")
    for parent in (output, *output.parents):
        if parent.is_symlink():
            raise ValueError("Training output path must not contain symlinks")
        if parent == ROOT:
            break
    if resume_from is None:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite non-empty training output: {output}")
        return output, None
    checkpoint = Path(resume_from)
    if checkpoint.is_symlink():
        raise ValueError("Resume checkpoint must not be a symlink")
    checkpoint = checkpoint.resolve()
    if checkpoint.parent != (output / "checkpoints").resolve():
        raise ValueError("Resume checkpoint must belong to this model's current output directory")
    if not checkpoint.is_dir() or not checkpoint.name.startswith("step_"):
        raise FileNotFoundError(f"Invalid resume checkpoint: {checkpoint}")
    return output, checkpoint


def make_run_config(profile: dict[str, Any], preflight: dict[str, Any]) -> dict[str, Any]:
    training = profile["training"]
    return {
        "schema": "medmcqa-only-bf16-lora-run-v1",
        "profile": profile["profile"],
        "base_model": profile["model_id"],
        "model_revision": profile["revision"],
        "model_path": preflight["model_path"],
        "training_data": profile["data_path"],
        "training_data_bytes": profile["data_bytes"],
        "training_rows": profile["rows"],
        "training_source": "MedMCQA official Train only",
        "method": training["method"],
        "precision": training["precision"],
        "max_length": profile["max_length"],
        "truncation": False,
        "micro_batch_size": training["micro_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "effective_batch_size": training["effective_batch_size"],
        "epochs": training["epochs"],
        "total_optimizer_steps": expected_steps(profile),
        "learning_rate": training["learning_rate"],
        "weight_decay": training["weight_decay"],
        "optimizer": training["optimizer"],
        "scheduler": training["scheduler"],
        "warmup_ratio": training["warmup_ratio"],
        "warmup_steps": training["warmup_steps"],
        "seed": training["seed"],
        "gradient_checkpointing": training["gradient_checkpointing"],
        "lora": training["lora"],
        "save_steps": training["save_steps"],
        "official_validation_during_training": False,
        "medexpqa_training_rows": 0,
    }


def ensure_no_live_run(output: Path) -> None:
    status_path = output / "status.json"
    if not status_path.is_file():
        return
    status = read_json(status_path)
    if status.get("status") != "RUNNING":
        return
    pid = status.get("pid")
    if not isinstance(pid, int):
        raise RuntimeError("Previous run status is RUNNING but has no verifiable PID")
    try:
        import psutil
        process = psutil.Process(pid)
        if process.is_running():
            created = status.get("process_create_time")
            if created is None or abs(process.create_time() - float(created)) < 1.0:
                raise RuntimeError(f"Training process PID {pid} is still running; do not resume concurrently")
    except ImportError as exc:
        raise RuntimeError("psutil is required to verify a stale RUNNING status before resume") from exc
    except psutil.NoSuchProcess:
        return


def run_training(profile_name: str, profile: dict[str, Any], preflight: dict[str, Any],
                 resume_from: str | Path | None = None, stop_after_step: int | None = None) -> dict[str, Any]:
    if not preflight.get("ready_to_train"):
        raise RuntimeError("CPU preflight passed, but the current runtime does not match the pinned training environment")
    gpu = check_training_environment(profile)
    output, checkpoint = validate_output_path(profile, resume_from)
    if checkpoint is not None:
        ensure_no_live_run(output)
    dataset = JsonlOffsetDataset(ROOT / profile["data_path"], profile)
    engine, tokenizer = load_engine(profile, Path(preflight["model_path"]))
    collator = make_collator(engine, tokenizer, profile)
    run_config = make_run_config(profile, preflight)
    total_steps = run_config["total_optimizer_steps"]
    if stop_after_step is not None and not 1 <= stop_after_step <= total_steps:
        raise ValueError(f"--stop-after-step must be between 1 and {total_steps}")

    if checkpoint is None:
        output.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoints", "adapter", "config", "logs"):
            (output / name).mkdir(exist_ok=False)
        write_json(output / "config" / "training_config.json", run_config, exclusive=True)
        write_json(output / "config" / "preflight.json", preflight, exclusive=True)
        write_json(output / "config" / "gpu_preflight.json", gpu, exclusive=True)
    else:
        saved_config = read_json(output / "config" / "training_config.json")
        if saved_config != run_config:
            raise RuntimeError("Existing run config differs from the pinned current training profile")
        if not (output / "logs" / "metrics.jsonl").is_file():
            raise RuntimeError("Cannot resume: optimizer-step metrics log is missing")

    status = {
        "status": "RUNNING",
        "profile": profile_name,
        "pid": os.getpid(),
        "process_create_time": None,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "model_loading",
    }
    try:
        import psutil
        status["process_create_time"] = psutil.Process(os.getpid()).create_time()
    except ImportError:
        pass
    write_json(output / "status.json", status)

    try:
        random.seed(profile["training"]["seed"])
        np.random.seed(profile["training"]["seed"])
        torch.manual_seed(profile["training"]["seed"])
        torch.cuda.manual_seed_all(profile["training"]["seed"])
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)

        model, model_audit = load_bf16_lora_model(profile, Path(preflight["model_path"]))
        if checkpoint is None:
            write_json(output / "config" / "model_audit.json", model_audit, exclusive=True)
        trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
        training = profile["training"]
        optimizer = torch.optim.AdamW(
            [parameter for _, parameter in trainable],
            lr=training["learning_rate"],
            weight_decay=training["weight_decay"],
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, training["warmup_steps"], total_steps
        )
        state = {
            "step": 0, "micro_step": 0, "epoch": 0, "cursor": 0,
            "order": torch.randperm(
                len(dataset),
                generator=torch.Generator().manual_seed(training["seed"]),
            ).tolist(),
            "samples": 0, "real_tokens": 0, "supervised_tokens": 0,
            "maximum_actual_length": 0,
        }
        if checkpoint is not None:
            state, resume_proof = restore_checkpoint(checkpoint, model, optimizer, scheduler, run_config)
            logged_steps = [
                json.loads(line)["step"]
                for line in (output / "logs" / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if logged_steps != list(range(1, state["step"] + 1)):
                raise RuntimeError("Metrics log does not end exactly at the restored checkpoint")
            write_json(output / f"resume_verification_step_{state['step']:06d}.json", resume_proof, exclusive=True)

        write_json(output / "status.json", {**status, "phase": "training", "step": state["step"]})
        optimizer.zero_grad(set_to_none=True)
        process = None
        try:
            import psutil
            process = psutil.Process(os.getpid())
        except ImportError:
            pass
        execution_cap = min(stop_after_step or total_steps, total_steps)
        while state["step"] < execution_cap and state["epoch"] < training["epochs"]:
            order = state["order"]
            cursor = state["cursor"]
            loader = DataLoader(
                dataset,
                batch_size=training["micro_batch_size"],
                sampler=order[cursor:],
                num_workers=0,
                pin_memory=False,
                drop_last=False,
                collate_fn=collator,
                generator=torch.Generator().manual_seed(training["seed"] + state["epoch"]),
            )
            iterator = iter(loader)
            while state["cursor"] < len(order) and state["step"] < execution_cap:
                torch.cuda.synchronize()
                started = time.perf_counter()
                remaining = len(order) - state["cursor"]
                micro_batches_in_window = min(
                    training["gradient_accumulation_steps"],
                    math.ceil(remaining / training["micro_batch_size"]),
                )
                sample_ids, real_lengths, micro_losses = [], [], []
                real_tokens = 0
                supervised_tokens = 0
                before = {name: parameter.detach().cpu().clone() for name, parameter in trainable}
                for _ in range(micro_batches_in_window):
                    batch, labels, info = next(iterator)
                    count = len(info["sample_ids"])
                    expected_indices = order[state["cursor"]:state["cursor"] + count]
                    expected_ids = [dataset[index]["id"] for index in expected_indices]
                    if info["sample_ids"] != expected_ids:
                        raise RuntimeError("DataLoader sample order differs from the saved epoch permutation")
                    lengths = info["real_lengths"]
                    if max(lengths) > MAX_LENGTH:
                        raise RuntimeError("A batch exceeded 512 tokens after the frozen preflight")
                    sample_ids.extend(info["sample_ids"])
                    real_lengths.extend(lengths)
                    real_tokens += sum(lengths)
                    supervised_tokens += int((labels != -100).sum())
                    batch = {key: value.to("cuda:0") for key, value in batch.items()}
                    labels = labels.to("cuda:0")
                    result = model(**batch, labels=labels, use_cache=False)
                    loss = result.loss
                    if loss is None or not torch.isfinite(loss).item():
                        raise RuntimeError(f"Non-finite training loss at optimizer step {state['step'] + 1}")
                    micro_losses.append(float(loss.detach().item()))
                    (loss / micro_batches_in_window).backward()
                    state["cursor"] += count
                    state["micro_step"] += 1
                    del result, loss, batch, labels

                gradients = [parameter.grad for _, parameter in trainable]
                if any(gradient is None for gradient in gradients):
                    raise RuntimeError("At least one LoRA parameter has no gradient")
                gradient_norm = torch.linalg.vector_norm(torch.stack([
                    torch.linalg.vector_norm(gradient.float()) for gradient in gradients
                ]))
                if not torch.isfinite(gradient_norm).item() or gradient_norm.item() <= 0:
                    raise RuntimeError("LoRA gradient norm is zero or non-finite")
                learning_rate_used = optimizer.param_groups[0]["lr"]
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                changed = sum(
                    not torch.equal(before[name], parameter.detach().cpu())
                    for name, parameter in trainable
                )
                if changed == 0 and learning_rate_used != 0.0:
                    raise RuntimeError("LoRA parameters did not change after optimizer step")
                if any(not torch.isfinite(parameter).all().item() for _, parameter in trainable):
                    raise RuntimeError("Non-finite LoRA parameter after optimizer update")
                state["step"] += 1
                state["samples"] += len(sample_ids)
                state["real_tokens"] += real_tokens
                state["supervised_tokens"] += supervised_tokens
                state["maximum_actual_length"] = max(state["maximum_actual_length"], max(real_lengths))
                elapsed = time.perf_counter() - started
                record = {
                    "step": state["step"],
                    "epoch": state["epoch"],
                    "cursor": state["cursor"],
                    "micro_steps_in_window": micro_batches_in_window,
                    "sample_count": len(sample_ids),
                    "sample_ids": sample_ids,
                    "real_lengths": real_lengths,
                    "loss": sum(micro_losses) / len(micro_losses),
                    "micro_losses": micro_losses,
                    "lr_used": learning_rate_used,
                    "lr_next": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(gradient_norm.item()),
                    "changed_lora_tensors": changed,
                    "base_frozen": True,
                    "seconds": elapsed,
                    "real_input_tokens": real_tokens,
                    "supervised_tokens": supervised_tokens,
                    "tokens_per_second": real_tokens / elapsed,
                    "samples_per_second": len(sample_ids) / elapsed,
                    "gpu_allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                    "gpu_reserved_peak_bytes": torch.cuda.max_memory_reserved(),
                    "rss_bytes": process.memory_info().rss if process is not None else None,
                }
                append_jsonl(output / "logs" / "metrics.jsonl", record)
                at_epoch_end = state["cursor"] == len(order)
                if at_epoch_end:
                    state["epoch"] += 1
                    state["cursor"] = 0
                    state["order"] = (
                        torch.randperm(
                            len(dataset),
                            generator=torch.Generator().manual_seed(training["seed"] + state["epoch"]),
                        ).tolist()
                        if state["epoch"] < training["epochs"] else []
                    )
                stop_requested = stop_after_step is not None and state["step"] >= stop_after_step
                should_save = (
                    state["step"] % training["save_steps"] == 0
                    or state["step"] == total_steps
                    or stop_requested
                )
                if should_save:
                    checkpoint_path = save_checkpoint(
                        output, model, optimizer, scheduler, state, run_config
                    )
                    write_json(output / "status.json", {
                        **status, "phase": "checkpoint_saved", "step": state["step"],
                        "checkpoint": str(checkpoint_path),
                    })
                if state["step"] == 1 or state["step"] % 10 == 0 or should_save:
                    write_json(output / "status.json", {
                        **status, "phase": "training", "step": state["step"],
                        "total_steps": total_steps, "samples": state["samples"],
                    })
                    print(
                        f"[{profile_name}] step={state['step']}/{total_steps} "
                        f"loss={record['loss']:.6f} lr={learning_rate_used:.8g} "
                        f"samples={state['samples']}",
                        flush=True,
                    )
                if stop_requested or state["step"] >= execution_cap or at_epoch_end:
                    break
            if stop_after_step is not None and state["step"] >= stop_after_step:
                break

        paused = state["step"] < total_steps
        if not paused:
            if state["samples"] != len(dataset) or state["epoch"] != training["epochs"]:
                raise RuntimeError("Training did not complete exactly one full epoch")
            adapter_dir = output / "adapter"
            model.save_pretrained(adapter_dir, safe_serialization=True)
        summary = {
            "status": "PAUSED" if paused else "PASS",
            "profile": profile_name,
            "steps_completed": state["step"],
            "total_steps": total_steps,
            "samples_seen": state["samples"],
            "training_rows": len(dataset),
            "epochs_completed": state["epoch"],
            "real_input_tokens": state["real_tokens"],
            "supervised_tokens": state["supervised_tokens"],
            "maximum_actual_length": state["maximum_actual_length"],
            "base_frozen": True,
            "method": "BF16 LoRA",
            "gpu": gpu,
            "adapter_saved": not paused,
            "evaluation_pending": not paused,
            "oom": False,
        }
        write_json(output / "training_result.json", summary, exclusive=True)
        write_json(output / "status.json", {
            **status, "status": summary["status"], "phase": "complete",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "step": state["step"],
        })
        return summary
    except BaseException as exc:
        write_json(output / f"failure_{os.getpid()}.json", {
            "status": "FAIL",
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "oom": isinstance(exc, torch.cuda.OutOfMemoryError),
        })
        write_json(output / "status.json", {
            **status, "status": "FAIL", "phase": "failed",
            "error": str(exc),
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise


def build_parser(profile_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Pinned MedMCQA-only BF16 LoRA training entry: {profile_name}"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true", help="CPU-only checks; never load model weights or train.")
    mode.add_argument("--run", action="store_true", help="Start/resume GPU training after every safety gate passes.")
    parser.add_argument("--model-path", type=Path, default=None, help="Optional path; must resolve to the pinned snapshot.")
    parser.add_argument("--confirm-full-training", action="store_true", help="Required explicit confirmation before GPU training.")
    parser.add_argument("--resume-from", type=Path, default=None, help="Resume from a verified optimizer-boundary checkpoint.")
    parser.add_argument("--stop-after-step", type=int, default=None, help="Optional controlled pause at an optimizer-step boundary.")
    return parser


def main(profile_name: str) -> None:
    parser = build_parser(profile_name)
    args = parser.parse_args()
    if args.run and not args.confirm_full_training:
        parser.error("--run requires --confirm-full-training")
    if args.preflight and (args.resume_from is not None or args.stop_after_step is not None or args.confirm_full_training):
        parser.error("--preflight cannot be combined with run/resume/confirmation options")
    if args.resume_from is not None and not args.run:
        parser.error("--resume-from requires --run")
    if args.stop_after_step is not None and not args.run:
        parser.error("--stop-after-step requires --run")
    profile = load_profile(profile_name)
    model_path = resolve_model_path(profile, args.model_path)
    preflight = run_preflight(profile_name, model_path)
    report_path = PREFLIGHT_OUTPUT_ROOT / profile_name / "preflight.json"
    write_json(report_path, preflight)
    print(json.dumps(preflight, indent=2, ensure_ascii=False, default=str), flush=True)
    print(f"Preflight report: {report_path}", flush=True)
    if args.preflight:
        if not preflight["ready_to_train"]:
            print("CPU preflight passed, but the current environment does not match the pinned training runtime.", flush=True)
        else:
            print("PASS: ready for the separately confirmed GPU training command; no training was started.", flush=True)
        return
    if not preflight["ready_to_train"]:
        raise RuntimeError("Training is blocked by a runtime version mismatch; no model weights were loaded")
    result = run_training(
        profile_name, profile, preflight,
        resume_from=args.resume_from,
        stop_after_step=args.stop_after_step,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["qwen25_3b", "qwen35_4b"], required=True)
    main(parser.parse_args().profile)
