"""Recompute full Chat Template token lengths for frozen model-specific MedMCQA releases.

This is a CPU/tokenizer-only preflight. It never loads model weights, truncates
samples, changes training JSONL, or downloads assets.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from medical_sft.paths import ROOT
REPORT = ROOT / "data/processed/train/model_specific/model_specific_build_report.json"
MAX_LENGTH = 512

SPECS = {
    "qwen25_3b": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "data_path": "data/processed/train/model_specific/qwen25_3b/medmcqa_train.jsonl",
        "loader": "tokenizer",
    },
    "qwen35_4b": {
        "model_id": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "data_path": "data/processed/train/model_specific/qwen35_4b/medmcqa_train.jsonl",
        "loader": "processor",
    },
}

def load_model_tokenizer(model_name: str, model_path: Path, spec: dict[str, Any]):
    if spec["loader"] == "processor":
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False
        )
        if not getattr(processor, "chat_template", None):
            raise ValueError("Qwen3.5 processor has no chat template")
        if not hasattr(processor, "tokenizer"):
            raise ValueError("Qwen3.5 processor does not expose its tokenizer")
        return processor, processor.tokenizer, "AutoProcessor"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=False
    )
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Qwen2.5 Instruct tokenizer has no chat template")
    return tokenizer, tokenizer, "AutoTokenizer"


def normalize_input_ids(encoded: Any) -> list[list[int]]:
    ids = encoded["input_ids"] if isinstance(encoded, dict) or hasattr(encoded, "__getitem__") else encoded
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if not isinstance(ids, list):
        ids = list(ids)
    if ids and isinstance(ids[0], int):
        return [ids]
    return [list(row) for row in ids]


def apply_templates(model_name: str, engine: Any, conversations: list[list[dict[str, str]]]):
    spec = SPECS[model_name]
    if spec["loader"] == "processor":
        full = engine.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            enable_thinking=False,
            processor_kwargs={"padding": False, "truncation": False},
        )
        prompts = engine.apply_chat_template(
            [messages[:-1] for messages in conversations],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            enable_thinking=False,
            processor_kwargs={"padding": False, "truncation": False},
        )
    else:
        full = engine.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            padding=False,
            truncation=False,
        )
        prompts = engine.apply_chat_template(
            [messages[:-1] for messages in conversations],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            padding=False,
            truncation=False,
        )
    return normalize_input_ids(full), normalize_input_ids(prompts)


def validate_row(row: dict[str, Any], spec: dict[str, Any], line_number: int) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"Expected exactly two chat messages at line {line_number}")
    if messages[0].get("role") != "user" or messages[1].get("role") != "assistant":
        raise ValueError(f"Unexpected message roles at line {line_number}")
    expected_target = f"Answer: {row.get('gold_answer')}\nExplanation: {row.get('explanation')}"
    if messages[1].get("content") != expected_target:
        raise ValueError(f"Answer+Explanation target drift at line {line_number}")
    protocol = row.get("training_protocol")
    if not isinstance(protocol, dict):
        raise ValueError(f"Missing training protocol at line {line_number}")
    for key, expected in (
        ("model_id", spec["model_id"]),
        ("revision", spec["revision"]),
        ("max_length", MAX_LENGTH),
    ):
        if protocol.get(key) != expected:
            raise ValueError(f"Training protocol {key} mismatch at line {line_number}")
    if spec["loader"] == "processor" and protocol.get("enable_thinking") is not False:
        raise ValueError(f"Qwen3.5 thinking must be explicitly disabled at line {line_number}")
    if spec["loader"] == "tokenizer" and protocol.get("truncation") is not False:
        raise ValueError(f"Qwen2.5 truncation must be explicitly disabled at line {line_number}")


def run_preflight(model_name: str, model_path: Path, batch_size: int, output_root: Path) -> dict[str, Any]:
    spec = SPECS[model_name]
    release = json.loads(REPORT.read_text(encoding="utf-8"))
    if release.get("status") != "PASS_METADATA_AND_STRUCTURE":
        raise ValueError("Model-specific release report is not in the expected verified state")
    data_info = release["models"][model_name]
    data_path = ROOT / spec["data_path"]
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if data_info["rows"] != release["source_rows"]:
        raise ValueError("Model-specific row count differs from the common release")

    engine, tokenizer, loader_name = load_model_tokenizer(model_name, model_path, spec)
    output_dir = output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=False)

    rows_seen = 0
    max_full = 0
    max_prompt = 0
    min_target = None
    overlength = []
    mismatches = []
    length_bins = Counter()
    max_examples = []
    with data_path.open("rb") as handle, (output_dir / "overlength.jsonl").open("x", encoding="utf-8") as over_file, (output_dir / "length_mismatches.jsonl").open("x", encoding="utf-8") as mismatch_file:
        while True:
            raw_batch = []
            for _ in range(batch_size):
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    raise ValueError(f"Blank JSONL line, near row {rows_seen + len(raw_batch) + 1}")
                raw_batch.append(json.loads(line.decode("utf-8")))
            if not raw_batch:
                break
            line_start = rows_seen + 1
            for offset, row in enumerate(raw_batch):
                validate_row(row, spec, line_start + offset)
            conversations = [row["messages"] for row in raw_batch]
            full_ids, prompt_ids = apply_templates(model_name, engine, conversations)
            if len(full_ids) != len(raw_batch) or len(prompt_ids) != len(raw_batch):
                raise ValueError(f"Tokenizer batch cardinality mismatch at row {line_start}")
            for offset, (row, full, prompt) in enumerate(zip(raw_batch, full_ids, prompt_ids)):
                line_number = line_start + offset
                full_len, prompt_len = len(full), len(prompt)
                target_len = full_len - prompt_len
                if target_len <= 0 or full[:prompt_len] != prompt:
                    raise ValueError(f"Chat-template prompt prefix mismatch at line {line_number}")
                actual = (full_len, prompt_len, target_len)
                recorded = (
                    row.get("full_token_length"),
                    row.get("prompt_token_length"),
                    row.get("target_token_length"),
                )
                if recorded != actual:
                    item = {
                        "id": row.get("id"),
                        "line": line_number,
                        "recorded": recorded,
                        "actual": actual,
                    }
                    mismatches.append(item)
                    mismatch_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                if full_len > MAX_LENGTH or prompt_len >= MAX_LENGTH:
                    item = {
                        "id": row.get("id"),
                        "full_tokens": full_len,
                        "prompt_tokens": prompt_len,
                        "target_tokens": target_len,
                    }
                    overlength.append(item)
                    over_file.write(json.dumps(item, ensure_ascii=False) + "\n")
                if full_len > max_full:
                    max_full = full_len
                    max_examples = [{
                        "id": row.get("id"),
                        "full_tokens": full_len,
                        "prompt_tokens": prompt_len,
                        "target_tokens": target_len,
                    }]
                elif full_len == max_full and len(max_examples) < 10:
                    max_examples.append({
                        "id": row.get("id"),
                        "full_tokens": full_len,
                        "prompt_tokens": prompt_len,
                        "target_tokens": target_len,
                    })
                max_prompt = max(max_prompt, prompt_len)
                min_target = target_len if min_target is None else min(min_target, target_len)
                bucket = "0-127" if full_len <= 127 else "128-255" if full_len <= 255 else "256-383" if full_len <= 383 else "384-512" if full_len <= 512 else ">512"
                length_bins[bucket] += 1
            rows_seen += len(raw_batch)
            if rows_seen % (batch_size * 80) == 0 or rows_seen == release["source_rows"]:
                print(f"[{model_name}] checked {rows_seen}/{release['source_rows']} rows; overlength={len(overlength)} metadata_mismatches={len(mismatches)}", flush=True)

    if rows_seen != data_info["rows"]:
        raise ValueError("Training data row count mismatch")
    if overlength and mismatches:
        status = "OVER_LENGTH_AND_METADATA_MISMATCH"
    elif overlength:
        status = "OVER_LENGTH"
    elif mismatches:
        status = "METADATA_MISMATCH"
    else:
        status = "PASS"

    result = {
        "status": status,
        "model_id": spec["model_id"],
        "revision": spec["revision"],
        "model_path": str(model_path),
        "loader": loader_name,
        "tokenizer_class": type(tokenizer).__name__,
        "transformers_version": __import__("transformers").__version__,
        "data_path": spec["data_path"],
        "rows": rows_seen,
        "max_length": MAX_LENGTH,
        "truncation": False,
        "enable_thinking": False if spec["loader"] == "processor" else None,
        "lengths": {
            "max_full_tokens": max_full,
            "max_prompt_tokens": max_prompt,
            "min_target_tokens": min_target,
            "overlength_count": len(overlength),
            "metadata_mismatch_count": len(mismatches),
            "length_histogram": dict(length_bins),
            "max_length_examples": max_examples,
        },
        "outputs": {
            "overlength": "overlength.jsonl",
            "metadata_mismatches": "length_mismatches.jsonl",
        },
    }
    (output_dir / "length_preflight.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(SPECS), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "preflight" / "model_specific",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.batch_size > 512:
        parser.error("--batch-size must be between 1 and 512")
    spec = SPECS[args.model]
    if args.model_path is None:
        if args.model == "qwen35_4b":
            config = json.loads((ROOT / "configs/medmcqa_only_training/qwen35_4b.json").read_text(encoding="utf-8"))
            model_path = (ROOT / config["model_path_relative"]).resolve()
        else:
            parser.error("--model-path is required for qwen25_3b; an exact Instruct revision snapshot must be supplied")
    else:
        model_path = args.model_path.resolve()
    if not model_path.is_dir():
        parser.error(f"Model tokenizer snapshot directory does not exist: {model_path}")
    output_root = args.output_root.resolve()
    if (output_root / args.model).exists():
        parser.error(f"Refusing to overwrite existing preflight output: {output_root / args.model}")
    run_preflight(args.model, model_path, args.batch_size, output_root)


if __name__ == "__main__":
    main()
