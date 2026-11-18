"""Prepared Qwen2.5-3B-Instruct Base runner for the final Answer + Explanation prompt.

Unlike the historical 2026-11-05 run, this runner replaces the source
Answer-only suffix with the exact Answer + Explanation instruction used by
final SFT training. It is guarded by explicit evaluation approval and is not
run by preflight or unit tests.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from collections import Counter

from medical_sft.paths import ROOT
OUT = ROOT / "outputs/qwen25_3b_instruct_baseline_matched_20261110"
MODEL = ROOT / "models/Qwen2.5-3B-Instruct-aa8e72537993"
REV = "aa8e72537993ba99e69dfaafa59ed015b17504d1"
PARSER_SHA = "d1a1e0dbb9fd5ed6c614954989829546fdf2ae6ad90134f964d5e86b2e6f73c4"
BASE_CONFIG_SHA = "d76e8151ef9e5abcd168eabe8416579816ca7615561d37c2481feb56abd84a2f"
AUDIT_DIR = ROOT / "outputs/answer_format_audit_20261105"
BASE_REFERENCE_DIR = ROOT / "outputs/qwen25_3b_baseline_validation_20261105"
DO_SAMPLE = False
MAX_NEW_TOKENS = 32
USE_CACHE = True
OUTPUT_CREATED = False


def sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write(name: str, value: object) -> None:
    path = OUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    global OUTPUT_CREATED
    if os.environ.get("ALLOW_APPROVED_GPU_EVALUATION") != "1":
        raise RuntimeError(
            "GPU evaluation is disabled unless ALLOW_APPROVED_GPU_EVALUATION=1 is explicitly approved"
        )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from medical_sft.evaluation.medmcqa_evaluation import (
        final_sft_user_message,
        load_evaluation,
        parse_model_answer,
        score_predictions,
    )
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite matched baseline output: {OUT}")
    if not AUDIT_DIR.is_dir():
        raise FileNotFoundError(AUDIT_DIR)
    sys.path.insert(0, str(AUDIT_DIR))
    import audit

    parser_path = AUDIT_DIR / "audit.py"
    if sha(parser_path) != PARSER_SHA:
        raise ValueError("Answer-format audit parser SHA256 mismatch")
    audit.selftest()

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise ValueError("Expected CUDA_VISIBLE_DEVICES=1")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise ValueError("Matched baseline must run fully offline")
    if torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Expected one visible BF16-capable GPU")
    torch.set_num_threads(2)

    manifest_path = MODEL / "verified_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("revision") != REV or manifest.get("model") != "Qwen/Qwen2.5-3B-Instruct":
        raise ValueError("Pinned Qwen2.5-3B-Instruct snapshot identity mismatch")
    for name, entry in manifest["files"].items():
        path = MODEL / name
        if not path.is_file() or path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError(f"Model snapshot integrity failure: {name}")

    rows, info = load_evaluation("validation")
    if len(rows) != 4183 or sha(info["input_path"]) != "2710c733da35fa98bb7c7ea97b755cb1ce741ba26cbfe015d75ab94bf2c7089b":
        raise ValueError("Official Validation source identity mismatch")
    for row in rows:
        final_sft_user_message(row)
        audit.options(row["prompt"])

    reference_config_path = BASE_REFERENCE_DIR / "config.json"
    if sha(reference_config_path) != BASE_CONFIG_SHA:
        raise ValueError("Historical Base run reference config SHA256 mismatch")
    reference_config = json.loads(reference_config_path.read_text(encoding="utf-8"))
    if reference_config.get("data_sha256") != sha(info["input_path"]) or reference_config.get("n") != len(rows):
        raise ValueError("Historical Base run reference points to a different evaluation set")
    reference_predictions = [
        json.loads(line)
        for line in (BASE_REFERENCE_DIR / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(reference_predictions) != len(rows) or any(
        old["id"] != row["id"] or old["gold"] != row["gold"]
        for old, row in zip(reference_predictions, rows)
    ):
        raise ValueError("Historical Base reference IDs/Gold do not match the locked evaluation set")

    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 12 * 1024**3:
        raise RuntimeError(f"GPU free memory {free_bytes} bytes < 12 GiB")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False)
    if not tokenizer.chat_template:
        raise ValueError("Pinned Instruct tokenizer has no chat template")

    # Create a new, dedicated output directory. The historical 2026-11-05
    # predictions and metrics are never overwritten.
    OUT.mkdir(parents=True, exist_ok=False)
    OUTPUT_CREATED = True
    write("model_manifest.json", manifest)
    write(
        "config.json",
        {
            "status": "matched_prompt_runner",
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "revision": REV,
            "dtype": "bfloat16",
            "validation_sha256": sha(info["input_path"]),
            "rows": len(rows),
            "prompt_policy": "reconstruct final Answer + Explanation prompt; replace legacy Answer-only suffix; no extra single-letter suffix",
            "prompt_template": "{question_and_options}\\n{FINAL_SFT_INSTRUCTION}",
            "serialization": "Instruct tokenizer.apply_chat_template(user content, add_generation_prompt=True)",
            "chat_template": tokenizer.chat_template,
            "do_sample": DO_SAMPLE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "use_cache": USE_CACHE,
            "adapter": None,
            "strict_parser": "medmcqa_evaluation.parse_model_answer",
            "normalized_parser": "outputs/answer_format_audit_20261105/audit.py classify + options",
            "normalized_parser_sha256": PARSER_SHA,
            "offline": True,
            "truncation": False,
            "comparison_rule": "Use final_sft_user_message(row) identically for Base and SFT, with the same model-family chat template, generation settings, and parsers.",
        },
    )
    write("status.json", {"status": "RUNNING", "phase": "loading", "pid": os.getpid()})
    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
    ).to("cuda:0")
    if not all(parameter.dtype == torch.bfloat16 for parameter in model.parameters() if parameter.is_floating_point()):
        raise ValueError("Model contains non-BF16 floating point parameters")
    model.requires_grad_(False).eval()

    records = []
    start = time.perf_counter()
    with (OUT / "predictions.jsonl").open("x", encoding="utf-8") as handle, torch.inference_mode():
        for index, row in enumerate(rows):
            inputs = tokenizer.apply_chat_template(
                [final_sft_user_message(row)],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to("cuda:0")
            prompt_tokens = inputs["input_ids"].shape[-1]
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=DO_SAMPLE,
                use_cache=USE_CACHE,
            )
            raw = tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True).strip()
            strict = parse_model_answer(raw)
            category, normalized, reason = audit.classify(raw, audit.options(row["prompt"]))
            record = {
                "id": row["id"],
                "gold": row["gold"],
                "raw_output": raw,
                "strict_prediction": strict,
                "category": category,
                "normalized_prediction": normalized,
                "reason": reason,
                "strict_correct": strict == row["gold"],
                "normalized_correct": normalized == row["gold"],
                "prompt_tokens": prompt_tokens,
            }
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if (index + 1) % 32 == 0:
                handle.flush()
                write("status.json", {"status": "RUNNING", "phase": "evaluation", "completed": index + 1, "pid": os.getpid()})
                print(f"evaluation {index + 1}/{len(rows)}", flush=True)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    strict_metrics = score_predictions([row["gold"] for row in rows], [record["strict_prediction"] for record in records])
    normalized_metrics = score_predictions(
        [row["gold"] for row in rows], [record["normalized_prediction"] for record in records]
    )
    categories = Counter(record["category"] for record in records)
    if len(records) != 4183 or sum(categories.values()) != 4183:
        raise ValueError("Matched baseline prediction count mismatch")

    metrics = {
        "total": len(rows),
        "strict": {**strict_metrics, "invalid_rate": strict_metrics["invalid_predictions"] / len(rows)},
        "normalized": {
            **normalized_metrics,
            "invalid_rate": normalized_metrics["invalid_predictions"] / len(rows),
            "categories": {key: categories[key] for key in "ABCD"},
        },
        "inference_seconds": elapsed,
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    write("metrics.json", metrics)
    del model, tokenizer, inputs, output
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    report = f"""# Qwen2.5-3B-Instruct Base：匹配 Prompt 的 MedMCQA 官方 Validation

未经本项目医学 SFT，未加载 LoRA，也未更新参数。模型 revision：{REV}。

| 指标 | Strict | Normalized |
|---|---:|---:|
| Accuracy | {strict_metrics['accuracy']:.4%} | {normalized_metrics['accuracy']:.4%} |
| Correct / Total | {strict_metrics['correct']} / {len(rows)} | {normalized_metrics['correct']} / {len(rows)} |
| Invalid Predictions | {strict_metrics['invalid_predictions']} | {normalized_metrics['invalid_predictions']} |

**Prompt 与最终 SFT 训练对齐：**从锁定的官方 Validation 题目与选项中移除旧 Answer-only 尾部指令，替换为最终训练使用的 Answer + Explanation 指令；不追加单字母限制后缀。使用 Instruct tokenizer chat template、greedy generation、max_new_tokens={MAX_NEW_TOKENS}、use_cache=True，并同时保留严格解析与规范化解析结果。对应 SFT 评估必须复用同一 Prompt、生成参数和解析器。

官方 Validation 已用于此前模型选择，因此本结果不是盲测。原 2026-11-05 追加后缀版本保留为历史探索结果，不覆盖、不替换。
"""
    (OUT / "README.md").write_text(report, encoding="utf-8")
    write("status.json", {"status": "PASS", "phase": "complete", "completed": len(rows), "pid": os.getpid(), "model_released": True})
    print("COMPLETED", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if OUTPUT_CREATED and OUT.exists():
            write(
                "status.json",
                {
                    "status": "FAIL",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "automatic_restart": False,
                },
            )
        raise
