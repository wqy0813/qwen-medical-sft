"""Build per-model MedMCQA SFT JSONL files from the frozen common candidate.

No cleaning, reordering, option changes, or tokenization happens here. The
common file already carries independently versioned length metadata for both
pinned models; this script promotes the matching metadata into each release.
"""
from __future__ import annotations

import json
from pathlib import Path

from medical_sft.paths import ROOT
SOURCE = ROOT / "data/processed/train/common/medmcqa_train_common_137928.jsonl"
EXPECTED_ROWS = 137_928
MAX_LENGTH = 512
MODELS = {
    "qwen25_3b": {
        "output": ROOT / "data/processed/train/model_specific/qwen25_3b/medmcqa_train.jsonl",
        "protocol_location": "training_protocol",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
        "extra_protocol": {"truncation": False},
    },
    "qwen35_4b": {
        "output": ROOT / "data/processed/train/model_specific/qwen35_4b/medmcqa_train.jsonl",
        "protocol_location": "source_training_metadata",
        "model_id": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "extra_protocol": {"enable_thinking": False},
    },
}
LENGTH_KEYS = ("full_token_length", "prompt_token_length", "target_token_length")


def main() -> None:
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    for spec in MODELS.values():
        if spec["output"].exists():
            raise FileExistsError(f"Refusing to overwrite model-specific data: {spec['output']}")

    handles = {}
    counts = {name: 0 for name in MODELS}
    max_lengths = {name: 0 for name in MODELS}
    ids_by_model = {name: set() for name in MODELS}
    try:
        for name, spec in MODELS.items():
            spec["output"].parent.mkdir(parents=True, exist_ok=True)
            handles[name] = spec["output"].open("x", encoding="utf-8", newline="\n")

        with SOURCE.open("r", encoding="utf-8") as src:
            for line_number, line in enumerate(src, 1):
                if not line.strip():
                    raise ValueError(f"Blank line in common candidate at line {line_number}")
                row = json.loads(line)
                if row.get("source") != "MedMCQA":
                    raise ValueError(f"Unexpected source at line {line_number}")
                if row.get("messages", [{} , {}])[-1].get("content") != (
                    f"Answer: {row.get('gold_answer')}\nExplanation: {row.get('explanation')}"
                ):
                    raise ValueError(f"Answer/explanation target mismatch at line {line_number}")

                for name, spec in MODELS.items():
                    model_row = dict(row)
                    if spec["protocol_location"] == "training_protocol":
                        length_source = row
                        protocol = row.get("training_protocol")
                    else:
                        metadata = row.get("source_training_metadata")
                        if not isinstance(metadata, dict):
                            raise ValueError(f"Missing source model metadata at line {line_number}")
                        length_source = metadata
                        protocol = metadata.get("training_protocol")
                    if not isinstance(protocol, dict):
                        raise ValueError(f"Missing {name} training protocol at line {line_number}")
                    if protocol.get("model_id") != spec["model_id"] or protocol.get("revision") != spec["revision"]:
                        raise ValueError(f"Unexpected {name} model revision at line {line_number}")
                    lengths = [length_source.get(key) for key in LENGTH_KEYS]
                    if any(type(v) is not int for v in lengths):
                        raise ValueError(f"Missing or invalid {name} lengths at line {line_number}")
                    full, prompt, target = lengths
                    if not (0 < full <= MAX_LENGTH and 0 < prompt < full and target > 0 and full == prompt + target):
                        raise ValueError(f"Invalid {name} token accounting at line {line_number}: {lengths}")
                    if row["id"] in ids_by_model[name]:
                        raise ValueError(f"Duplicate ID for {name}: {row['id']}")
                    ids_by_model[name].add(row["id"])
                    model_row.update(dict(zip(LENGTH_KEYS, lengths)))
                    model_row["training_protocol"] = {
                        "model_id": spec["model_id"],
                        "revision": spec["revision"],
                        "max_length": MAX_LENGTH,
                        **spec["extra_protocol"],
                    }
                    handles[name].write(json.dumps(model_row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts[name] += 1
                    max_lengths[name] = max(max_lengths[name], full)
    finally:
        for handle in handles.values():
            handle.close()

    if any(count != EXPECTED_ROWS for count in counts.values()):
        raise ValueError(f"Unexpected output counts: {counts}")
    if ids_by_model["qwen25_3b"] != ids_by_model["qwen35_4b"]:
        raise ValueError("Model-specific sample ID sets differ")

    report = {
        "status": "PASS_METADATA_AND_STRUCTURE",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_rows": EXPECTED_ROWS,
        "transformations": "Promote each model's already-recorded token lengths and protocol only; no content edits, filtering, or reordering.",
        "tokenizer_recomputed_this_build": False,
        "max_length": MAX_LENGTH,
        "models": {
            name: {
                "model_id": spec["model_id"],
                "revision": spec["revision"],
                "path": str(spec["output"].relative_to(ROOT)),
                "rows": counts[name],
                "max_full_token_length": max_lengths[name],
            }
            for name, spec in MODELS.items()
        },
        "same_sample_ids": True,
        "answer_explanation_preserved": True,
        "truncation": False,
    }
    report_path = ROOT / "data/processed/train/model_specific/model_specific_build_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite report: {report_path}")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    main()
