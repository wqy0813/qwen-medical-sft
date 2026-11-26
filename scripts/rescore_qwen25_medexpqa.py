"""Recompute the Qwen2.5 MedExpQA results from the published model outputs."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "release" / "qwen25-medexpqa" / "predictions.jsonl"
sys.path.insert(0, str(ROOT / "src"))

from medical_sft.evaluation.medexpqa_evaluation import parse, score


def parse_output(row, field):
    strict, normalized = parse(row[field], row["allowed_options"])
    return {
        "gold": row["gold"],
        "option_count": len(row["allowed_options"]),
        "strict_prediction": strict,
        "normalized_prediction": normalized,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--changes", type=Path, help="Save questions whose correctness changed to a new CSV file")
    args = parser.parse_args()

    results = {"Base": [], "SFT": []}
    changes = []
    with PREDICTIONS.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            base = parse_output(row, "base_output")
            sft = parse_output(row, "sft_output")
            results["Base"].append(base)
            results["SFT"].append(sft)

            gold = row["gold"]
            base_answer = base["normalized_prediction"]
            sft_answer = sft["normalized_prediction"]
            base_correct = base_answer == gold
            sft_correct = sft_answer == gold
            if base_correct != sft_correct:
                changes.append({
                    "id": row["id"],
                    "source_split": row["source_split"],
                    "gold": gold,
                    "base_answer": base_answer,
                    "sft_answer": sft_answer,
                    "change": "wrong_to_correct" if sft_correct else "correct_to_wrong",
                })

    for version, predictions in results.items():
        metrics, _ = score(predictions)
        overall = metrics["overall"]
        print(f"{version}: {overall['normalized_correct']}/{overall['n']} "
              f"({overall['normalized_accuracy']:.2%}), "
              f"{overall['invalid_count']} invalid outputs")

    counts = Counter(row["change"] for row in changes)
    print(f"Wrong to correct: {counts['wrong_to_correct']}")
    print(f"Correct to wrong: {counts['correct_to_wrong']}")

    if args.changes:
        fields = ["id", "source_split", "gold", "base_answer", "sft_answer", "change"]
        with args.changes.open("x", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            writer.writerows(changes)
        print(f"Saved {len(changes)} changed results to {args.changes}")


if __name__ == "__main__":
    main()
