"""Build the Qwen2.5 case explorer from the published MedExpQA outputs."""

from collections import Counter
import json
from shutil import copyfile

from rescore_qwen25_medexpqa import PREDICTIONS, ROOT, parse_output, score


SITE = ROOT / "site"
OUTPUT = ROOT / "outputs" / "qwen25-pages"


def outcome(base, sft, gold):
    if base == gold:
        return "both_correct" if sft == gold else "regressed"
    if sft == gold:
        return "format" if base is None else "fixed"
    return "both_missed"


def main():
    featured = {
        case["id"]: case for case in json.loads((SITE / "featured_cases.json").read_text(encoding="utf-8"))
    }
    rows = []
    predictions = {"base": [], "sft": []}
    with PREDICTIONS.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            for version in predictions:
                parsed = parse_output(row, f"{version}_output")
                predictions[version].append(parsed)
                row[f"{version}_answer"] = parsed["normalized_prediction"]
            row["outcome"] = outcome(row["base_answer"], row["sft_answer"], row["gold"])
            example = featured.get(row["id"])
            if example:
                if "".join(example["options"]) != row["allowed_options"] or example["gold"] != row["gold"]:
                    raise ValueError(f"Featured question {row['id']} does not match the prediction record")
                row.update(question=example["question"], options=example["options"])
            rows.append(row)

    featured_rows = {row["outcome"]: row["id"] for row in rows if "question" in row}
    if set(featured_rows) != {"fixed", "format", "regressed"} or len(featured) != 3:
        raise ValueError("Include one featured question for each of the three changes")
    featured_order = {question_id: index for index, question_id in enumerate(featured)}
    rows.sort(key=lambda row: (featured_order.get(row["id"], len(featured)), row["id"]))
    scores = {version: score(values)[0]["overall"] for version, values in predictions.items()}
    data = {
        "metrics": scores,
        "counts": dict(Counter(row["outcome"] for row in rows)),
        "featured": featured_rows,
        "rows": rows,
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    for asset in ("index.html", "styles.css", "app.js"):
        copyfile(SITE / asset, OUTPUT / asset)
    (OUTPUT / "data.json").write_text(
        json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    print(f"Built {len(rows)} paired results in {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
