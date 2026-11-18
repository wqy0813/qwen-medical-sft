"""Independent final cleaner. Default=preflight; bounded sample is not a release.

No model weights, training, network, source rewriting or legacy classifiers.
Full execution requires both --mode full and --confirm-full-run 173648.
Do not use full mode until the user approves the development report.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from collections import Counter
from pathlib import Path

from medical_sft.data.medmcqa_final_rules import (VERSION, apply_dataset_checks, classify_record, content_from_raw,
                                 digest, parse_answer, summarize)
from medical_sft.data.medmcqa_training_contract import (INSTRUCTION, MAX_LENGTH, MODEL_ID, MODEL_REVISION,
                                      TEMPLATE_KWARGS, check_length, load_local_processor, training_record)
from medical_sft.evaluation.medmcqa_evaluation import evaluation_messages

from medical_sft.paths import ROOT
EXPECTED = {"train_scope": 173648, "raw.train": 182822, "dev_eval": 9139}


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL record: {path}:{line_number}")
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"].strip():
                raise ValueError(f"Missing/invalid ID: {path}:{line_number}")
            yield row


def source_eval_prompt(raw):
    # Reconstruct the historical evaluation prompt; this is not a training target.
    return str(raw["question"]).strip()+"\n"+"\n".join(
        f"{l}. {str(raw[f'op{l.lower()}']).strip()}" for l in "ABCD"
    )+"\nReturn exactly one line in the format: Answer: X"


def resolve_input(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest input leaves project root")
    return path


def load_inputs(root=ROOT, sample_limit=0, include_ids=()):
    """Validate the whole input boundary, but only return selected raw training rows.

    Scanning raw IDs/counts for provenance is not classification or full cleaning.
    Historical rule dependency hashes/quarantines are deliberately not consulted.
    """
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    manifest_path = root / "data_release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("schema_version") != 1 or manifest.get("active_evaluation") != "dev_eval" or
        manifest.get("reserved_evaluation") != "reserved" or
        manifest.get("dataset", {}).get("id") != "openlifescienceai/medmcqa"):
        raise ValueError("Unexpected data release manifest or evaluation roles")
    paths = {}
    for role, count in EXPECTED.items():
        entry = manifest["artifacts"][role]
        if entry["rows"] != count:
            raise ValueError(f"Unexpected input count in release manifest: {role}")
        path = resolve_input(root, entry["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[role] = path
    train_scope = list(read_jsonl(paths["train_scope"]))
    dev = list(read_jsonl(paths["dev_eval"]))
    for role, rows in (("train_scope", train_scope), ("dev_eval", dev)):
        if len(rows) != EXPECTED[role] or len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Record count/ID uniqueness mismatch: {role}")
    tids, dids = {r["id"] for r in train_scope}, {r["id"] for r in dev}
    if tids & dids:
        raise ValueError("Train/dev ID intersection")
    for row in dev:
        evaluation_messages(row)  # Lossless adapter; no second evaluation data file.
    if not set(include_ids) <= tids:
        raise ValueError("Requested sample IDs leave the authorized training scope")
    selected_ids = tids if sample_limit == 0 else {r["id"] for r in train_scope[:sample_limit]} | set(include_ids)
    wanted = selected_ids | dids
    raw_map, source_lines, raw_ids = {}, {}, set()
    raw_count = 0
    parquet = pq.ParquetFile(paths["raw.train"])
    required = {"id", "question", "opa", "opb", "opc", "opd", "cop", "exp", "choice_type"}
    if not required <= set(parquet.schema_arrow.names):
        raise ValueError("Raw schema is missing required fields")
    for batch in parquet.iter_batches(batch_size=4096):
        for raw in batch.to_pylist():
            raw_count += 1
            sid = raw["id"]
            if not isinstance(sid, str) or not sid.strip() or sid in raw_ids:
                raise ValueError("Raw IDs cannot be mapped one-to-one")
            raw_ids.add(sid)
            if sid in wanted:
                raw_map[sid], source_lines[sid] = raw, raw_count
    if raw_count != EXPECTED["raw.train"] or not (tids | dids) <= raw_ids:
        raise ValueError("Raw count/mapping mismatch")
    omitted = raw_ids - tids - dids
    omission_rows = manifest.get("documented_omissions", [])
    omission_ids = [row.get("id") for row in omission_rows]
    if (len(omitted) != len(omission_rows) or len(omission_ids) != len(set(omission_ids)) or
        set(omission_ids) != omitted):
        raise ValueError("Raw omissions do not match documented IDs in data_release_manifest.json")
    selected = []
    for i, scope_row in enumerate(train_scope, 1):
        sid = scope_row["id"]
        if sid not in selected_ids:
            continue
        raw = raw_map[sid]
        selected.append((i, raw, dict(raw_path=manifest["artifacts"]["raw.train"]["path"],
                                     raw_row=source_lines[sid], scope_path=manifest["artifacts"]["train_scope"]["path"],
                                     )))
    for old in dev:
        raw = raw_map[old["id"]]
        if (parse_answer(raw.get("cop"))[0] != "valid" or old["prompt"] != source_eval_prompt(raw) or
            parse_answer(old.get("gold")) != parse_answer(raw.get("cop"))):
            raise ValueError(f"Dev content lineage mismatch: {old['id']}")
    evidence = dict(expected_counts=EXPECTED, actual_counts=dict(train_scope=len(train_scope), raw_train=raw_count,
                    dev_eval=len(dev), documented_omissions=len(omitted)),
                    selected_count=len(selected), selected_ids=[raw["id"] for _, raw, _ in selected],
                    dev_eval_only=True, lineage_passed=True,
                    source_version=manifest["dataset"]["source_version"],
                    documented_omission_reasons=dict(Counter(row["reason"] for row in omission_rows)),
                    fields=dict(id="id", question="question", options=["opa", "opb", "opc", "opd"],
                                gold="cop", explanation="exp"))
    return selected, [raw_map[r["id"]] for r in dev], evidence


def process_records(selected, dev, processor):
    records = []
    for i, raw, source in selected:
        result = classify_record(raw, i, source)
        if result["quality_class"] in ("A", "B"):
            result.update(check_length(processor, result))
        records.append(result)
    groups = apply_dataset_checks(records, dev)
    report = summarize(records, groups)
    return records, groups, report


def provenance_metadata(root, snapshot):
    versions = {}
    for package in ("transformers", "tokenizers", "torch", "pyarrow"):
        versions[package] = importlib.metadata.version(package)
    return dict(rules_version=VERSION, packages=versions, model_id=MODEL_ID,
                model_revision=MODEL_REVISION, instruction=INSTRUCTION,
                template_kwargs=TEMPLATE_KWARGS, max_length=MAX_LENGTH)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2)+"\n", encoding="utf-8")


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True, separators=(",", ":"))+"\n")


def write_outputs(output, records, groups, report):
    output.mkdir(parents=True, exist_ok=False)
    marker = output / "RUN_INCOMPLETE"
    marker.write_text("Do not consume outputs until this marker is removed.\n", encoding="utf-8")
    classes = {"A": "medmcqa_A.jsonl", "B": "medmcqa_B_cleaned.jsonl",
               "C": "medmcqa_C_review.jsonl", "D": "medmcqa_D_excluded.jsonl"}
    for c, name in classes.items():
        write_jsonl(output/name, (r for r in records if r["quality_class"] == c))
    write_jsonl(output/"medmcqa_AB_quality_candidates.jsonl", (r for r in records if r["quality_class"] in ("A", "B")))
    def candidate(r):
        return dict(r, **training_record(r))
    write_jsonl(output/"medmcqa_train_candidates_512.jsonl",
                (candidate(r) for r in records if r["training_status"] == "eligible_candidate"))
    write_jsonl(output/"medmcqa_over_512.jsonl", (r for r in records if r["length_status"] == "over_limit"))
    write_jsonl(output/"medmcqa_train_dev_overlap_isolated.jsonl", (r for r in records if "dev_eval_overlap" in r["duplicate_flags"]))
    write_jsonl(output/"medmcqa_duplicate_groups.jsonl", groups)
    write_jsonl(output/"medmcqa_repair_log.jsonl", (event for r in records for event in r["repairs"]))
    write_json(output/"medmcqa_cleaning_report.json", report)
    (output/"README.md").write_text(
        f"# MedMCQA {report['scope']}\n\n"
        f"Processed records: {len(records)}. Rules: {VERSION}. Training release ready: false.\n\n"
        "Original inputs are read-only. No model weights or training were used.\n"
        "Quality classes are disjoint; isolation sidecars may overlap. Use blocked_records_union, "
        "not the sum of reason counts. Different explanations for one consistent gold remain candidates.\n\n"
        "Sample duplicate checks cover only the recorded selected IDs; dev checks use all 9,139 active dev records. "
        "A sample is not a complete release. No medical correctness certification is implied.\n",
        encoding="utf-8")
    marker.unlink()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "sample", "full"), default="preflight")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--include-id", action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confirm-full-run", choices=("173648",))
    args = parser.parse_args(argv)
    if args.mode == "full":
        if args.confirm_full_run != "173648" or args.include_id or args.limit != 200:
            parser.error("Full mode requires explicit approval token; sample controls cannot be combined")
    elif args.confirm_full_run:
        parser.error("Full-run token is only valid with --mode full")
    if args.mode != "full" and not (1 <= args.limit <= 1000):
        parser.error("Development sample limit must be 1..1000")
    if args.mode != "preflight" and args.output is None:
        parser.error("An explicit new output directory is required")
    return args


def main(argv=None):
    args = parse_args(argv)
    output = args.output.resolve() if args.output else None
    if output:
        if not output.is_relative_to(ROOT) or output == ROOT or output.exists():
            raise ValueError("Output must be a new directory inside the project")
        for forbidden in (ROOT/"data/raw", ROOT/"data/processed", ROOT/".git", ROOT/".venv"):
            if output.is_relative_to(forbidden):
                raise ValueError("Output cannot enter protected source/code infrastructure")
    limit = 0 if args.mode == "full" else args.limit
    selected, dev, inputs = load_inputs(sample_limit=limit, include_ids=args.include_id)
    if args.mode != "full" and len(selected) > 1000:
        raise ValueError("Selected development sample exceeds 1,000 records")
    if args.mode == "preflight":
        report = dict(scope="preflight_only", inputs=inputs, classified_records=0,
                      training_release_ready=False, training_started=False)
        if output:
            output.mkdir(parents=True, exist_ok=False)
            write_json(output/"preflight_report.json", report)
        print(json.dumps(dict(scope=report["scope"], input_boundary_passed=True,
                              classified_records=0, selected_count=len(selected))))
        return
    processor, snapshot = load_local_processor()
    records, groups, report = process_records(selected, dev, processor)
    report.update(scope="full_cleaning" if args.mode == "full" else "development_sample",
                  inputs=inputs, provenance=provenance_metadata(ROOT, snapshot),
                  training_started=False)
    write_outputs(output, records, groups, report)
    print(json.dumps(dict(scope=report["scope"], records=len(records), quality=report["quality_counts"],
                          training_status=report["training_status_counts"], output=str(output), training_started=False)))


if __name__ == "__main__":
    main()
