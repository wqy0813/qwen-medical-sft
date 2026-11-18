"""Run the repository's data-independent or full unittest suite.

Usage:
    python scripts/run_tests.py                 # public, data-independent tests
    python scripts/run_tests.py public
    python scripts/run_tests.py data            # requires restored frozen datasets
    python scripts/run_tests.py all
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = ROOT / "tests"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS_DIR))

PUBLIC_MODULES = (
    "test_medmcqa_final",
    "test_medmcqa_evaluation",
    "test_model_specific_verification",
    "test_qwen25_instruct_baseline_protocol",
)

PUBLIC_CASES = (
    "test_current_medmcqa_training.CurrentMedMCQATrainingTests.test_locked_profiles",
    "test_current_medmcqa_training.CurrentMedMCQATrainingTests.test_record_validation",
    "test_current_medmcqa_training.CurrentMedMCQATrainingTests.test_source_ids_must_belong_to_initial_train_scope",
    "test_current_medmcqa_training.CurrentMedMCQATrainingTests.test_record_validation_rejects_medexpqa_and_overlength",
    "test_current_medmcqa_training.CurrentMedMCQATrainingTests.test_model_snapshot_paths_are_pinned",
    "test_medexpqa_evaluate.EvaluationTests.test_parsing_and_option_boundaries",
    "test_medexpqa_evaluate.EvaluationTests.test_metrics_and_matrices",
    "test_package_layout.PackageLayoutTests.test_project_paths_resolve_from_package",
    "test_package_layout.PackageLayoutTests.test_module_help_from_outside_project",
)

DATA_CASES = (
    "test_current_medmcqa_training.CurrentMedMCQATrainingTests.test_initial_train_scope_is_hash_locked_and_has_173648_ids",
    "test_current_medmcqa_training.CurrentMedMCQATrainingTests.test_final_evaluation_protocol_is_locked_and_data_hashes_match",
    "test_medexpqa_evaluate.EvaluationTests.test_real_data_and_original_order",
    "test_medexpqa_evaluation_protocol.MedExpQAFullEvaluationTests.test_all_618_records_are_loaded_with_unique_ids",
    "test_medexpqa_evaluation_protocol.MedExpQAFullEvaluationTests.test_official_split_composition_is_preserved",
    "test_medexpqa_evaluation_protocol.MedExpQAFullEvaluationTests.test_all_train_dispositions_are_included",
    "test_medexpqa_evaluation_protocol.MedExpQAFullEvaluationTests.test_each_record_has_a_valid_question_gold_and_option_set",
    "test_medexpqa_evaluation_protocol.MedExpQAFullEvaluationTests.test_audit_only_checks_the_two_final_medmcqa_training_files",
    "test_medexpqa_evaluation_protocol.MedExpQAFullEvaluationTests.test_audit_covers_all_618_records_and_checks_final_training_inputs",
    "test_package_layout.PackageLayoutTests.test_existing_training_data_is_not_overwritten",
    "test_package_layout.PackageLayoutTests.test_two_medmcqa_splits_and_medexpqa_can_be_loaded",
)

EXPECTED_COUNTS = {"public": 106, "data": 11, "all": 117}


def named_suite(names: tuple[str, ...]) -> unittest.TestSuite:
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for name in names:
        suite.addTests(loader.loadTestsFromName(name))
    return suite


def build_suite(group: str) -> unittest.TestSuite:
    if group == "all":
        return unittest.defaultTestLoader.discover(
            start_dir=str(TESTS_DIR),
            pattern="test_*.py",
        )

    suite = unittest.TestSuite()
    if group == "public":
        for module_name in PUBLIC_MODULES:
            module = importlib.import_module(module_name)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        suite.addTests(named_suite(PUBLIC_CASES))
    elif group == "data":
        suite.addTests(named_suite(DATA_CASES))
    return suite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "group",
        nargs="?",
        choices=("public", "data", "all"),
        default="public",
        help="public: no frozen dataset files; data: requires restored datasets; all: all tests",
    )
    args = parser.parse_args()

    suite = build_suite(args.group)
    actual = suite.countTestCases()
    expected = EXPECTED_COUNTS[args.group]
    if actual != expected:
        parser.error(
            f"{args.group} suite contains {actual} tests; expected {expected}. "
            "Update the explicit test grouping after reviewing the changed test inventory."
        )

    print(f"Running {args.group} suite: {actual} tests", flush=True)
    if args.group == "data":
        print("This suite requires the frozen data files described in data/README.md.", flush=True)
    elif args.group == "public":
        print("This suite does not require restored dataset bodies, model weights, or outputs/.", flush=True)

    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
