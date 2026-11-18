"""Final MedExpQA evaluation-only release loading and integrity checks."""
import collections
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medical_sft.evaluation import medexpqa_evaluation as evaluate


class MedExpQAFullEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.info = evaluate.load_full_evaluation()

    def test_all_618_records_are_loaded_with_unique_ids(self):
        self.assertEqual(self.info["purpose"], "evaluation_only")
        self.assertEqual(self.info["total_rows"], 618)
        self.assertEqual(self.info["unique_ids"], 618)
        self.assertEqual(len(self.rows), 618)
        self.assertEqual(len({str(row["id"]) for row in self.rows}), 618)

    def test_official_split_composition_is_preserved(self):
        counts = collections.Counter(row["source_split"] for row in self.rows)
        self.assertEqual(dict(counts), {"train": 430, "validation": 63, "test": 125})
        self.assertEqual(self.info["split_rows"], {"train": 430, "validation": 63, "test": 125})

    def test_all_train_dispositions_are_included(self):
        train_files = collections.Counter(
            row["source_file"] for row in self.rows if row["source_split"] == "train"
        )
        self.assertEqual(
            dict(train_files),
            {
                "medexpqa_train_candidates_512.jsonl": 378,
                "medexpqa_over_512_isolated.jsonl": 31,
                "medexpqa_quality_excluded.jsonl": 21,
            },
        )

    def test_each_record_has_a_valid_question_gold_and_option_set(self):
        for row in self.rows:
            with self.subTest(sample_id=row["id"]):
                self.assertTrue(row["question"].strip())
                self.assertIn(row["gold"], row["options"])
                self.assertIn(row["option_count"], (4, 5))
                self.assertEqual(row["option_count"], len(row["options"]))
                self.assertIsInstance(row["raw"], dict)
                self.assertGreaterEqual(row["line"], 1)
                self.assertTrue(row["source_file"])

    def test_audit_only_checks_the_two_final_medmcqa_training_files(self):
        self.assertEqual(
            evaluate.TRAIN_FILES,
            (
                "data/processed/train/model_specific/qwen25_3b/medmcqa_train.jsonl",
                "data/processed/train/model_specific/qwen35_4b/medmcqa_train.jsonl",
            ),
        )

    def test_audit_covers_all_618_records_and_checks_final_training_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            first = Path(temp) / "qwen25.jsonl"
            second = Path(temp) / "qwen35.jsonl"
            fixture = {
                "id": "MedMCQA:fixture-1",
                "source": "MedMCQA",
                "source_id": "fixture-1",
                "question": self.rows[0]["question"],
            }
            first.write_text(json.dumps(fixture) + "\n", encoding="utf-8")
            second.write_text(json.dumps({**fixture, "question": "A separate non-overlapping question?"}) + "\n", encoding="utf-8")
            with patch.object(evaluate, "TRAIN_FILES", (str(first), str(second))):
                report = evaluate.audit()

        self.assertEqual(report["total_rows"], 618)
        self.assertEqual(report["split_rows"], {"train": 430, "validation": 63, "test": 125})
        self.assertEqual(len(report["files"]), 5)
        self.assertEqual(len(report["training_checks"]), 2)
        self.assertEqual(report["training_checks"][0]["rows"], 1)
        self.assertEqual(len(report["training_checks"][0]["question_hits"]), 1)
        self.assertEqual(len(report["training_checks"][1]["question_hits"]), 0)


if __name__ == "__main__":
    unittest.main()
