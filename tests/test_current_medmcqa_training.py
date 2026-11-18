from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medical_sft.training import current_medmcqa_training as training


class CurrentMedMCQATrainingTests(unittest.TestCase):
    def test_locked_profiles(self):
        for profile_name in ("qwen25_3b", "qwen35_4b"):
            with self.subTest(profile=profile_name):
                profile = training.load_profile(profile_name)
                config = profile["training"]
                self.assertEqual(profile["rows"], 137_928)
                self.assertIn(profile["model_id"], {
                    "Qwen/Qwen2.5-3B-Instruct",
                    "Qwen/Qwen3.5-4B",
                })
                self.assertTrue(profile["data_path"].endswith("medmcqa_train.jsonl"))
                self.assertEqual((config["micro_batch_size"], config["gradient_accumulation_steps"],
                                  config["effective_batch_size"]), (2, 4, 8))
                self.assertEqual(config["learning_rate"], 5e-5)
                self.assertEqual((config["epochs"], config["seed"]), (1, 42))
                self.assertEqual((config["lora"]["r"], config["lora"]["alpha"]), (8, 16))
                self.assertFalse(config["truncation"])
                self.assertEqual(profile["max_length"], 512)
                self.assertEqual(training.expected_steps(profile), 17_241)
                self.assertEqual(config["warmup_steps"], 518)

    def _valid_row(self, profile):
        return {
            "id": "MedMCQA:sample-001", "source": "MedMCQA", "source_id": "sample-001",
            "quality_class": "A", "question": "Which option is correct?",
            "options": {"A": "one", "B": "two", "C": "three", "D": "four"},
            "original_gold": 0, "gold_index_base": 0, "gold_answer": "A",
            "explanation": "A is correct because it is the stated answer.",
            "messages": [
                {"role": "user", "content": "Question and options"},
                {"role": "assistant", "content": "Answer: A\nExplanation: A is correct because it is the stated answer."},
            ],
            "full_token_length": 20, "prompt_token_length": 10, "target_token_length": 10,
            "training_protocol": profile["training_protocol"], "provenance": {"source": "official_train"},
        }

    def test_record_validation(self):
        for profile_name in ("qwen25_3b", "qwen35_4b"):
            profile = training.load_profile(profile_name)
            training.validate_record(self._valid_row(profile), 1, profile)

    def test_source_ids_must_belong_to_initial_train_scope(self):
        training.assert_source_ids_in_train_scope(
            {"train-a", "train-b"},
            {"train-a", "train-b", "dev-a"},
        )
        with self.assertRaisesRegex(ValueError, "outside the original MedMCQA Train scope"):
            training.assert_source_ids_in_train_scope(
                {"train-a", "leaked-dev-id"},
                {"train-a", "dev-a"},
            )

    def test_initial_train_scope_is_hash_locked_and_has_173648_ids(self):
        ids, info = training.load_initial_train_scope_ids()
        self.assertEqual(len(ids), 173_648)
        self.assertEqual(info["rows"], 173_648)
        self.assertEqual(
            info["sha256"],
            "69a418ca2238d36eed36aa38d7a615a70e8a282a7f272fc9329cb38e04f3154c",
        )

    def test_final_evaluation_protocol_is_locked_and_data_hashes_match(self):
        protocol = training.verify_evaluation_protocol()
        self.assertEqual(protocol["initial_train_scope"]["rows"], 173_648)
        self.assertTrue(protocol["initial_train_scope"]["ids_disjoint_from_internal_validation"])
        internal = protocol["internal_validation"]
        self.assertEqual(internal["rows"], 9_139)
        self.assertEqual(internal["seed"], 20261020)
        self.assertEqual(internal["target_fraction"], 0.05)
        self.assertEqual(protocol["test_sets"][0]["rows"], 4_183)
        self.assertEqual(protocol["test_sets"][1]["rows"], 618)
        self.assertEqual(protocol["test_sets"][1]["unique_ids"], 618)
        self.assertEqual(protocol["test_sets"][1]["cross_dataset_id_overlap"], 0)
        self.assertEqual(
            protocol["test_sets"][1]["split_rows"],
            {"train": 430, "validation": 63, "test": 125},
        )

    def test_record_validation_rejects_medexpqa_and_overlength(self):
        profile = training.load_profile("qwen35_4b")
        row = self._valid_row(profile)
        row["source"] = "MedExpQA"
        with self.assertRaisesRegex(ValueError, "MedMCQA only"):
            training.validate_record(row, 1, profile)
        row["source"] = "MedMCQA"
        row["full_token_length"] = 513
        row["target_token_length"] = 503
        with self.assertRaisesRegex(ValueError, "overlength"):
            training.validate_record(row, 1, profile)

    def test_model_snapshot_paths_are_pinned(self):
        for name in ("qwen25_3b", "qwen35_4b"):
            profile = training.load_profile(name)
            with self.assertRaisesRegex(ValueError, "pinned snapshot"):
                training.resolve_model_path(profile, training.ROOT)



if __name__ == "__main__":
    unittest.main()
