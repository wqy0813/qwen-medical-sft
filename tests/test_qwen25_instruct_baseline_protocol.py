import unittest

from medical_sft.evaluation.medmcqa_evaluation import FINAL_SFT_INSTRUCTION, final_sft_user_message
from medical_sft.data.medmcqa_training_contract import INSTRUCTION as TRAINING_INSTRUCTION
from medical_sft.evaluation.qwen25_instruct_baseline_matched_20261110 import DO_SAMPLE, MAX_NEW_TOKENS, USE_CACHE


class MatchedQwen25InstructBaselineProtocolTests(unittest.TestCase):
    def test_prompt_replaces_answer_only_suffix_with_final_answer_explanation_instruction(self):
        prompt = (
            "Which option is correct?\n"
            "A. One\nB. Two\nC. Three\nD. Four\n"
            "Return exactly one line in the format: Answer: X"
        )
        message = final_sft_user_message({"prompt": prompt})
        expected = (
            "Which option is correct?\n"
            "A. One\nB. Two\nC. Three\nD. Four\n"
            + FINAL_SFT_INSTRUCTION
        )
        self.assertEqual(message, {"role": "user", "content": expected})
        self.assertIn("Answer: X\nExplanation: <brief explanation>", message["content"])
        self.assertNotIn("Return exactly one line", message["content"])
        self.assertNotIn("X must be exactly one letter", message["content"])
        self.assertNotIn("Output no other text", message["content"])

    def test_missing_or_drifted_prompt_is_rejected(self):
        for row in ({}, {"prompt": "Question without the locked format"}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                final_sft_user_message(row)

    def test_evaluation_instruction_is_identical_to_training_instruction(self):
        self.assertEqual(FINAL_SFT_INSTRUCTION, TRAINING_INSTRUCTION)

    def test_generation_settings_are_fixed_for_base_sft_parity(self):
        self.assertFalse(DO_SAMPLE)
        self.assertEqual(MAX_NEW_TOKENS, 32)
        self.assertTrue(USE_CACHE)


if __name__ == "__main__":
    unittest.main()
