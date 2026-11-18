"""Evaluation-source isolation, archival safety and synthetic blind CSV tests."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from medical_sft.evaluation import medmcqa_evaluation as ev


class EvaluationFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='medmcqa-evaluation-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = dict(artifacts={})
        self.put('dev_eval', ev.INTERNAL_PATH, [dict(id='dev1', prompt='Q1\nA. One\nB. Two', gold='A'),
                                                dict(id='dev2', prompt='Q2\nA. One\nB. Two', gold='B')])
        self.put('reserved', 'data/evaluation/medmcqa_official_validation/validation.jsonl', [dict(id='val1', prompt='Validation Q', gold=0)])
        self.put('raw.validation', 'data/raw/medmcqa/validation.parquet', [])
        self.alias = self.root / ev.RETIRED_PATH
        dev, _ = ev.load_evaluation('dev', root=self.root)
        self.alias.write_text(''.join(json.dumps(dict(id=r['id'], messages=ev.evaluation_messages(r)))+'\n' for r in dev), encoding='utf-8')
        self.medexpqa_ids = []
        for index, relative in enumerate(ev.MEDEXPQA_EVALUATION_PATHS):
            sample_id = f'expqa-fixture-{index}'
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({'id': sample_id}) + '\n', encoding='utf-8')
            self.medexpqa_ids.append(sample_id)

    def put(self, role, name, rows):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        self.manifest['artifacts'][role] = dict(path=name, rows=len(rows))
        (self.root/'data_release_manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_dev_loader_needs_no_alias_file(self):
        self.alias.unlink()
        rows, info = ev.load_evaluation(root=self.root)
        self.assertEqual([r['id'] for r in rows], ['dev1', 'dev2'])
        self.assertEqual(info['metric_name'], 'MedMCQA Internal Validation Accuracy')

    def test_official_validation_is_allowed_and_zero_based_gold_maps_to_A(self):
        rows, info = ev.load_evaluation('validation', root=self.root)
        self.assertEqual(rows[0]['gold'], 'A')
        self.assertTrue(info['official_validation_used'])
        self.assertEqual(info['metric_name'], 'MedMCQA Official Validation Test Accuracy')

    def test_canonical_validation_input_is_inferred(self):
        _, info = ev.load_evaluation(input_path='data/evaluation/medmcqa_official_validation/validation.jsonl', root=self.root)
        self.assertEqual(info['split'], 'validation')

    def test_mixed_split_and_input_are_rejected(self):
        with self.assertRaises(ValueError):
            ev.load_evaluation('dev', 'data/evaluation/medmcqa_official_validation/validation.jsonl', root=self.root)

    def test_test_or_unknown_input_cannot_enter_local_accuracy(self):
        with self.assertRaises(ValueError):
            ev.load_evaluation(input_path='official_test.parquet', root=self.root)
        with self.assertRaises(ValueError):
            ev.load_evaluation('test', root=self.root)

    def test_unlabeled_gold_in_a_canonical_file_is_rejected(self):
        self.put('reserved', 'data/evaluation/medmcqa_official_validation/validation.jsonl', [dict(id='test', prompt='Q', gold=-1)])
        with self.assertRaisesRegex(ValueError, 'official Test'):
            ev.load_evaluation('validation', root=self.root)

    def test_duplicate_dev_ids_are_rejected(self):
        self.put('dev_eval', ev.INTERNAL_PATH, [dict(id='same', prompt='Q', gold='A')]*2)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            ev.load_evaluation('dev', root=self.root)

    def test_messages_adapter_is_lossless_and_does_not_modify_row(self):
        row = dict(id='x', prompt='Question and all options', gold='C')
        before = copy.deepcopy(row)
        self.assertEqual(ev.evaluation_messages(row), [dict(role='user', content=row['prompt']), dict(role='assistant', content='Answer: C')])
        self.assertEqual(row, before)

    def test_raw_gold_mapping_rejects_minus_one_bool_and_numeric_strings(self):
        self.assertEqual([ev.gold_letter(i) for i in range(4)], list('ABCD'))
        for x in [-1, 4, True, 0.0, '0', None]:
            with self.subTest(x=x), self.assertRaises(ValueError): ev.gold_letter(x)

    def test_answer_explanation_protocol_and_ambiguity(self):
        self.assertEqual(ev.parse_model_answer('Answer: c\nExplanation: The reason.'), 'C')
        self.assertEqual(ev.parse_model_answer('Answer: A'), 'A')
        for x in ['Answer: A\nExplanation: Reason.\nAnswer: B', 'Answer: 1', 'I guess A', None]:
            self.assertIsNone(ev.parse_model_answer(x))

    def test_accuracy_denominator_contains_only_valid_gold(self):
        result = ev.score_predictions([0, -1, 'B'], ['Answer: A', 'Answer: C', 'invalid'])
        self.assertEqual(result['valid_gold_records'], 2)
        self.assertEqual(result['invalid_gold_records'], 1)
        self.assertEqual(result['accuracy'], 0.5)
        self.assertEqual(result['invalid_predictions'], 1)

    def test_no_accuracy_can_be_reported_for_blind_test(self):
        with self.assertRaisesRegex(ValueError, 'Official Test'):
            ev.score_predictions([-1, None], ['Answer: A', 'Answer: B'])

    def test_evaluation_paths_are_forbidden_as_training_inputs(self):
        for name in [
            ev.INTERNAL_PATH,
            ev.RETIRED_PATH,
            'data/evaluation/medmcqa_official_validation/validation.jsonl',
            'data/evaluation/medexpqa_full/custom_training.jsonl',
            *ev.MEDEXPQA_EVALUATION_PATHS,
        ]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                ev.assert_training_safe(self.root/name, root=self.root)

    def test_renamed_internal_eval_ids_are_still_blocked_from_training(self):
        with self.assertRaisesRegex(ValueError, 'dev1'):
            ev.assert_training_safe(self.root/'renamed_training.jsonl', ['dev1'], root=self.root)

    def test_validation_ids_are_also_blocked_from_training(self):
        with self.assertRaisesRegex(ValueError, 'val1'):
            ev.assert_training_safe(self.root/'training.jsonl', ['val1'], root=self.root)

    def test_disjoint_training_ids_pass(self):
        ev.assert_training_safe(self.root/'training.jsonl', ['train1'], root=self.root)

    def test_medexpqa_full_evaluation_ids_are_blocked_from_training(self):
        with self.assertRaisesRegex(ValueError, 'Training/evaluation ID overlap'):
            ev.assert_training_safe(
                self.root/'training.jsonl',
                [self.medexpqa_ids[0]],
                root=self.root,
            )

    def test_missing_medexpqa_evaluation_file_blocks_training(self):
        (self.root / ev.MEDEXPQA_EVALUATION_PATHS[0]).unlink()
        with self.assertRaisesRegex(ValueError, 'MedExpQA evaluation source is missing'):
            ev.assert_training_safe(self.root/'training.jsonl', ['train1'], root=self.root)

    def test_only_the_two_final_sft_model_revisions_are_registered(self):
        self.assertEqual(set(ev.MODEL_REVISIONS), {
            'Qwen/Qwen2.5-3B-Instruct',
            'Qwen/Qwen3.5-4B',
        })


if __name__ == '__main__':
    unittest.main()
