"""Exercise editable imports and module CLIs without loading model weights."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from medical_sft.paths import ROOT
from medical_sft.data import build_model_specific_medmcqa
from medical_sft.evaluation import medmcqa_evaluation, medexpqa_evaluation


class PackageLayoutTests(unittest.TestCase):
    def test_project_paths_resolve_from_package(self):
        self.assertTrue((ROOT / 'pyproject.toml').is_file())
        self.assertEqual(build_model_specific_medmcqa.SOURCE.parent.parent.parent.parent.parent, ROOT)
        self.assertEqual(medexpqa_evaluation.ROOT, ROOT)

    def test_existing_training_data_is_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            build_model_specific_medmcqa.main()

    def test_two_medmcqa_splits_and_medexpqa_can_be_loaded(self):
        self.assertEqual(len(medmcqa_evaluation.load_evaluation('dev')[0]), 9139)
        self.assertEqual(len(medmcqa_evaluation.load_evaluation('validation')[0]), 4183)
        self.assertEqual(len(medexpqa_evaluation.load_full_evaluation()[0]), 618)

    def test_module_help_from_outside_project(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        modules = ('medical_sft.data.build_model_specific_medmcqa',
                   'medical_sft.training.train_qwen25_3b_medmcqa',
                   'medical_sft.training.train_qwen35_4b_medmcqa')
        with tempfile.TemporaryDirectory() as directory:
            for module in modules:
                with self.subTest(module=module):
                    result = subprocess.run([sys.executable, '-B', '-m', module, '--help'],
                                            cwd=directory, env=env, text=True,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('usage:', result.stdout)

if __name__ == '__main__':
    unittest.main()
