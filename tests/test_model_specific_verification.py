import json
import tempfile
import unittest
from pathlib import Path
from medical_sft.data.verify_model_specific_medmcqa import verify

class TrainingDataVerificationTests(unittest.TestCase):
    def test_count_content_order_and_length(self):
        row = {'id':'MedMCQA:1','source':'MedMCQA','source_id':'1','question':'Q','options':{'A':'a','B':'b','C':'c','D':'d'},'gold_answer':'A','explanation':'E','messages':[{'role':'user','content':'Q'},{'role':'assistant','content':'Answer: A\nExplanation: E'}], 'full_token_length':5,'prompt_token_length':3,'target_token_length':2}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [root / name for name in ('common','first','second')]
            for path in files:
                path.write_text(json.dumps(row) + '\n', encoding='utf-8')
            self.assertEqual(verify(files[0], {'first':files[1], 'second':files[2]}, 1)['status'], 'PASS')
            broken = dict(row, full_token_length=513)
            files[1].write_text(json.dumps(broken) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                verify(files[0], {'first':files[1], 'second':files[2]}, 1)
            files[1].write_text(json.dumps(row) + '\n', encoding='utf-8')
            files[2].write_text('', encoding='utf-8')
            with self.assertRaises(ValueError):
                verify(files[0], {'first':files[1], 'second':files[2]}, 1)

if __name__ == '__main__':
    unittest.main()
