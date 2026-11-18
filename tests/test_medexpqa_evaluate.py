import unittest
from medical_sft.evaluation import medexpqa_evaluation as e

class EvaluationTests(unittest.TestCase):
    def test_parsing_and_option_boundaries(self):
        for raw, allowed, expected in [
            ('Answer: A','ABCD',('A','A')), ('Answer: E','ABCD',(None,None)),
            ('Answer: E','ABCDE',('E','E')), (' e. ','ABCDE',(None,'E')),
            ('answer: b','ABCD',(None,'B')), ('Answer: B\nAnswer: C','ABCD',(None,None)),
            ('The answer is B','ABCD',(None,None)), ('','ABCD',(None,None))]:
            self.assertEqual(e.parse(raw, allowed), expected, raw)

    def test_metrics_and_matrices(self):
        rows=[{'gold':'A','strict_prediction':'A','normalized_prediction':'A','option_count':4},
              {'gold':'B','strict_prediction':None,'normalized_prediction':'B','option_count':4},
              {'gold':'E','strict_prediction':None,'normalized_prediction':None,'option_count':5}]
        m,c=e.score(rows)
        self.assertEqual(m['overall']['strict_correct'],1)
        self.assertEqual(m['overall']['normalized_correct'],2)
        self.assertEqual(m['overall']['invalid_count'],1)
        self.assertEqual(m['four_options']['normalized_accuracy'],1)
        self.assertEqual(m['five_options']['normalized_accuracy'],0)
        self.assertEqual(c['normalized']['E']['INVALID'],1)
        self.assertEqual(sum(sum(x.values()) for x in c['strict'].values()),3)

    def test_real_data_and_original_order(self):
        for split,count in [('validation',63),('test',125)]:
            rows=e.load(split)
            self.assertEqual(len(rows),count)
            for r in rows:
                self.assertEqual(list(r['options']), list(e.LABELS[:len(r['options'])]))
                self.assertEqual(list(r['options'].values()), [r['raw']['options'][str(i)] for i in range(1,len(r['options'])+1)])

if __name__=='__main__': unittest.main()
