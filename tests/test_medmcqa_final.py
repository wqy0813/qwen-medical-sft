"""Frozen final-rule positive/negative cases. No network, weights or training."""
from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from medical_sft.data import build_medmcqa_final as runner
from medical_sft.data import medmcqa_final_rules as rules
from medical_sft.data.medmcqa_answers import parse_answer


def raw(sid="sample", exp="Vitamin C deficiency impairs collagen synthesis.", **updates):
    r = dict(id=sid, question="Which structure develops from the embryonic region?",
             opa="Vitamin A", opb="Vitamin B12", opc="Urogenital sinus", opd="Vitamin D",
             cop=2, exp=exp, choice_type="single", subject_name="Anatomy", topic_name=None)
    r.update(updates)
    return r


def checked(r, index=1, length="within_limit"):
    result = rules.classify_record(r, index)
    result.update(length_status=length, full_token_length=513 if length == "over_limit" else 100)
    return result


def fixture_sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AnswerTests(unittest.TestCase):
    def test_valid_integer_and_letter_formats(self):
        for value, expected in [(0,"A"),(1,"B"),(2,"C"),(3,"D"),("A","A"),("a","A"),("b","B"),(" c ","C"),("d","D")]:
            with self.subTest(value=value):
                self.assertEqual(parse_answer(value), ("valid",expected))

    def test_missing_and_invalid_are_distinct(self):
        for v in [None,""," \t\r\n"]:
            self.assertEqual(parse_answer(v), ("missing",None))
        self.assertEqual(parse_answer(), ("missing",None))
        for v in ["AB","C123",4,-1,True,False,"0","1",0.0,[],{},"Answer: C"]:
            with self.subTest(value=v):
                self.assertEqual(parse_answer(v), ("invalid",None))

    def test_missing_class_D_invalid_class_C(self):
        for v in [None,""," "]:
            self.assertEqual(rules.classify_record(raw(cop=v))["quality_class"],"D")
        for v in ["AB","C123",4,True,"0"]:
            self.assertEqual(rules.classify_record(raw(cop=v))["quality_class"],"C")
        r=raw();del r['cop']
        self.assertEqual(rules.classify_record(r)['answer_parse_status'],'missing')
        self.assertEqual(rules.classify_record(r)['quality_class'],'D')

    def test_answer_index_is_replaced_by_the_shared_strict_answer_parser(self):
        for value in ["", "AB", "C123", "0", True, 4]:
            self.assertEqual(parse_answer(value), ("missing", None) if value == "" else ("invalid", None))
        self.assertEqual(parse_answer("c"), ("valid", "C"))
        self.assertEqual(parse_answer(2), ("valid", "C"))


class ContentTests(unittest.TestCase):
    def classify(self, exp="Vitamin C deficiency impairs collagen synthesis.", **updates):
        return rules.classify_record(raw(exp=exp, **updates),1)

    def test_basic_A_preserves_entire_original(self):
        source=raw(); saved=copy.deepcopy(source);r=rules.classify_record(source)
        self.assertEqual(r['quality_class'],'A')
        self.assertEqual(r['original'],saved)
        self.assertEqual(source,saved)
        self.assertEqual(r['reason_codes'],['no_explicit_issue_found'])

    def test_missing_question_and_explanation_are_D(self):
        for value in [None,"","  "]:
            self.assertEqual(self.classify(question=value)['quality_class'],'D')
            self.assertEqual(self.classify(exp=value)['quality_class'],'D')

    def test_exact_explanation_placeholders(self):
        for value in rules.PLACEHOLDERS:
            r=self.classify(exp=' '+value.upper()+' ')
            self.assertEqual(r['quality_class'],'D',value)
            self.assertIn('placeholder_explanation',r['reason_codes'])
        self.assertEqual(self.classify(exp='This explanation is not available in the source.')['quality_class'],'A')

    def test_all_missing_or_label_options_D(self):
        for values in [dict(opa='',opb=None,opc=' ',opd=''),dict(zip(['opa','opb','opc','opd'],'ABCD'))]:
            self.assertEqual(self.classify(**values)['quality_class'],'D')

    def test_partial_missing_C(self):
        self.assertEqual(self.classify(opb=' ')['quality_class'],'C')

    def test_duplicate_options_use_conservative_normalization(self):
        self.assertEqual(self.classify(opa='Vitamin A',opb=' Ｖｉｔａｍｉｎ   a ')['quality_class'],'C')
        self.assertEqual(self.classify(opa='1-2',opb='12')['quality_class'],'A')

    def test_number_single_word_and_combination_options_are_valid(self):
        for updates in [dict(opa='0',opb='1',opc='2',opd='3'),
                        dict(opa='All of the above',opb='None of the above',opc='Both A and B',opd='Neither A nor B'),
                        dict(opa='none',opb='null',opc='na',opd='-')]:
            self.assertEqual(self.classify(**updates)['quality_class'],'A')

    def test_image_in_question_each_option_and_explanation_D(self):
        for field in ['question','opa','opb','opc','opd','exp']:
            for resource in ['<img src="asset">','![scan](scan.png)','[image missing]',
                             'https://example.test/scan.jpg?x=1','images/scan.png','data:image/png;base64,AAA']:
                with self.subTest(field=field,resource=resource):
                    r=self.classify(**{field:resource})
                    self.assertEqual(r['quality_class'],'D')
                    self.assertIn('contains_image_or_image_dependency',r['reason_codes'])

    def test_structured_image_resources_and_attachments_D(self):
        for updates in [dict(image_id='123'),dict(image_payload={'bytes':'AAAA'}),
                        dict(attachments=[{'type':'image','resource':'opaque-id'}])]:
            self.assertEqual(self.classify(**updates)['quality_class'],'D')

    def test_missing_image_explicit_dependency_D(self):
        for question in ['Identify the structure shown in the image.',
                         'Which structure is indicated on the diagram?',
                         'Identify the structure shown below.', '图中所示结构是什么？']:
            self.assertEqual(self.classify(question=question)['quality_class'],'D')

    def test_image_words_alone_are_weak_and_never_D(self):
        r=self.classify(question='Which algorithm is used in image processing?')
        self.assertEqual(r['quality_class'],'A')
        self.assertTrue(r['weak_hints'])

    def test_answer_only_forms_D(self):
        for exp in ['Ans. C','Ans. C) Urogenital sinus','Answer: C','C is correct',
                    'C','Urogenital sinus','The correct answer is C.']:
            with self.subTest(exp=exp):
                self.assertEqual(self.classify(exp=exp)['quality_class'],'D')

    def test_fixable_letter_without_body_is_D_not_B(self):
        r=self.classify(exp='Ans. A) Urogenital sinus')
        self.assertEqual(r['quality_class'],'D')
        self.assertEqual(r['content']['explanation'],'Ans. A) Urogenital sinus')
        self.assertEqual(r['repairs'],[])
        self.assertEqual(r['original']['cop'],2)

    def test_fixable_letter_with_body_is_B_and_only_letter_changes(self):
        exp='Ans. A) Urogenital sinus\n\nThe urogenital sinus develops into several structures\nof the urinary system.'
        r=self.classify(exp=exp)
        self.assertEqual(r['quality_class'],'B')
        self.assertEqual(r['content']['explanation'],exp.replace('Ans. A','Ans. C'))
        self.assertEqual(r['original']['cop'],2)
        event=r['repairs'][0]
        for key in ['before','after','span','old_letter','new_letter','matched_option','id','input_index','rule']:
            self.assertIn(key,event)

    def test_correct_letter_without_repeated_option_does_not_degrade(self):
        for prefix in ['Ans. C','Answer: C','Correct answer: C','The correct answer is C']:
            self.assertEqual(self.classify(exp=prefix+'\n\nActual explanatory content remains present.')['quality_class'],'A')

    def test_mismatched_letter_other_option_C(self):
        r=self.classify(exp='Ans. A) Vitamin B12\n\nActual explanatory content.')
        self.assertEqual(r['quality_class'],'C')
        self.assertIn('explicit_answer_content_conflict',r['reason_codes'])

    def test_mismatched_letter_missing_or_uncertain_text_C(self):
        for claim in ['Ans. A','Ans. A) Unmatched option','Ans. A) Urogenital sinus plus further prose']:
            self.assertEqual(self.classify(exp=claim+'\n\nActual explanatory content.')['quality_class'],'C')

    def test_multiple_contradictory_declarations_C(self):
        r=self.classify(exp='Ans. A) Urogenital sinus\n\nExplanation body.\nAnswer: C')
        self.assertEqual(r['quality_class'],'C')
        self.assertIn('contradictory_answer_declarations',r['reason_codes'])
        self.assertFalse(r['repairs'])

    def test_later_independent_wrong_declaration_is_not_auto_repaired(self):
        r=self.classify(exp='Actual explanatory content.\n\nAnswer: A) Urogenital sinus')
        self.assertEqual(r['quality_class'],'C')
        self.assertFalse(r['repairs'])

    def test_normal_option_lists_and_body_letters_not_declarations(self):
        for exp in ['A. Vitamin A deficiency\nB. Vitamin B12 deficiency\nC. Vitamin C deficiency\nD. Vitamin D deficiency',
                    'Vitamins A, B, C and D have distinct functions.']:
            self.assertEqual(self.classify(exp=exp)['quality_class'],'A')

    def test_reference_only_D(self):
        for ref in ["Ref: Harrison's Principles of Internal Medicine,\n19th Edition.",
                    'References:\n[1] A reference title.', 'Bibliography:\nISBN: 123456']:
            self.assertEqual(self.classify(exp='Ans. C\n\n'+ref)['quality_class'],'D')

    def test_reference_followed_by_body_kept(self):
        exp="Ans. C\n\nRef: Harrison's Principles of Internal Medicine,\n19th Edition.\n\nVitamin C deficiency impairs collagen synthesis."
        r=self.classify(exp=exp)
        self.assertEqual(r['quality_class'],'A')
        self.assertEqual(r['content']['explanation'],exp)

    def test_body_before_reference_kept(self):
        exp='Ans. C\n\nActual explanatory content.\n\nReferences:\n[1] A reference title.'
        self.assertEqual(self.classify(exp=exp)['quality_class'],'A')

    def test_ambiguous_reference_C_and_never_deleted(self):
        exp='Ans. C\n\nRef: Unspecified book or explanatory material'
        r=self.classify(exp=exp)
        self.assertEqual(r['quality_class'],'C')
        self.assertEqual(r['content']['explanation'],exp)

    def test_mixed_unseparated_reference_body_never_answer_only_D(self):
        exp="Ans. C\nRef: Harrison's Principles, 19th Edition.\nVitamin C deficiency impairs collagen synthesis."
        r=self.classify(exp=exp)
        self.assertEqual(r['quality_class'],'C')
        self.assertEqual(r['content']['explanation'],exp)

    def test_safe_format_html_and_entities_B(self):
        for exp, expected in [('<b>Vitamin C</b> supports collagen synthesis.','Vitamin C supports collagen synthesis.'),
                              ('Vitamin A &amp; vitamin C have roles.','Vitamin A & vitamin C have roles.')]:
            r=self.classify(exp=exp)
            self.assertEqual(r['quality_class'],'B')
            self.assertEqual(r['content']['explanation'],expected)

    def test_html_preserves_paragraph_boundaries(self):
        r=self.classify(exp='<p>First complete sentence.</p><p>Second complete sentence.</p>')
        self.assertEqual(r['quality_class'],'B')
        self.assertIn('\n\n',r['content']['explanation'])
        self.assertIn('First complete sentence.',r['content']['explanation'])
        self.assertIn('Second complete sentence.',r['content']['explanation'])

    def test_complex_html_is_C_not_tag_stripped(self):
        for exp in ['T<sub>4</sub> concentration is measured.','x<sup>2</sup> is a square.',
                    '<math><mi>x</mi></math>','<table><tr><td>value</td></tr></table>',
                    '<b>Unclosed formatting text.','<span title="Extra information">Visible content.</span>',
                    'Text &amp;lt; uncertain entity boundary.']:
            r=self.classify(exp=exp)
            self.assertEqual(r['quality_class'],'C')
            self.assertEqual(r['content']['explanation'],exp)

    def test_false_image_field_is_not_an_actual_resource(self):
        self.assertEqual(self.classify(image=False)['quality_class'],'A')

    def test_comparison_symbols_are_preserved(self):
        exp='A value < 5 and a value > 3 can coexist.'
        self.assertEqual(self.classify(exp=exp)['content']['explanation'],exp)

    def test_consecutive_exact_sentences_B(self):
        r=self.classify(exp='Vitamin C is necessary. Vitamin C is necessary.')
        self.assertEqual(r['quality_class'],'B')
        self.assertEqual(r['content']['explanation'],'Vitamin C is necessary.')

    def test_consecutive_exact_paragraphs_B(self):
        exp='Vitamin C is necessary.\n\nVitamin C is necessary.'
        r=self.classify(exp=exp)
        self.assertEqual(r['quality_class'],'B')
        self.assertEqual(r['content']['explanation'],'Vitamin C is necessary.')

    def test_nonconsecutive_similar_words_phrases_unchanged(self):
        for exp in ['First complete sentence. Another complete sentence. First complete sentence.',
                    'Vitamin C is necessary. Vitamin C is useful.',
                    'The patient had had a previous infection.', 'This is very very important.',
                    'The patient was was admitted.', 'glaucomaAnterior text remains.',
                    'Vitamin C Vitamin C supports normal synthesis.']:
            r=self.classify(exp=exp)
            self.assertEqual(r['quality_class'],'A',exp)
            self.assertEqual(r['content']['explanation'],exp)

    def test_abbreviations_decimals_headings_lists_quotes_code_not_deleted(self):
        for exp in ['Dr. Smith is present. Dr. Smith is present.',
                    'The value is 1.5 units. The value is 1.5 units.',
                    'INTRODUCTION\n\nINTRODUCTION','A. First option.\n\nA. First option.',
                    '"A complete quote."\n\n"A complete quote."',
                    '    Complete code sentence.\n\n    Complete code sentence.',
                    '```\nSome code example.\nSome code example.\n```']:
            r=self.classify(exp=exp)
            self.assertEqual(r['content']['explanation'],exp)

    def test_repairs_are_idempotent(self):
        for exp in ['  <b>Actual content remains.</b>  ',
                    'First complete sentence. First complete sentence.',
                    'First complete sentence.\n\nFirst complete sentence.']:
            first=self.classify(exp=exp)
            again=self.classify(exp=first['content']['explanation'])
            self.assertEqual(again['quality_class'],'A')
            self.assertEqual(again['repairs'],[])

    def test_whitespace_cleanup_preserves_multiline_indentation(self):
        exp='    Aligned content column.\n    Another aligned column.'
        r=self.classify(exp=exp)
        self.assertEqual(r['content']['explanation'],exp)

    def test_definite_damage_C_low_confidence_weak_only(self):
        self.assertEqual(self.classify(exp='Important \ufffd content is damaged.')['quality_class'],'C')
        self.assertEqual(self.classify(exp='A possibly mojibake Ã© token appears.')['quality_class'],'A')

    def test_D_overrides_C_and_B(self):
        r=self.classify(exp='<b>Actual content remains.</b>', question='', cop='AB')
        self.assertEqual(r['quality_class'],'D')
        self.assertEqual(r['repairs'],[])

    def test_C_overrides_B_and_preserves_original(self):
        exp='<b>Actual content remains.</b>'
        r=self.classify(exp=exp,cop='AB')
        self.assertEqual(r['quality_class'],'C')
        self.assertEqual(r['content']['explanation'],exp)
        self.assertFalse(r['repairs'])

    def test_disabled_rules_never_degrade_quality(self):
        for exp in ['treatment therapy dose pregnancy mortality biosynthesis formed from.',
                    'The dose is 10 mg/kg per day.', 'This condition is described in the paragraph.',
                    'Short.', 'Unrelated ordinary explanation without overlap.', 'x'*1200]:
            self.assertEqual(self.classify(exp=exp,choice_type='multi')['quality_class'],'A',exp[:50])


class DatasetTests(unittest.TestCase):
    def test_fingerprints_preserve_punctuation_order_and_normalize_case(self):
        a=checked(raw())
        b=checked(raw(question='  WHICH structure develops from the embryonic region?  '))
        self.assertEqual(a['fingerprints_repaired']['prompt'],b['fingerprints_repaired']['prompt'])
        c=checked(raw(opa='Vitamin B12',opb='Vitamin A'))
        self.assertNotEqual(a['fingerprints_repaired']['prompt'],c['fingerprints_repaired']['prompt'])
        d=checked(raw(question=raw()['question'].replace('?','')))
        self.assertNotEqual(a['fingerprints_repaired']['prompt'],d['fingerprints_repaired']['prompt'])

    def test_original_and_repaired_fingerprints_both_saved(self):
        r=checked(raw(exp='<b>Actual explanatory content.</b>'))
        self.assertNotEqual(r['fingerprints_original']['full_record'],r['fingerprints_repaired']['full_record'])

    def test_full_duplicate_keeps_first_input_not_lexical_id(self):
        rows=[checked(raw('z'),1),checked(raw('a'),2)]
        rules.apply_dataset_checks(rows,[])
        self.assertEqual(rows[0]['training_status'],'eligible_candidate')
        self.assertEqual(rows[1]['training_status'],'isolated_duplicate')
        self.assertEqual(rows[1]['duplicate_representative_id'],'z')
        self.assertEqual(rows[1]['quality_class'],'A')

    def test_complete_duplicate_after_repair(self):
        rows=[checked(raw('a',exp='<b>Actual explanatory content.</b>'),1),
              checked(raw('b',exp='actual explanatory content.'),2)]
        rules.apply_dataset_checks(rows,[])
        self.assertEqual(rows[1]['training_status'],'isolated_duplicate')

    def test_same_gold_different_explanations_both_kept(self):
        rows=[checked(raw('a'),1),checked(raw('b',exp='Different actual explanatory content.'),2)]
        groups=rules.apply_dataset_checks(rows,[])
        self.assertTrue(all(r['training_status']=='eligible_candidate' for r in rows))
        self.assertTrue(all('same_prompt_multiple_explanations' in r['duplicate_flags'] for r in rows))
        self.assertEqual(sum(g.get('relation')=='same_prompt_multiple_explanations' for g in groups),1)

    def test_duplicate_gold_conflict_all_C_but_D_precedence(self):
        rows=[checked(raw('a'),1),checked(raw('b',cop=0),2),checked(raw('d',exp=None,cop=1),3)]
        rules.apply_dataset_checks(rows,[])
        self.assertEqual([r['quality_class'] for r in rows],['C','C','D'])

    def test_question_only_and_option_order_changes_do_not_isolate(self):
        rows=[checked(raw('a'),1),checked(raw('b',opa='Vitamin B12',opb='Vitamin A'),2)]
        rules.apply_dataset_checks(rows,[])
        self.assertTrue(all(r['training_status']=='eligible_candidate' for r in rows))

    def test_dev_same_gold_isolates_training_only(self):
        rows=[checked(raw())];dev=[raw('dev')];saved=copy.deepcopy(dev)
        rules.apply_dataset_checks(rows,dev)
        self.assertEqual(rows[0]['quality_class'],'A')
        self.assertEqual(rows[0]['training_status'],'isolated_dev_overlap')
        self.assertEqual(dev,saved)

    def test_dev_different_gold_is_C_and_preserves_gold(self):
        rows=[checked(raw())]
        rules.apply_dataset_checks(rows,[raw('dev',cop=0)])
        self.assertEqual(rows[0]['quality_class'],'C')
        self.assertEqual(rows[0]['training_status'],'isolated_quality_C')
        self.assertEqual(rows[0]['original']['cop'],2)
        self.assertIn('dev_answer_conflict',rows[0]['training_block_reasons'])

    def test_all_block_reasons_and_priority(self):
        rows=[checked(raw('a'),1,'over_limit'),checked(raw('b'),2,'over_limit')]
        groups=rules.apply_dataset_checks(rows,[raw('dev')])
        self.assertEqual(rows[1]['training_status'],'isolated_dev_overlap')
        self.assertEqual(set(rows[1]['training_block_reasons']),{'over_512','dev_eval_overlap','exact_duplicate_copy'})
        summary=rules.summarize(rows,groups)
        self.assertEqual(summary['blocked_records_union'],2)
        self.assertGreater(sum(summary['training_block_reason_counts'].values()),2)

    def test_overlength_does_not_change_quality(self):
        r=checked(raw(),length='over_limit')
        rules.apply_dataset_checks([r],[])
        self.assertEqual(r['quality_class'],'A')
        self.assertEqual(r['training_status'],'isolated_over_512')

    def test_missing_length_cannot_become_eligible(self):
        with self.assertRaises(ValueError):rules.apply_dataset_checks([rules.classify_record(raw())],[])

    def test_duplicate_ids_fail_without_losing_records(self):
        rows=[checked(raw()),checked(raw())]
        with self.assertRaisesRegex(ValueError,'Duplicate input ID'):
            rules.apply_dataset_checks(rows,[])
        self.assertEqual(len(rows),2)

    def test_deterministic_dataset_checks(self):
        source=[checked(raw('a')),checked(raw('b')),checked(raw('c',exp='Different actual explanation.'))]
        one,two=copy.deepcopy(source),copy.deepcopy(source)
        g1=rules.apply_dataset_checks(one,[]);g2=rules.apply_dataset_checks(two,[])
        self.assertEqual(one,two)
        self.assertEqual(rules.summarize(one,g1),rules.summarize(two,g2))

    def test_training_status_D_and_C_override_other_reasons(self):
        for cop,expected in [(None,'excluded_quality_D'),('AB','isolated_quality_C')]:
            r=checked(raw(cop=cop),length='over_limit')
            rules.apply_dataset_checks([r],[raw('dev')])
            self.assertEqual(r['training_status'],expected)
            self.assertIn('dev_eval_overlap',r['training_block_reasons'])
            self.assertIn('over_512',r['training_block_reasons'])

    def test_all_content_quality_block_causes_survive_D_priority(self):
        r=checked(raw(question='',cop='AB'))
        rules.apply_dataset_checks([r],[])
        self.assertEqual(r['training_status'],'excluded_quality_D')
        self.assertIn('missing_question',r['training_block_reasons'])
        self.assertIn('invalid_gold_answer',r['training_block_reasons'])


class InputAndExecutionTests(unittest.TestCase):
    def test_default_preflight_and_full_requires_explicit_token(self):
        self.assertEqual(runner.parse_args([]).mode,'preflight')
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):runner.parse_args(['--mode','full','--output','unused'])
            with self.assertRaises(SystemExit):runner.parse_args(['--mode','sample','--limit','173648','--output','unused'])

    def test_output_cannot_overwrite_existing_directory(self):
        with self.assertRaisesRegex(ValueError,'new directory'):
            runner.main(['--mode','sample','--output',str(runner.ROOT)])

    def test_manifest_path_escape_rejected(self):
        with self.assertRaises(ValueError):runner.resolve_input(runner.ROOT,'../outside')

    def test_final_pipeline_imports_only_the_current_modules(self):
        expected = {
            'medmcqa_answers.py': set(),
            'medmcqa_final_rules.py': {'medical_sft.data.medmcqa_answers'},
            'medmcqa_training_contract.py': set(),
            'build_medmcqa_final.py': {'medical_sft.data.medmcqa_final_rules', 'medical_sft.data.medmcqa_training_contract'},
        }
        current_modules = {'medical_sft.data.' + name[:-3] for name in expected}
        for name, modules in expected.items():
            path = runner.ROOT / 'src/medical_sft/data' / name
            tree = ast.parse(path.read_text(encoding='utf-8-sig'))
            imports = {node.module for node in ast.walk(tree)
                       if isinstance(node, ast.ImportFrom) and node.module}
            self.assertEqual(imports & current_modules, modules)
            source = path.read_text(encoding='utf-8-sig')
            for forbidden in ['HIGH_RISK_MEDICAL_CLAIM', 'MEDICATION_DOSE_UNIT', 'RATIONALE_CUE',
                              'content_tokens', 'low_question_explanation_lexical_overlap',
                              'known_unresolved_ids']:
                self.assertNotIn(forbidden, source)

    def test_output_partition_repairs_and_candidate_schema(self):
        records=[checked(raw('a')),checked(raw('b',exp='<b>Actual explanatory content.</b>')),
                 checked(raw('c',cop='AB')),checked(raw('d',exp=None))]
        groups=rules.apply_dataset_checks(records,[])
        report=rules.summarize(records,groups);report['scope']='fixture'
        with tempfile.TemporaryDirectory(prefix='medmcqa-output-') as tmp:
            out=Path(tmp)/'new'
            runner.write_outputs(out,records,groups,report)
            classes=[]
            for n in ['medmcqa_A.jsonl','medmcqa_B_cleaned.jsonl','medmcqa_C_review.jsonl','medmcqa_D_excluded.jsonl']:
                classes.extend(list(runner.read_jsonl(out/n)))
            self.assertEqual(len(classes),4)
            self.assertEqual({r['id'] for r in classes},{'a','b','c','d'})
            candidates=list(runner.read_jsonl(out/'medmcqa_train_candidates_512.jsonl'))
            for r in candidates:
                self.assertIn('\nExplanation: ',r['messages'][1]['content'])
                self.assertEqual(r['training_status'],'eligible_candidate')
            self.assertFalse((out/'RUN_INCOMPLETE').exists())
            from medical_sft.data.verify_medmcqa_final import verify_output
            verified=verify_output(out,verify_sources=False)
            self.assertTrue(verified['passed'])
            with (out/'medmcqa_A.jsonl').open('a',encoding='utf-8') as f:f.write('{}\n')
            with self.assertRaises((ValueError, KeyError, AssertionError)):verify_output(out,verify_sources=False)
            with self.assertRaises(FileExistsError):runner.write_outputs(out,records,groups,report)


class InputFixtureTests(unittest.TestCase):
    def setUp(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.temp = tempfile.TemporaryDirectory(prefix='medmcqa-input-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.expected={'train_scope':3,'raw.train':40,'dev_eval':2}
        mock=patch.object(runner,'EXPECTED',self.expected);mock.start();self.addCleanup(mock.stop)
        self.raws=[raw(f'id{i}',question=f'Question {i}?') for i in range(40)]
        self.raw_path=self.root/'raw.parquet'
        pq.write_table(pa.Table.from_pylist(self.raws),self.raw_path)
        self.train_scope=[dict(id=r['id']) for r in self.raws[:3]]
        self.dev=[dict(id=r['id'],prompt=runner.source_eval_prompt(r),gold='C') for r in self.raws[3:5]]
        self.entries={'raw.train':dict(path='raw.parquet',rows=40,version='fixture')}
        for role,rows in [('train_scope',self.train_scope),('dev_eval',self.dev)]:
            p=self.root/f'{role}.jsonl';runner.write_jsonl(p,rows)
            self.entries[role]=dict(path=p.name,rows=len(rows),version='fixture')
        for e in self.entries.values():e['sha256']=fixture_sha(self.root/e['path'])
        self.manifest=dict(
            schema_version=1,
            dataset=dict(id='openlifescienceai/medmcqa',source_version='fixture'),
            active_evaluation='dev_eval',
            reserved_evaluation='reserved',
            artifacts=self.entries,
            documented_omissions=[dict(id=r['id'],reason='documented_fixture') for r in self.raws[5:]],
        )
        self.write_manifest()

    def write_manifest(self):
        (self.root/'data_release_manifest.json').write_text(json.dumps(self.manifest),encoding='utf-8')

    def replace_jsonl(self,role,rows):
        path=self.root/self.entries[role]['path']
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        self.entries[role]['sha256']=fixture_sha(path)
        self.write_manifest()

    def test_bounded_selection_and_all_dev_aliases_pass(self):
        selected,dev,evidence=runner.load_inputs(self.root,sample_limit=1,include_ids=['id2'])
        self.assertEqual([r['id'] for _,r,_ in selected],['id0','id2'])
        self.assertEqual(len(dev),2)
        self.assertTrue(evidence['lineage_passed'])

    def test_wrong_count_refused_even_if_manifest_updated(self):
        self.entries['train_scope']['rows']=4;self.write_manifest()
        with self.assertRaisesRegex(ValueError,'input count'):runner.load_inputs(self.root,1)

    def test_duplicate_scope_ids_refused(self):
        rows=copy.deepcopy(self.train_scope);rows[1]=copy.deepcopy(rows[0]);self.replace_jsonl('train_scope',rows)
        with self.assertRaisesRegex(ValueError,'uniqueness'):runner.load_inputs(self.root,1)

    def test_dev_gold_lineage_mismatch_refused(self):
        rows=copy.deepcopy(self.dev);rows[0]['gold']='A'
        self.replace_jsonl('dev_eval',rows)
        with self.assertRaisesRegex(ValueError,'Dev content lineage'):runner.load_inputs(self.root,1)

    def test_missing_raw_mapping_refused(self):
        rows=copy.deepcopy(self.train_scope);rows[0]['id']='not_in_raw';self.replace_jsonl('train_scope',rows)
        with self.assertRaisesRegex(ValueError,'mapping mismatch'):runner.load_inputs(self.root,1)

    def test_invalid_source_gold_reaches_classifier_not_silently_coerced(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        self.raws[0]['cop']=None
        pq.write_table(pa.Table.from_pylist(self.raws),self.raw_path)
        self.entries['raw.train']['sha256']=fixture_sha(self.raw_path);self.write_manifest()
        selected,_,_=runner.load_inputs(self.root,1)
        r=rules.classify_record(selected[0][1])
        self.assertEqual((r['answer_parse_status'],r['quality_class']),('missing','D'))

    def test_unrelated_manifest_metadata_does_not_change_the_declared_scope(self):
        self.manifest['unrelated_metadata']={'ids_not_to_use':['id0']}
        self.write_manifest()
        selected,_,_=runner.load_inputs(self.root,1)
        self.assertEqual([r['id'] for _,r,_ in selected],['id0'])

    def test_sample_id_cannot_come_from_dev_or_unknown_source(self):
        for sid in ['id3','missing']:
            with self.assertRaisesRegex(ValueError,'training scope'):runner.load_inputs(self.root,1,[sid])


if __name__ == '__main__':
    unittest.main()
