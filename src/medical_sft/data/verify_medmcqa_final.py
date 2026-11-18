"""Independently verify final-cleaner artifacts (sample or approved full output).

Checks preservation, repair replay and partition/candidate set consistency.
Does not call a model, train, or modify input/output files.
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from medical_sft.data.build_medmcqa_final import ROOT, read_jsonl, resolve_input
from medical_sft.data.medmcqa_answers import parse_answer
from medical_sft.data.medmcqa_final_rules import classify_record, content_from_raw, fingerprints, set_field, text_fields
from medical_sft.data.medmcqa_training_contract import check_length, load_local_processor, training_record


FINAL_DELIVERY_FILES = frozenset({
    'README.md', 'medmcqa_cleaning_report.json', 'medmcqa_A.jsonl',
    'medmcqa_B_cleaned.jsonl', 'medmcqa_C_review.jsonl', 'medmcqa_D_excluded.jsonl',
    'medmcqa_AB_quality_candidates.jsonl', 'medmcqa_train_candidates_512.jsonl',
    'medmcqa_over_512.jsonl', 'medmcqa_train_dev_overlap_isolated.jsonl',
    'medmcqa_duplicate_groups.jsonl', 'medmcqa_repair_log.jsonl',
    'medmcqa_training_blocked_union.jsonl',
})


def verify_source_data(root, report):
    manifest = json.loads((Path(root) / 'data_release_manifest.json').read_text(encoding='utf-8'))
    for role in ('train_scope', 'raw.train', 'dev_eval', 'reserved', 'raw.validation'):
        path = resolve_input(Path(root), manifest['artifacts'][role]['path'])
        if not path.is_file():
            raise FileNotFoundError(path)
    return manifest


def verify_output(output, processor=None, verify_sources=True):
    if not __debug__:
        raise RuntimeError('Verification must run with Python assertions enabled')
    output=Path(output)
    if (output/'RUN_INCOMPLETE').exists():
        raise ValueError('Output is incomplete')
    report=json.loads((output/'medmcqa_cleaning_report.json').read_text(encoding='utf-8'))
    names={'A':'medmcqa_A.jsonl','B':'medmcqa_B_cleaned.jsonl','C':'medmcqa_C_review.jsonl','D':'medmcqa_D_excluded.jsonl'}
    rows=[]
    counts={}
    for c,name in names.items():
        group=list(read_jsonl(output/name))
        assert all(r['quality_class']==c for r in group),name
        counts[c]=len(group)
        rows.extend(group)
    byid={r['id']:r for r in rows}
    assert len(byid)==len(rows)==report['input_records']
    assert counts==report['quality_counts']
    if 'inputs' in report:
        assert set(byid)==set(report['inputs']['selected_ids'])
    ordered=sorted(rows,key=lambda r:r['input_index'])
    def norm(value):
        return ' '.join(unicodedata.normalize('NFKC',str(value or '')).split()).casefold()
    def prompt_key(content):
        return tuple(norm(content[k]) for k in ['question'])+tuple(norm(content['options'][l]) for l in 'ABCD')
    prompt_groups=defaultdict(list)
    full_groups=defaultdict(list)
    for r in ordered:
        key=prompt_key(r['content'])
        prompt_groups[key].append(r)
        # Invalid answers are quality-blocked and still have explicit typed keys.
        answer=(r['answer_parse_status'],r['gold_answer']) if r['answer_parse_status']=='valid' else (
            r['answer_parse_status'],type(r['original'].get('cop')).__name__,json.dumps(r['original'].get('cop'),sort_keys=True))
        full_groups[(key,answer,norm(r['content']['explanation']))].append(r)
    for members in full_groups.values():
        for i,r in enumerate(members):
            assert ('exact_duplicate_copy' in r['duplicate_flags'])==(i>0)
            if i:assert r['duplicate_representative_id']==members[0]['id']
    for members in prompt_groups.values():
        golds={r['gold_answer'] for r in members if r['answer_parse_status']=='valid'}
        for r in members:
            assert ('duplicate_answer_conflict' in r['duplicate_flags'])==(len(golds)>1)
            if len(golds)>1:assert r['quality_class'] in ('C','D')
    with (output/'medmcqa_duplicate_groups.jsonl').open(encoding='utf-8') as f:
        group_reports=[json.loads(line) for line in f]
    expected_groups={}
    for dimension,keyname in [('question_only','question'),('prompt','prompt'),('full_record','full_record')]:
        grouped=defaultdict(list)
        for r in ordered:grouped[r['fingerprints_repaired'][keyname]].append(r)
        for fingerprint,members in grouped.items():
            if len(members)>1:expected_groups[(dimension,fingerprint)]=[r['id'] for r in members]
    assert {(g['kind'],g['fingerprint']):g['ids'] for g in group_reports}==expected_groups
    assert len(group_reports)==len(expected_groups)
    events=[]
    for r in rows:
        original=r['original']
        assert original['id']==r['id']
        assert parse_answer(original.get('cop'))==(r['answer_parse_status'],r['gold_answer'])
        replay=content_from_raw(original)
        assert fingerprints(replay,original.get('cop'))==r['fingerprints_original']
        for event in r['repairs']:
            assert event['id']==r['id'] and event['input_index']==r['input_index']
            assert dict(text_fields(replay))[event['field']]==event['before']
            assert 'span' in event
            set_field(replay,event['field'],event['after'])
            events.append(event)
        assert replay==r['content'],r['id']
        assert fingerprints(replay,original.get('cop'))==r['fingerprints_repaired']
        if r['quality_class']=='A':
            assert not r['repairs']
        if r['quality_class']=='B':
            assert r['repairs']
            again=dict(original,question=replay['question'],exp=replay['explanation'])
            again.update({f'op{l.lower()}':replay['options'][l] for l in 'ABCD'})
            second=classify_record(again)
            assert second['quality_class']=='A' and not second['repairs'],r['id']
        q=r['quality_class'];blocks=r['training_block_reasons'];flags=r['duplicate_flags']
        assert len(blocks)==len(set(blocks))
        if q in ('C','D'):assert f'quality_{q}' in blocks
        for reason in r['quality_block_reasons']:assert reason in blocks
        for flag in flags:
            if flag!='same_prompt_multiple_explanations':assert flag in blocks
        if r['length_status']=='over_limit':assert 'over_512' in blocks
        expected=('excluded_quality_D' if q=='D' else 'isolated_quality_C' if q=='C' else
                  'isolated_dev_overlap' if 'dev_eval_overlap' in flags else
                  'isolated_duplicate' if 'exact_duplicate_copy' in flags else
                  'isolated_over_512' if r['length_status']=='over_limit' else 'eligible_candidate')
        assert r['training_status']==expected
        if r['length_status']!='not_checked':
            assert r['length_status']==('within_limit' if r['full_token_length']<=512 else 'over_limit')
            if processor is not None:
                measured=check_length(processor,r)
                for key,value in measured.items():assert r[key]==value,(r['id'],key)
    # Compare every sidecar as a set, retaining complete records and messages.
    expected_sets={
        'medmcqa_AB_quality_candidates.jsonl':{r['id'] for r in rows if r['quality_class'] in ('A','B')},
        'medmcqa_train_candidates_512.jsonl':{r['id'] for r in rows if r['training_status']=='eligible_candidate'},
        'medmcqa_over_512.jsonl':{r['id'] for r in rows if r['length_status']=='over_limit'},
        'medmcqa_train_dev_overlap_isolated.jsonl':{r['id'] for r in rows if 'dev_eval_overlap' in r['duplicate_flags']},
    }
    for filename,expected_ids in expected_sets.items():
        view=list(read_jsonl(output/filename))
        assert len(view)==len({r['id'] for r in view})
        assert {r['id'] for r in view}==expected_ids,filename
        for r in view:
            parent=byid[r['id']]
            if filename=='medmcqa_train_candidates_512.jsonl':
                assert r['quality_class'] in ('A','B') and r['length_status']=='within_limit'
                assert not r['training_block_reasons']
                assert r['messages']==training_record(parent)['messages']
                assert {k:v for k,v in r.items() if k!='messages'}==parent
            else:assert r==parent
    events.sort(key=lambda e:(e['input_index'],e['event_index']))
    logged=list(read_jsonl(output/'medmcqa_repair_log.jsonl'))
    assert events==logged
    assert len(events)==report['repair_events']
    reasons=Counter(b for r in rows for b in r['training_block_reasons'])
    assert dict(reasons)==report['training_block_reason_counts']
    assert sum(bool(r['training_block_reasons']) for r in rows)==report['blocked_records_union']
    assert sorted(r['id'] for r in rows if r['training_block_reasons'])==report['blocked_ids_union']
    if verify_sources:
        manifest=verify_source_data(ROOT,report)
        import pyarrow.parquet as pq
        scope=list(read_jsonl(ROOT/manifest['artifacts']['train_scope']['path']))
        active_dev=list(read_jsonl(ROOT/manifest['artifacts']['dev_eval']['path']))
        assert len(scope)==manifest['artifacts']['train_scope']['rows']
        assert len(active_dev)==manifest['artifacts']['dev_eval']['rows']
        dev_ids={r['id'] for r in active_dev}
        raw_by_id={}
        raw_line_by_id={}
        line=0
        for batch in pq.ParquetFile(ROOT/manifest['artifacts']['raw.train']['path']).iter_batches(batch_size=4096):
            for original in batch.to_pylist():
                line+=1
                if original['id'] in byid or original['id'] in dev_ids:
                    raw_by_id[original['id']]=original
                    raw_line_by_id[original['id']]=line
        assert line==manifest['artifacts']['raw.train']['rows']
        dev_keys=defaultdict(list)
        for d in active_dev:
            source=raw_by_id[d['id']]
            assert parse_answer(source['cop'])==parse_answer(d['gold'])
            dev_keys[prompt_key(content_from_raw(source))].append(parse_answer(source['cop'])[1])
        for r in rows:
            assert raw_by_id[r['id']]==r['original'],('raw content mismatch',r['id'])
            assert scope[r['input_index']-1]['id']==r['id']
            assert r['source']['raw_row']==raw_line_by_id[r['id']]
            overlap=dev_keys.get(prompt_key(r['content']),[])
            assert ('dev_eval_overlap' in r['duplicate_flags'])==bool(overlap)
            conflict=bool(overlap and r['answer_parse_status']=='valid' and any(g!=r['gold_answer'] for g in overlap))
            assert ('dev_answer_conflict' in r['duplicate_flags'])==conflict
    return dict(passed=True, scope=report['scope'], records=len(rows), quality_counts=counts,
                repair_events=len(events), real_lengths_rechecked=processor is not None,
                sources_rechecked=verify_sources, training_started=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--real-tokenizer',action='store_true')
    args=parser.parse_args()
    processor=load_local_processor()[0] if args.real_tokenizer else None
    print(json.dumps(verify_output(args.output,processor),sort_keys=True))


if __name__=='__main__':main()
