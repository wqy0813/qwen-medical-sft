"""Offline integrity audit for the frozen 618-row MedExpQA evaluation release."""
import argparse
import collections
import json
import os
from pathlib import Path
import re
import unicodedata

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from medical_sft.paths import ROOT
DATA = ROOT / 'data/evaluation/medexpqa_full'
LABELS = 'ABCDE'
TRAIN_FILES = (
    'data/processed/train/model_specific/qwen25_3b/medmcqa_train.jsonl',
    'data/processed/train/model_specific/qwen35_4b/medmcqa_train.jsonl',
)

def norm(text):
    return ' '.join(unicodedata.normalize('NFKC', text).split()).casefold()

def load(split):
    p = DATA / (split + '_preserved.jsonl')
    rows = []
    with p.open(encoding='utf-8') as f:
        for line, raw in enumerate(f, 1):
            r = json.loads(raw)
            if not isinstance(r, dict) or not isinstance(r.get('full_question'), str) or not r['full_question'].strip():
                raise ValueError(f'{p}:{line}: invalid question/object')
            o = r.get('options')
            if not isinstance(o, dict) or set(o) != set('12345'):
                raise ValueError(f'{p}:{line}: invalid option keys')
            if any(v is not None and (not isinstance(v, str) or not v.strip()) for v in o.values()):
                raise ValueError(f'{p}:{line}: invalid option value')
            keys = [k for k in '12345' if isinstance(o[k], str) and o[k].strip()]
            if keys not in [list('1234'), list('12345')] or type(r.get('correct_option')) is not int or str(r['correct_option']) not in keys:
                raise ValueError(f'{p}:{line}: invalid option pattern/gold')
            if r.get('id') is None: raise ValueError(f'{p}:{line}: missing id')
            rows.append({'id': r['id'], 'line': line, 'question': r['full_question'],
                         'options': {LABELS[int(k)-1]: o[k] for k in keys},
                         'gold': LABELS[r['correct_option']-1], 'raw': r})
    return rows

def load_full_evaluation():
    """Load all 618 retained MedExpQA records for evaluation only.

    The Train evaluation split includes all three disposition files (within-limit,
    overlength and quality-excluded); no training-admission filter is applied.
    """
    manifest_path = DATA / 'manifest.json'
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('purpose') != 'evaluation_only' or manifest.get('total_rows') != 618:
        raise ValueError('MedExpQA manifest is not the frozen 618-row evaluation-only release')

    rows = []
    split_counts = {}
    seen_ids = set()
    train_spec = manifest['splits']['train']
    train_parts = train_spec['composition']
    if sum(part['rows'] for part in train_parts) != train_spec['rows']:
        raise ValueError('MedExpQA Train composition does not sum to 430')

    for part in train_parts:
        path = DATA / part['file']
        if not path.is_file():
            raise FileNotFoundError(path)
        count = 0
        with path.open(encoding='utf-8') as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    raise ValueError(f'{part["file"]}:{line_number}: blank evaluation record')
                record = json.loads(raw)
                original = record.get('original') if isinstance(record, dict) else None
                if not isinstance(original, dict):
                    raise ValueError(f'{part["file"]}:{line_number}: missing original source record')
                sample_id = record.get('id')
                if not isinstance(sample_id, (str, int)) or not str(sample_id):
                    raise ValueError(f'{part["file"]}:{line_number}: invalid record ID')
                if str(original.get('id')) != str(sample_id):
                    raise ValueError(f'{part["file"]}:{line_number}: original/top-level ID mismatch')
                question = original.get('full_question')
                options_raw = original.get('options')
                correct = original.get('correct_option')
                if not isinstance(question, str) or not question.strip():
                    raise ValueError(f'{part["file"]}:{line_number}: missing question')
                if not isinstance(options_raw, dict) or set(options_raw) != set('12345'):
                    raise ValueError(f'{part["file"]}:{line_number}: invalid numeric option keys')
                option_keys = [
                    key for key in '12345'
                    if isinstance(options_raw[key], str) and options_raw[key].strip()
                ]
                if option_keys not in [list('1234'), list('12345')]:
                    raise ValueError(f'{part["file"]}:{line_number}: expected four or five non-empty options')
                if type(correct) is not int or str(correct) not in option_keys:
                    raise ValueError(f'{part["file"]}:{line_number}: invalid original Gold option')
                options = {LABELS[int(key) - 1]: options_raw[key] for key in option_keys}
                ident = str(sample_id)
                if ident in seen_ids:
                    raise ValueError(f'Duplicate MedExpQA evaluation ID: {ident}')
                seen_ids.add(ident)
                rows.append({
                    'id': ident,
                    'line': line_number,
                    'question': question,
                    'options': options,
                    'gold': LABELS[correct - 1],
                    'option_count': len(options),
                    'source_split': 'train',
                    'source_file': part['file'],
                    'raw': original,
                })
                count += 1
        if count != part['rows']:
            raise ValueError(f'MedExpQA Train evaluation row count mismatch: {part["file"]}')
        split_counts['train'] = split_counts.get('train', 0) + count

    for split in ('validation', 'test'):
        spec = manifest['splits'][split]
        path = DATA / spec['file']
        if not path.is_file():
            raise FileNotFoundError(path)
        split_rows = load(split)
        if len(split_rows) != spec['rows']:
            raise ValueError(f'MedExpQA {split} evaluation row count mismatch')
        for row in split_rows:
            ident = str(row['id'])
            if ident in seen_ids:
                raise ValueError(f'Duplicate MedExpQA evaluation ID: {ident}')
            seen_ids.add(ident)
            rows.append({
                'id': ident,
                'line': row['line'],
                'question': row['question'],
                'options': row['options'],
                'gold': row['gold'],
                'option_count': len(row['options']),
                'source_split': split,
                'source_file': spec['file'],
                'raw': row['raw'],
            })
        split_counts[split] = len(split_rows)

    if len(rows) != 618 or split_counts != {'train': 430, 'validation': 63, 'test': 125}:
        raise ValueError('MedExpQA full evaluation split counts do not match 430/63/125')
    return rows, {
        'purpose': 'evaluation_only',
        'total_rows': len(rows),
        'unique_ids': len(seen_ids),
        'split_rows': split_counts,
        'files': [
            {'path': part['file'], 'rows': part['rows'], 'sha256': part['sha256']}
            for part in train_parts
        ] + [
            {'path': manifest['splits'][split]['file'],
             'rows': manifest['splits'][split]['rows'],
             'sha256': manifest['splits'][split]['sha256']}
            for split in ('validation', 'test')
        ],
    }


def parse(raw, allowed):
    """Strict: exact Answer: X; normalized: exact bare X or Answer: X, case-insensitive.

    Whitespace and a final period are tolerated only by normalization. No prose,
    multiple declarations or out-of-range letters are accepted.
    """
    strict_match = re.fullmatch(r'Answer: ([A-E])', raw.strip())
    normalized_match = re.fullmatch(r'\s*(?:answer\s*:\s*)?([A-E])\s*\.?\s*', raw, re.I)
    strict = strict_match.group(1) if strict_match and strict_match.group(1) in allowed else None
    normalized = normalized_match.group(1).upper() if normalized_match else None
    return strict, normalized if normalized is not None and normalized in allowed else None

def score(predictions):
    n = len(predictions)
    def part(items):
        m = len(items)
        return {'n': m, 'strict_correct': sum(r['strict_prediction'] == r['gold'] for r in items),
                'normalized_correct': sum(r['normalized_prediction'] == r['gold'] for r in items),
                'strict_accuracy': sum(r['strict_prediction'] == r['gold'] for r in items)/m if m else None,
                'normalized_accuracy': sum(r['normalized_prediction'] == r['gold'] for r in items)/m if m else None,
                'invalid_count': sum(r['normalized_prediction'] is None for r in items),
                'invalid_rate': sum(r['normalized_prediction'] is None for r in items)/m if m else None}
    matrices = {kind: {g: {p: 0 for p in list(LABELS)+['INVALID']} for g in LABELS} for kind in ('strict', 'normalized')}
    for r in predictions:
        for kind in matrices: matrices[kind][r['gold']][r[kind+'_prediction'] or 'INVALID'] += 1
    return {'overall': part(predictions), 'four_options': part([r for r in predictions if r['option_count']==4]),
            'five_options': part([r for r in predictions if r['option_count']==5])}, matrices

def audit():
    rows, release = load_full_evaluation()
    result = {
        'purpose': 'evaluation_only',
        'total_rows': len(rows),
        'split_rows': release['split_rows'],
        'files': {},
        'duplicate_ids': [],
        'identical_records': [],
        'same_question_groups': [],
        'training_checks': [],
        'scope': 'All 618 records in the frozen evaluation release are checked against both final MedMCQA model-specific training files; this does not prove absence of semantic variants or overlap in unlisted external datasets.',
    }
    ids, records, questions = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for row in rows:
        ident = {'split': row['source_split'], 'line': row['line'], 'id': row['id']}
        ids[str(row['id'])].append(ident)
        records[json.dumps(row['raw'], sort_keys=True, ensure_ascii=False)].append(ident)
        questions[norm(row['question'])].append(ident)

    result['files'] = {item['path']: {'rows': item['rows']} for item in release['files']}

    for label, group in [('duplicate_ids', ids), ('identical_records', records), ('same_question_groups', questions)]:
        result[label] = [members for members in group.values() if len(members) > 1]

    for rel in TRAIN_FILES:
        path = ROOT / rel
        if not path.is_file():
            result['training_checks'].append({'path': rel, 'status': 'missing'})
            continue
        hits_id, hits_question, missing = [], [], 0
        source_counts = collections.Counter()
        count = 0
        with path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f'Blank final MedMCQA training row: {rel}:{line_number}')
                record = json.loads(line)
                count += 1
                source = record.get('source')
                source_counts[str(source)] += 1
                sample_id = str(record.get('source_id', record.get('id')))
                question = record.get('question')
                if sample_id in ids:
                    hits_id.append({'line': line_number, 'id': sample_id, 'matches': ids[sample_id]})
                if isinstance(question, str) and norm(question) in questions:
                    hits_question.append({'line': line_number, 'id': sample_id, 'matches': questions[norm(question)]})
                if not isinstance(question, str) or not question.strip():
                    missing += 1
        result['training_checks'].append({
            'path': rel,
            'rows': count,
            'source_counts': dict(source_counts),
            'id_hits': hits_id,
            'question_hits': hits_question,
            'missing_question': missing,
        })

    return result

def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2); f.write('\n')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode', choices=['audit'])
    ap.add_argument('--out', type=Path, required=True, help='New output directory; never overwrites')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / 'data_audit.json', audit())

if __name__=='__main__': main()
