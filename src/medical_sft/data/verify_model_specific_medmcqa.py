"""Check final common and model-specific training records without rebuilding history."""
import json
from itertools import zip_longest
from pathlib import Path

from medical_sft.paths import ROOT
COMMON = ROOT / 'data/processed/train/common/medmcqa_train_common_137928.jsonl'
SPECS = {
    'qwen25_3b': ROOT / 'data/processed/train/model_specific/qwen25_3b/medmcqa_train.jsonl',
    'qwen35_4b': ROOT / 'data/processed/train/model_specific/qwen35_4b/medmcqa_train.jsonl',
}

def rows(path):
    with Path(path).open(encoding='utf-8') as handle:
        for line, text in enumerate(handle, 1):
            if not text.strip():
                raise ValueError(f'{path}:{line}: blank record')
            yield json.loads(text)

def verify(common, files, expected_rows=137928):
    sentinel = object()
    seen = set()
    maximum = {name: 0 for name in files}
    fields = ('id', 'source', 'source_id', 'question', 'options', 'gold_answer', 'explanation', 'messages')
    count = 0
    for items in zip_longest(rows(common), *(rows(p) for p in files.values()), fillvalue=sentinel):
        if any(item is sentinel for item in items):
            raise ValueError('Training file lengths differ')
        source, *models = items
        count += 1
        if source['id'] in seen:
            raise ValueError(f'Duplicate ID: {source["id"]}')
        seen.add(source['id'])
        if source.get('source') != 'MedMCQA' or source.get('gold_answer') not in {'A', 'B', 'C', 'D'} or source.get('gold_answer') not in source.get('options', {}):
            raise ValueError(f'Invalid source or answer: {source["id"]}')
        for (name, _), row in zip(files.items(), models):
            if any(row.get(key) != source.get(key) for key in fields):
                raise ValueError(f'{name}: content or order differs at row {count}')
            full, prompt, target = (row.get(k) for k in ('full_token_length', 'prompt_token_length', 'target_token_length'))
            if any(type(n) is not int for n in (full, prompt, target)) or not (0 < full <= 512 and prompt > 0 and target > 0 and full == prompt + target):
                raise ValueError(f'{name}: invalid length at row {count}')
            if row['messages'][1]['content'] != f"Answer: {row['gold_answer']}\nExplanation: {row['explanation']}":
                raise ValueError(f'{name}: invalid training target at row {count}')
            maximum[name] = max(maximum[name], full)
    if count != expected_rows:
        raise ValueError(f'Expected {expected_rows} rows, found {count}')
    return {'status': 'PASS', 'rows_per_model': count, 'max_lengths': maximum}

def main():
    print(json.dumps(verify(COMMON, SPECS), indent=2))

if __name__ == '__main__':
    main()
