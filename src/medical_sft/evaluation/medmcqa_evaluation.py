"""MedMCQA internal-validation and official-test loading/scoring protocol.

The internal validation split is drawn from official Train; official Validation
is the first final test set. MedExpQA's full 618-row evaluation set is loaded by
medical_sft.evaluation.medexpqa_evaluation and is never a training source.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from medical_sft.data.medmcqa_answers import parse_answer as parse_gold

from medical_sft.paths import ROOT
PROTOCOL_VERSION = 'medmcqa-final-eval-v3'
INTERNAL_PATH = 'data/processed/validation/internal_medmcqa/dev_eval.jsonl'
RETIRED_PATH = 'data/processed/dev_sft.jsonl'
SOURCES = {
    'dev': ('dev_eval', 'MedMCQA Internal Validation Accuracy'),
    'validation': ('reserved', 'MedMCQA Official Validation Test Accuracy'),
}
MODEL_REVISIONS = {
    'Qwen/Qwen2.5-3B-Instruct': 'aa8e72537993ba99e69dfaafa59ed015b17504d1',
    'Qwen/Qwen3.5-4B': '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a',
}
MEDEXPQA_EVALUATION_PATHS = (
    'data/evaluation/medexpqa_full/medexpqa_train_candidates_512.jsonl',
    'data/evaluation/medexpqa_full/medexpqa_over_512_isolated.jsonl',
    'data/evaluation/medexpqa_full/medexpqa_quality_excluded.jsonl',
    'data/evaluation/medexpqa_full/validation_preserved.jsonl',
    'data/evaluation/medexpqa_full/test_preserved.jsonl',
)
ANSWER = re.compile(r'Answer:[ \t]*([A-D])(?:[ \t]*\r?\n[ \t]*Explanation:[\s\S]*)?[ \t]*', re.I)
LEGACY_ANSWER_ONLY_SUFFIX = "Return exactly one line in the format: Answer: X"
FINAL_SFT_INSTRUCTION = (
    "Answer the question and provide a concise explanation in English. "
    "Use exactly this format:\nAnswer: X\nExplanation: <brief explanation>"
)


def final_sft_user_message(row):
    """Rebuild the exact Answer + Explanation instruction used by final SFT."""
    prompt = row.get("prompt")
    suffix = "\n" + LEGACY_ANSWER_ONLY_SUFFIX
    if not isinstance(prompt, str) or not prompt.endswith(suffix):
        raise ValueError("Evaluation prompt does not match the locked legacy source format")
    question_and_options = prompt[:-len(suffix)]
    if not question_and_options.strip():
        raise ValueError("Evaluation prompt has no question/options content")
    return {
        "role": "user",
        "content": question_and_options + "\n" + FINAL_SFT_INSTRUCTION,
    }


def is_relative_to(path, parent):
    """Compatibility helper for Python versions without Path.is_relative_to."""
    try:
        Path(path).relative_to(Path(parent))
        return True
    except ValueError:
        return False


def active_artifacts(manifest):
    """The old alias entry remains historical provenance, not an active input."""
    return {role: info for role, info in manifest['artifacts'].items() if role != 'dev_sft'}


def gold_letter(value):
    status, letter = parse_gold(value)
    if status != 'valid':
        raise ValueError('No valid public gold label; official Test (-1) cannot be scored locally')
    return letter


def parse_model_answer(output):
    """Accept the frozen Answer line, optionally followed by Explanation."""
    if not isinstance(output, str):
        return None
    text = output.strip()
    match = ANSWER.fullmatch(text)
    if not match:
        return None
    # A second answer declaration is ambiguous, even if buried in Explanation.
    if len(re.findall(r'(?im)^\s*(?:final\s+)?answer\s*:', text)) != 1:
        return None
    return match.group(1).upper()


def evaluation_messages(row):
    return [{'role': 'user', 'content': row['prompt']},
            {'role': 'assistant', 'content': 'Answer: ' + gold_letter(row['gold'])}]


def load_evaluation(split=None, input_path=None, root=ROOT):
    root = Path(root).resolve()
    manifest = json.loads((root / 'data_release_manifest.json').read_text(encoding='utf-8'))
    choices = {key: (root / manifest['artifacts'][role]['path']).resolve()
               for key, (role, _) in SOURCES.items()}
    if input_path is not None:
        supplied = Path(input_path)
        supplied = (root / supplied).resolve()
        matches = [key for key, path in choices.items() if path == supplied]
        if len(matches) != 1 or (split is not None and split != matches[0]):
            raise ValueError('Use dev_eval or official Validation only; mixed/unknown/Test sources are forbidden')
        split = matches[0]
    split = split or 'dev'
    if split not in SOURCES:
        raise ValueError('Official Test cannot be used for local supervised accuracy')
    role, metric = SOURCES[split]
    info, path = manifest['artifacts'][role], choices[split]
    rows, ids = [], set()
    with path.open(encoding='utf-8') as f:
        for number, line in enumerate(f, 1):
            row = json.loads(line)
            if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in ids:
                raise ValueError(f'Missing or duplicate evaluation ID on line {number}')
            ids.add(row['id'])
            if not isinstance(row.get('prompt'), str) or not row['prompt'].strip():
                raise ValueError('Missing evaluation prompt')
            row = dict(row, gold=gold_letter(row.get('gold', row.get('cop'))))
            rows.append(row)
    if len(rows) != info['rows']:
        raise ValueError('Evaluation record count mismatch')
    return rows, {
        'split': split,
        'evaluation_role': 'internal_validation' if split == 'dev' else 'final_test',
        'source_split': 'official_train' if split == 'dev' else 'official_validation',
        'metric_name': metric,
        'input_path': path,
        'official_validation_used': split == 'validation',
        'protocol_version': PROTOCOL_VERSION,
    }


def score_predictions(golds, predictions):
    if len(golds) != len(predictions):
        raise ValueError('Gold/prediction count mismatch')
    valid, invalid_gold, correct, invalid_predictions = 0, 0, 0, 0
    for gold, prediction in zip(golds, predictions):
        try:
            letter = gold_letter(gold)
        except ValueError:
            invalid_gold += 1
            continue
        valid += 1
        parsed = prediction if isinstance(prediction, str) and prediction in 'ABCD' and len(prediction) == 1 else parse_model_answer(prediction)
        invalid_predictions += parsed is None
        correct += parsed == letter
    if not valid:
        raise ValueError('No valid gold labels: do not compute an Official Test accuracy')
    return {'valid_gold_records': valid, 'invalid_gold_records': invalid_gold, 'correct': correct,
            'invalid_predictions': invalid_predictions, 'accuracy': correct / valid}


def assert_training_safe(path, training_ids=(), root=ROOT):
    """Reject evaluation paths and any training IDs present in all three evaluation sources."""
    root, path = Path(root).resolve(), Path(path).resolve()
    manifest = json.loads((root / 'data_release_manifest.json').read_text(encoding='utf-8'))
    banned = {(root / manifest['artifacts'][role]['path']).resolve()
              for role in ('dev_eval', 'reserved', 'raw.validation')}
    banned.update((root / relative).resolve() for relative in MEDEXPQA_EVALUATION_PATHS)
    banned.add((root / RETIRED_PATH).resolve())
    banned_dirs = (
        root / 'data/evaluation/medmcqa_official_validation',
        root / 'data/evaluation/medexpqa_full',
    )
    if path in banned or any(is_relative_to(path, directory) for directory in banned_dirs):
        raise ValueError('Evaluation data or a historical evaluation alias cannot be a training input')
    protected = set()
    for split in SOURCES:
        rows, _ = load_evaluation(split, root=root)
        protected.update(str(row['id']) for row in rows)
    medexpqa_ids = set()
    for relative in MEDEXPQA_EVALUATION_PATHS:
        evaluation_path = root / relative
        if not evaluation_path.is_file():
            raise ValueError(f'MedExpQA evaluation source is missing: {relative}')
        with evaluation_path.open(encoding='utf-8') as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    raise ValueError(f'Blank MedExpQA evaluation record: {relative}:{line_number}')
                row = json.loads(line)
                if not isinstance(row.get('id'), (str, int)) or not str(row['id']):
                    raise ValueError(f'Missing MedExpQA evaluation ID: {relative}:{line_number}')
                sample_id = str(row['id'])
                if sample_id in medexpqa_ids:
                    raise ValueError(f'Duplicate MedExpQA evaluation ID: {sample_id}')
                medexpqa_ids.add(sample_id)
    protected.update(medexpqa_ids)
    overlap = protected & {str(sid) for sid in training_ids}
    if overlap:
        raise ValueError('Training/evaluation ID overlap: ' + ', '.join(sorted(overlap)))


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=('dev', 'validation'), default='validation')
    args = parser.parse_args()
    rows, info = load_evaluation(args.split)
    print(json.dumps({'split': args.split, 'rows': len(rows), 'role': info['evaluation_role']}))


if __name__ == '__main__':
    main()
