"""Bind qualitative notes to the exact saved answers; do not score their truth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def validate(notes_path, cases_path, manifest_path, diagnostics_path, output):
    notes, cases = lines(notes_path), lines(cases_path)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    diagnostics = json.loads(diagnostics_path.read_text(encoding='utf-8'))
    expected = set(manifest['regression_ids']) | set(diagnostics['fixed_fresh_review_ids'])
    assert {row['question_id'] for row in notes} == expected and len(notes) == len(expected) == 11
    lookup = {(r['question_id'], r['seed'], r['arm']): r for r in cases}
    arms = {'static_verified_is', 'dynamic_legacy_sc', 'dynamic_anchor_sc', 'dynamic_verified_is', 'dynamic_verified_noise'}
    checked = []
    for note in notes:
        assert note['seed'] == 20261110 and not note['method_changed_after_review']
        assert note['observations'] and note['review_scope']
        assert set(note['condition_answer_sha256']) == arms
        for arm, digest in sorted(note['condition_answer_sha256'].items()):
            key = note['question_id'], note['seed'], arm
            row = lookup[key]
            assert row['cohort'] == note['cohort']
            assert hashlib.sha256(row['answer'].encode()).hexdigest() == digest, key
            checked.append({'question_id': key[0], 'seed': key[1], 'arm': arm, 'answer_sha256': digest})
    receipt = {'references_verified': True, 'questions': len(notes), 'answer_references': len(checked),
        'regression_questions': len(manifest['regression_ids']),
        'fixed_fresh_questions': len(diagnostics['fixed_fresh_review_ids']),
        'scope': 'This audit verifies answer identity and the fixed review sample, not the truth of qualitative judgments or a population accuracy rate.',
        'notes_sha256': hashlib.sha256(notes_path.read_bytes()).hexdigest(),
        'checked': sorted(checked, key=lambda r: (r['question_id'], r['seed'], r['arm']))}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: receipt[k] for k in ['references_verified', 'questions', 'answer_references']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['notes', 'cases', 'manifest', 'diagnostics', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    validate(args.notes, args.cases, args.manifest, args.diagnostics, args.output)
