"""Cross-problem style measurements and a fixed, blinded semantic audit sample."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import statistics

import numpy as np
import coding_study as cs

WORK = cs.ROOT / 'style_review'
WORK.mkdir(exist_ok=True)
SEED = 'style-audit-20261106-v1'
manifest, problems, records, servers, attempts = cs.load()
assert len(records) == 3168 and not attempts


def fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest()


def numeric_summary(values):
    return {'mean': statistics.mean(values), 'median': statistics.median(values),
            'q25': float(np.quantile(values, .25)), 'q75': float(np.quantile(values, .75))}


def describe(ids, mode):
    rows = [records[pid, mode] for pid in ids]
    texts = [r['thinking_text'] for r in rows]
    return {'n': len(ids), 'characters': numeric_summary([len(t) for t in texts]),
            'thinking_with_code_fence': sum('```' in t for t in texts),
            'thinking_over_10000_characters': sum(len(t) > 10000 for t in texts),
            'recorded_output_length_exhausted': sum(r['length_exhausted'] for r in rows),
            'single_search_step': sum(len(r.get('trace', {}).get('steps', [])) == 1 for r in rows),
            'new_rollout_evaluations': numeric_summary([
                sum(s.get('rollout_evaluations_performed', 0) for s in r.get('trace', {}).get('steps', []))
                for r in rows])}


def compare(ids, left, right):
    a = np.array([len(records[pid, left]['thinking_text']) for pid in ids])
    b = np.array([len(records[pid, right]['thinking_text']) for pid in ids])
    rng = np.random.default_rng(20261106)
    strata = {}
    for i, pid in enumerate(ids):
        strata.setdefault(problems[pid].get('difficulty') or 'all', []).append(i)
    sampled = np.concatenate([rng.choice(ix, (5000, len(ix)), replace=True) for ix in strata.values()], axis=1)
    ci = np.quantile((b[sampled].sum(axis=1) / a[sampled].sum(axis=1) - 1) * 100, [.025, .975]).tolist()
    return {'n': len(ids), 'reference': left, 'alternative': right,
            'identical_thinking': sum(records[p, left]['thinking_text'] == records[p, right]['thinking_text'] for p in ids),
            'alternative_shorter': int((b < a).sum()), 'alternative_longer': int((b > a).sum()),
            'equal_character_count': int((a == b).sum()),
            'mean_character_difference': float((b - a).mean()),
            'aggregate_character_percent_change': float((b.sum() / a.sum() - 1) * 100),
            'aggregate_percent_change_paired_bootstrap95': ci,
            'median_per_problem_length_ratio': float(np.median(b / a)),
            'median_shared_prefix_fraction_of_shorter': statistics.median(
                len(os.path.commonprefix([records[p, left]['thinking_text'], records[p, right]['thinking_text']]))
                / min(len(records[p, left]['thinking_text']), len(records[p, right]['thinking_text'])) for p in ids)}


pairs = [('base', m) for m in cs.MODES if m != 'base'] + [
    ('is', 'sir-k1'), ('sir-k1', 'met-k1'), ('sir-k2', 'met-k2'),
    ('sir-k1', 'sir-k2'), ('met-k1', 'met-k2'), ('met-k2', 'block-met-k2'),
    ('met-k1-self-certainty', 'met-k2-self-certainty'),
    ('met-k2-self-certainty', 'block-met-k2-self-certainty'),
] + [(m, m + '-self-certainty') for m in ('is', 'met-k1', 'met-k2', 'block-met-k2')]
strata = {'livecodebench': [p for p in problems if problems[p]['dataset'] == 'livecodebench'],
          'humanevalplus': [p for p in problems if problems[p]['dataset'] == 'humanevalplus']}
for diff in ('easy', 'medium', 'hard'):
    strata['livecodebench_' + diff] = [p for p in strata['livecodebench'] if problems[p]['difficulty'] == diff]
summary = {'scope': 'All 264 fixed problems x 12 methods; final selected thinking only',
           'character_definition': 'Python Unicode string length, including spaces and thinking code drafts; not tokens.',
           'bootstrap': '5000 paired problem resamples; preserve difficulty stratum sizes; descriptive conditional intervals.',
           'strata': {name: {'methods': {m: describe(ids, m) for m in cs.MODES},
                            'comparisons': [compare(ids, a, b) for a, b in pairs]}
                      for name, ids in strata.items()}}
cs.save(WORK / 'metrics.json', summary)

protocol = {
    'seed': SEED, 'selection': 'Rank SHA256(seed + | + problem_id) within each stratum; no correctness/length filtering.',
    'strata_quotas': {'livecodebench_easy': 4, 'livecodebench_medium': 5,
                     'livecodebench_hard': 7, 'humanevalplus': 8},
    'comparison': ['is', 'is-self-certainty'],
    'rationale': 'Hold the fast-IS search settings fixed to examine the C-versus-SC writing tendency; other search variants assessed by all-problem identity/length data and illustrative original traces.',
    'rubric': {
        'exploration': {'0': 'One approach; only perfunctory dismissal of naive method', '1': 'Substantive comparison or revision of two approaches', '2': 'Sustained branching among approaches/state designs or repeated route changes'},
        'revisiting': {'0': 'No material return to a settled step', '1': 'One local clarification/correction/return', '2': 'Repeated reconsideration or re-derivation'},
        'walkthrough': {'0': 'No example evaluation', '1': 'Brief example checks', '2': 'Detailed step-by-step simulation or multiple prolonged checks'},
        'implementation': {'0': 'Conceptual plan, no extended draft/representation discussion', '1': 'One substantive code draft or implementation discussion', '2': 'Multiple code drafts or sustained representation/performance/edge-case engineering discussion'},
        'formalization': {'0': 'Direct rule or informal explanation', '1': 'Explicit state/recurrence/invariant/algebraic derivation'},
        'closure': {'0': 'Settles on an approach with an ordinary ending', '1': 'Ends with unresolved alternatives/doubt', '2': 'Text breaks off mid-derivation or mid-code'},
    },
    'procedure': 'Read full thinking; exact common-prefix references allowed only within the pair. Record semantic codes and short text-grounded notes before revealing mode mapping. No keyword-count classification.',
    'limits': 'One assistant reviewer; exploratory fixed sample, not an independent multi-rater study. Counts apply to sample, not population prevalence. Formatting/length are not reasoning quality.'
}
protocol_path = WORK / 'protocol.json'
if protocol_path.exists():
    assert json.loads(protocol_path.read_text(encoding='utf-8')) == protocol
else:
    cs.save(protocol_path, protocol)
packets, mapping = [], {}
for stratum, quota in protocol['strata_quotas'].items():
    chosen = sorted(strata[stratum], key=lambda pid: fingerprint(SEED + '|' + pid))[:quota]
    for pid in chosen:
        modes = ['is', 'is-self-certainty']
        if int(fingerprint(SEED + '|order|' + pid), 16) % 2:
            modes.reverse()
        packet = {'problem_id': pid, 'stratum': stratum, 'prompt': problems[pid]['prompt'], 'cases': []}
        for letter, mode in zip(('A', 'B'), modes):
            row = records[pid, mode]
            case_id = cs.digest([pid, row['thinking_text'], row['submitted_code']])[:20]
            identity = fingerprint(SEED + '|' + pid + '|' + letter)[:12]
            packet['cases'].append({'review_id': identity, 'side': letter, 'thinking_text': row['thinking_text']})
            mapping[identity] = {'case_id': case_id, 'problem_id': pid, 'mode': mode}
        packets.append(packet)
cs.save(WORK / 'packets.json', packets)
cs.save(WORK / 'key.json', mapping)
print(json.dumps({'fixed_sample_problems': len(packets), 'fixed_sample_texts': sum(len(p['cases']) for p in packets),
                  'total_thinking_characters': sum(len(c['thinking_text']) for p in packets for c in p['cases']),
                  'sample_problems': [p['problem_id'] for p in packets]}, ensure_ascii=False))

