"""Exact pre-registered routing of paired native arms, without new model calls."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

try:
    from inference_scaling.retrieval.grounded_route import select_grounded_arm
except ModuleNotFoundError:
    from grounded_route import select_grounded_arm  # Portable packaged policy.
from summarize_flare_gap import bootstrap
from summarize_grounded_retrieval import average, read_jsonl


def evaluate(snapshot, graded_path, registration, output):
    native = read_jsonl(graded_path)
    by_key = {(r['worker'], r['question_id'], r['seed'], r['arm']): r for r in native}
    raw, checks = {}, {}
    primary = json.loads((graded_path.parent/'summary.json').read_text(encoding='utf-8'))
    assert primary['audit_passed'] and primary['stage'] == 'formal'
    assert len(native) == len(by_key) == primary['unique_execution']['conditions'] == 320
    proof = json.loads(registration.read_text(encoding='utf-8'))
    assert proof['before_any_fresh_model_unit']
    for record in proof['records']:
        data = (snapshot/record['worker']/'formal.jsonl').read_bytes()
        prefix = data[:record['prefix_bytes']]
        key = 'preregistered_'+record['worker']
        checks[key+':prefix'] = hashlib.sha256(prefix).hexdigest() == record['prefix_sha256']
        rows = [json.loads(line) for line in prefix.splitlines() if line.strip()]
        seen = {row['unit_id'].split(':')[0] for row in rows if row['kind'] == 'execution_unit'}
        fresh = {row['question_id'] for row in native if row['cohort'] == 'fresh'}
        checks[key+':before_fresh'] = not (seen & fresh)
        checks[key+':seen_questions'] = sorted(seen) == record['seen_model_question_ids']
        checks[key+':measurements'] = sum(row['kind'] == 'measurement' for row in rows) == record['measurements']
    for folder in sorted(snapshot.iterdir()):
        if folder.is_dir() and (folder/'formal.jsonl').exists():
            for row in read_jsonl(folder/'formal.jsonl'):
                if row['kind'] == 'measurement':
                    key = folder.name, row['question_id'], row['seed'], row['arm']
                    assert key not in raw
                    raw[key] = row
    assert set(raw) == set(by_key)
    derived = []
    blocks = sorted({key[:3] for key in raw})
    assert len(blocks) == 40
    for worker, qid, seed in blocks:
        for context in ['static', 'dynamic']:
            anchor = raw[worker, qid, seed, context+'_anchor_sc']
            verified = raw[worker, qid, seed, context+'_verified_is']
            noise = raw[worker, qid, seed, context+'_verified_noise']
            present = bool(verified['result']['evidence_tokens'])
            prefix = f'{worker}:{qid}:{seed}:{context}'
            checks[prefix+':same_gate'] = anchor['result']['evidence_tokens'] == verified['result']['evidence_tokens'] == noise['result']['evidence_tokens']
            checks[prefix+':same_terminal_state'] = anchor['search_bypassed_natural_end'] == verified['search_bypassed_natural_end'] == noise['search_bypassed_natural_end']
            def probe_units(row):
                return [unit for unit in row['execution_units'] if unit.endswith(':probe')]
            checks[prefix+':same_paid_probe'] = probe_units(anchor) == probe_units(verified) == probe_units(noise)
            if present:
                checks[prefix+':same_relevant_root'] = anchor['search']['root_ids'] == verified['search']['root_ids']
            for content in ['relevant', 'unrelated']:
                source_arm = select_grounded_arm(context, present, content)
                selected = by_key[worker, qid, seed, source_arm]
                suffix = '_adaptive_is' if content == 'relevant' else '_adaptive_noise'
                value = {**selected, 'arm': context+suffix, 'content': content,
                    'source_arm': source_arm, 'verified_branch': present, 'derived_dispatch': True,
                    'execution_units': raw[worker, qid, seed, source_arm]['execution_units']}
                checks[prefix+':'+content+':total_cap'] = value['total_generated_tokens'] <= 8192
                derived.append(value)
    groups = defaultdict(list)
    for row in derived:
        groups[row['cohort'], row['arm']].append(row)
    fields = ['f1', 'exact_match', 'work_tokens', 'total_generated_tokens', 'native_pflops', 'seconds', 'verified_branch']
    sums = ['empty_answer', 'eos_inside_thinking', 'immediate_thinking_end', 'immediate_eos', 'forced_thinking_end', 'answer_truncated', 'invalid_extractor', 'invalid_verifiers', 'verifiers']
    summary = [{'cohort': c, 'arm': a, 'conditions': len(values), 'questions': len({r['question_id'] for r in values}),
        **{f: average([r[f] for r in values]) for f in fields}, **{f: sum(r[f] for r in values) for f in sums}}
        for (c, a), values in sorted(groups.items())]
    contrasts = []
    comparisons = [('dynamic_adaptive_is', x) for x in ['dynamic_legacy_sc', 'dynamic_anchor_sc', 'dynamic_verified_is', 'static_adaptive_is', 'dynamic_adaptive_noise', 'static_legacy_sc']]
    comparisons += [('static_adaptive_is', 'static_verified_is'), ('static_adaptive_is', 'static_anchor_sc')]
    for cohort in sorted({r['cohort'] for r in derived}):
        for left, right in comparisons:
            values = defaultdict(lambda: defaultdict(list))
            for row in native+derived:
                if row['cohort'] == cohort and row['arm'] in {left, right}:
                    values[row['question_id']][row['arm']].append(row['f1'])
            differences = [average(v[left])-average(v[right]) for v in values.values() if set(v) == {left, right}]
            contrasts.append({'cohort': cohort, 'left': left, 'right': right, **bootstrap(differences)})
    failed = [k for k, v in checks.items() if not v]
    result = {'audit_passed': not failed, 'audit_checks': len(checks), 'failed': failed,
        'native_conditions': len(native), 'derived_views': len(derived), 'additional_model_calls': 0,
        'checks': checks,
        'registration': proof, 'summary': summary, 'contrasts': contrasts,
        'limits': 'Exact gate-based dispatch among paired arms, not extra independent trials; gates and shared probes are paid. Same question-level seed-averaged bootstrap as the primary analysis.'}
    output.mkdir(parents=True, exist_ok=True)
    (output/'adaptive-summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    (output/'adaptive-graded.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in derived), encoding='utf-8')
    text = ['# Pre-registered gate-routed IS', '', result['limits'], '',
        '| Cohort | Policy | F1 % | Generated tokens | PFLOPs | Verification branch |', '|---|---|---:|---:|---:|---:|']
    for r in summary:
        text.append(f"| {r['cohort']} | {r['arm']} | {100*r['f1']:.3f} | {r['total_generated_tokens']:.1f} | {r['native_pflops']:.3f} | {100*r['verified_branch']:.1f}% |")
    text += ['', 'Paired question bootstrap, difference in F1 percentage points; descriptive, unadjusted.', '']
    for r in contrasts:
        lo, hi = r['bootstrap_95']
        text.append(f"- {r['cohort']}: {r['left']} − {r['right']}: {100*r['difference']:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}], n={r['n_questions']}.")
    text += ['', f"Audit: {not failed}; {len(checks)} checks; no additional generation."]
    (output/'ADAPTIVE.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    print(json.dumps({'audit_passed': not failed, 'checks': len(checks), 'failed': failed, 'derived_views': len(derived)}))
    return not failed


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['snapshot', 'graded', 'registration', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    raise SystemExit(0 if evaluate(a.snapshot, a.graded, a.registration, a.output) else 1)
