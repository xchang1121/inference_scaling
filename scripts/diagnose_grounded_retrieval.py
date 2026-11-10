"""Offline evidence provenance and fixed-case review; never supplied to generation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from summarize_flare_gap import grade
from summarize_grounded_retrieval import average, read_jsonl


def diagnose(snapshot, gold_path, output, stage='formal'):
    gold = {x['id']: x for x in read_jsonl(gold_path)}
    results, gate_rows, review, context_pairs = [], [], [], []
    fixed_fresh = {'DEV_Q016', 'DEV_Q008', 'DEV_Q022'}
    for folder in sorted(snapshot.iterdir()):
        if not folder.is_dir() or not (folder/(stage+'.jsonl')).exists():
            continue
        rows = read_jsonl(folder/(stage+'.jsonl'))
        cases = {x['id']: x for x in json.loads((folder/('prepared-'+stage+'.json')).read_text(encoding='utf-8'))['cases']}
        units = {x['unit_id']: x for x in rows if x['kind'] == 'execution_unit'}
        context_blocks = defaultdict(dict)
        for row in (r for r in rows if r['kind'] == 'measurement'):
            qid = row['question_id']
            parents = set(gold[qid]['document_ids'])
            used = [units[u] for u in row['execution_units']]
            events = [e for u in used for e in u['result']['events'] if u['unit_id'].endswith(':probe')]
            gates = [e for e in events if e['kind'] == 'grounded_evidence']
            injections = [e for e in row['result']['events'] if e['kind'] == 'injection']
            packet = injections[-1]['packet'] if injections else {}
            retrieved = set(packet.get('parent_ids', []))
            exposed = set()
            for e in gates:
                retrieved.update(e['candidates'].get('parent_ids', []))
                exposed.update(s['id'].split('@')[0] for s in e['candidates']['sources'])
            selected = {b['id'].split('@')[0] for b in packet.get('bodies', [])}
            real_quote = '\n'.join(b['text'] for b in packet.get('bodies', []))
            # Noise packet metadata retains original provenance; never grade it as exposed real text.
            quote_f1 = grade(real_quote, gold[qid]['answer'])['f1'] if row['content'] == 'relevant' and packet else None
            verdict = row['search'].get('selected_verdict')
            parsed = verdict if isinstance(verdict, dict) and 'coverage' in verdict else None
            value = {k: row[k] for k in ['worker', 'question_id', 'cohort', 'seed', 'arm', 'content']}
            value.update({'candidate_gold_parent_hit': bool(parents & retrieved),
                'exposed_gold_parent_hit': bool(parents & exposed),
                'selected_gold_parent_hit': bool(parents & selected),
                'selected_real_quote_gold_f1': quote_f1,
                'selected_verifier_valid': parsed is not None,
                'selected_verifier_positive_coverage': bool(parsed and parsed['coverage'] > 0),
                'selected_verifier_clean_full_coverage': bool(parsed and parsed['coverage'] == 2
                    and not any(parsed[k] for k in ['unsupported', 'scope_error', 'contradiction'])),
                'extractor_errors': [e.get('error') for e in gates if e.get('error')],
                'selected_verdict': verdict,
                'gate_count': len(gates), 'evidence_present': bool(packet)})
            results.append(value)
            context_blocks[qid, row['seed'], row['method'], row['content']][row['context']] = {
                'row': row, 'payload': injections[-1]['payload_ids'] if injections else [],
                'generated_tokens': sum(len(c['token_ids']) for u in used for c in u['calls']),
                'native_pflops': sum(u.get('graph_execution', {}).get('estimated_dense_padded_flops', 0) for u in used)/1e15,
                'seconds': sum(u['seconds'] for u in used)}
            if row['cohort'] == 'regression' or qid in fixed_fresh or stage == 'smoke':
                review.append({**value, 'question': cases[qid]['question'], 'gold': gold[qid],
                    'answer': row['result']['answer_text'], 'evidence_metadata': packet,
                    'gate_events': gates})
        for (qid, seed, method, content), pair in sorted(context_blocks.items()):
            if set(pair) != {'static', 'dynamic'}:
                continue
            static, dynamic = pair['static'], pair['dynamic']
            sr, dr = static['row'], dynamic['row']
            comparable = 'root_ids' in sr['search'] and 'root_ids' in dr['search']
            context_pairs.append({'worker': folder.name, 'question_id': qid, 'cohort': sr['cohort'],
                'seed': seed, 'method': method, 'content': content, 'search_roots_comparable': comparable,
                'same_search_root': sr['search']['root_ids'] == dr['search']['root_ids'] if comparable else None,
                'same_nonempty_reference': bool(static['payload']) and static['payload'] == dynamic['payload'],
                'same_answer_tokens': sr['result']['answer_ids'] == dr['result']['answer_ids'],
                'same_answer_text': sr['result']['answer_text'] == dr['result']['answer_text'],
                'dynamic_prefix_reset': any('discarded_confirmed_ids' in e for e in dr['result']['events']),
                **{'dynamic_minus_static_'+field: dynamic[field]-static[field] for field in ['generated_tokens', 'native_pflops', 'seconds']}})
        for uid, unit in units.items():
            if not uid.endswith(':probe'):
                continue
            qid, seed = uid.split(':')[:2]
            for event in unit['result']['events']:
                if event['kind'] == 'grounded_evidence':
                    gate_rows.append({'worker': folder.name, 'question_id': qid, 'seed': int(seed),
                        'unit_id': uid, 'accepted': bool(event.get('packet')),
                        'error': event.get('error'), 'raw': event.get('raw'),
                        'selected_support': event.get('support')})
    groups = defaultdict(list)
    for row in results:
        groups[(row['cohort'], row['arm'])].append(row)
    metrics = ['candidate_gold_parent_hit', 'exposed_gold_parent_hit', 'selected_gold_parent_hit',
        'selected_real_quote_gold_f1', 'selected_verifier_valid', 'selected_verifier_positive_coverage',
        'selected_verifier_clean_full_coverage', 'gate_count', 'evidence_present']
    summary = [{'cohort': c, 'arm': a, 'conditions': len(v),
        **{k: average([r[k] for r in v if r[k] is not None]) for k in metrics}}
        for (c, a), v in sorted(groups.items())]
    result = {'stage': stage, 'summary': summary, 'unique_gates': len(gate_rows),
        'unique_accepted_gates': sum(g['accepted'] for g in gate_rows),
        'unique_gate_errors': dict(Counter(g['error'] for g in gate_rows if g['error'])),
        'fixed_fresh_review_ids': sorted(fixed_fresh),
        'limits': ['All gold comparisons are post-generation diagnostics.',
            'Parent hits and quote lexical overlap do not establish answer support.',
            'exposed_gold_parent_hit refers to anchored extractor sources; legacy arms have no such source list.',
            'Noise retains source IDs/titles in its matched wrapper; selected parent metadata is not real evidence.',
            'Unscored quote and verifier fields are zero in aggregate; use evidence_present and selected_verifier_valid as denominators.']}
    paired_groups = defaultdict(list)
    for row in context_pairs:
        paired_groups[row['cohort'], row['method'], row['content']].append(row)
    result['context_comparisons'] = [{'cohort': c, 'method': m, 'content': t, 'condition_pairs': len(values),
        **{field: sum(bool(r[field]) for r in values) for field in ['search_roots_comparable', 'same_search_root', 'same_nonempty_reference', 'same_answer_tokens', 'same_answer_text', 'dynamic_prefix_reset']},
        **{field: average([r[field] for r in values]) for field in ['dynamic_minus_static_generated_tokens', 'dynamic_minus_static_native_pflops', 'dynamic_minus_static_seconds']}}
        for (c, m, t), values in sorted(paired_groups.items())]
    result['limits'].append('Context comparisons are descriptive paired-condition diagnostics; identical answers do not prove equal accuracy, and seeds are not independent questions.')
    output.mkdir(parents=True, exist_ok=True)
    (output/'evidence-diagnostics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for name, values in [('evidence-conditions.jsonl', results), ('unique-gates.jsonl', gate_rows), ('fixed-case-review.jsonl', review), ('context-comparisons.jsonl', context_pairs)]:
        (output/name).write_text(''.join(json.dumps(x, ensure_ascii=False)+'\n' for x in values), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['unique_gates', 'unique_accepted_gates', 'unique_gate_errors']}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--gold', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--stage', choices=['smoke', 'formal'], default='formal')
    a = p.parse_args()
    diagnose(a.snapshot, a.gold, a.output, a.stage)
