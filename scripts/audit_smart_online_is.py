"""Offline policy, branch, RNG, IS-selection and final-readout audit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from audit_online_is import ReplayLedger, audit_row as audit_baseline_row, canonical, same_search_record
from summarize_cot_guidance import Decoder, read
from inference_scaling.retrieval.cot_guidance import GuidanceConfig, search
from inference_scaling.retrieval.online_is import OnlineRetrievalConfig
from inference_scaling.retrieval.smart_online_is import (
    BRIDGE, CHECKPOINTS, FIXED_VARIANTS, RESUME, SIMPLE_QUERY_VARIANTS, UNCERTAIN, VARIANTS,
    SmartOnlineLedger, boundary, clean_question, overlaps,
)
from inference_scaling.retrieval.stepwise import token_hash


def audit_smart_row(row, cfg, rcfg, decode, noise, eos, close, check):
    def ck(condition, label):
        check(condition, row['unit_id']+':'+label)
    variant = row['arm']; root, selected = row['root_ids'], row['candidates'][0]
    positions = CHECKPOINTS if variant in {'boundary', 'adaptive'} else (32, 256) if variant == 'early' else (128, 512)
    rcfg = OnlineRetrievalConfig(positions=positions)
    is_static = False
    ck(row['variant'] == variant, 'variant_identity')
    ck(root == row['base_ids'] and not row['static_packets'] and not row['static_injected_ids'], 'question_only_root')
    ck(row['carried'] == {'sc': [], 'cn': [], 'lp': []} and row['presearch_work'] == 0, 'no_fixed_thought')
    ck(sum(len(c['token_ids']) for c in row['calls']) == row['search_work'] <= cfg.search_tokens, 'logical_search_cost')
    ck(len(selected) <= cfg.thinking_tokens, 'thought_limit')
    ck(decode(row['resume_ids']) == (BRIDGE if variant in {'bridge', 'early'} else RESUME), 'exact_continuation_instruction')
    events = {e['event_id']: e for e in row['events']}
    ck(list(events) == list(range(len(events))), 'ordered_unique_events')
    decisions = row['decisions']
    ck(len({d['branch_sha256'] for d in decisions}) == len(decisions), 'unique_prefix_decisions')
    ck({d['event_id'] for d in decisions if d['event_id'] is not None} == set(events), 'event_decision_bijection')
    for d in decisions:
        branch, position = d['branch_ids'], d['position']
        ck(len(branch) == position and position in positions, 'allowed_checkpoint')
        ck(not {eos, close}.intersection(branch), 'decision_before_terminal')
        ck(token_hash(branch) == d['branch_sha256'], 'decision_prefix_hash')
        phase = 'early' if position <= 160 else 'late'
        text = decode(branch[-256:], True)
        aligned = boundary(text)
        prior = [e for e in events.values() if e['position'] < position and branch[:e['position']] == e['branch_ids']]
        already = any(e['phase'] == phase for e in prior)
        eligible = not already and (variant in FIXED_VARIANTS or aligned or position in {160, 544})
        surprise = None
        if variant == 'adaptive' and phase == 'late' and eligible:
            proxy = SimpleNamespace(root=tuple(root), native_calls=row['native_calls'])
            surprise = SmartOnlineLedger.recent_surprisal(proxy, branch)
        cue = bool(UNCERTAIN.search(text))
        if variant == 'adaptive' and phase == 'late':
            eligible = eligible and (cue or surprise is not None and surprise >= .8)
        ck(d['phase'] == phase and d['boundary'] == aligned, 'decision_phase_boundary')
        ck(d['surprisal_last32'] == surprise and d['uncertainty_cue'] == cue, 'prefix_only_uncertainty')
        ck(d['eligible'] == eligible, 'trigger_policy')
        ck(('retrieval' in d) == eligible, 'retrieval_only_when_eligible')
        if not eligible:
            ck(d['event_id'] is None, 'skipped_checkpoint_no_event')
            continue
        retrieval = d['retrieval']; used = {e['packet']['id'] for e in prior}
        tail = text.strip()
        if variant in SIMPLE_QUERY_VARIANTS:
            ck(retrieval['query'] == row['question']+'\n'+tail, 'original_query')
            chosen = retrieval['selected_ids'][0] if retrieval['selected_ids'] else None
        else:
            expected_queries = [clean_question(row['question']), clean_question(row['question'])+'\n'+tail]
            ck(retrieval['queries'] == expected_queries, 'anchored_queries')
            ck(len(retrieval['rank_lists']) == len(retrieval['searches']) == 2, 'two_searches')
            weights = {}
            for weight, ids, query, log in zip((.75, .25), retrieval['rank_lists'], expected_queries, retrieval['searches'], strict=True):
                ck(log['query'] == query and log['selected_ids'] == ids, 'underlying_search_log')
                ck(not used.intersection(ids) and len(ids) == len(set(ids)) <= 8, 'search_excludes_used_windows')
                for rank, key in enumerate(ids, 1):
                    weights[key] = weights.get(key, 0.)+weight/(20+rank)
            ck(weights == retrieval['weights'], 'fusion_weights')
            ranked = sorted(weights, key=lambda key: (-weights[key], key))
            chosen = next((key for key in ranked if not any(overlaps(key, previous) for previous in used)), None)
            ck(retrieval['selected_ids'] == ([chosen] if chosen else []), 'fusion_and_overlap_selection')
        ck((d['event_id'] is not None) == (chosen is not None), 'inject_iff_eligible_evidence')
        if d['event_id'] is None:
            continue
        e = events[d['event_id']]; p = e['packet']
        ck(e['branch_ids'] == branch and e['branch_sha256'] == d['branch_sha256'], 'event_branch')
        ck(e['position'] == position and e['phase'] == phase, 'event_position')
        ck(p['id'] == chosen and e['retrieval'] == retrieval, 'event_retrieval')
        ck(e['query'] == (row['question'] if variant in SIMPLE_QUERY_VARIANTS else clean_question(row['question']))+'\n'+tail, 'event_query')
        ck(len(p['ids']) == 256 and token_hash(p['ids']) == p['sha256'], 'packet_integrity')
        ck(e['injected_ids'] == p['ids']+row['resume_ids'], 'exact_packet_and_continuation')
        ck(not {eos, close}.intersection(e['injected_ids']), 'no_structural_injection')

    def physical(logical, event_ids, *, final=False):
        ck(logical[:len(root)] == root, 'logical_root')
        thought = logical[len(root):]
        valid = sorted((e for e in events.values() if thought[:e['position']] == e['branch_ids']
                        and (not final or e['position'] < len(thought))), key=lambda e: e['position'])
        ck([e['event_id'] for e in valid] == event_ids, 'exact_branch_events')
        ck(len(valid) <= 2 and len({e['phase'] for e in valid}) == len(valid), 'at_most_one_injection_per_phase')
        stop = next((i for i, t in enumerate(thought) if t in {eos, close}), len(thought))
        for position in positions:
            if position > stop or final and position == len(thought):
                break
            ck(any(d['branch_ids'] == thought[:position] for d in decisions), 'all_checkpoint_decisions_recorded')
        result, previous = list(root), 0
        for e in valid:
            result += thought[previous:e['position']]+e['injected_ids']; previous = e['position']
        return result+thought[previous:]

    native = row['native_calls']
    used = set()
    for c in native:
        ck(physical(c['logical_prefix_ids'], c['event_ids']) == c['prefix_ids'], 'physical_prefix')
        used.update(c['event_ids'])
        ck(token_hash(c['prefix_ids']) == c['prefix_sha256'], 'native_hash')
        ck(0 < len(c['token_ids']) <= c['max_new_tokens'], 'native_length')
        ck(len(c['prefix_ids'])+c['max_new_tokens']+1+cfg.answer_tokens <= cfg.context_tokens, 'reserved_answer_context')
        ck(all(len(c[k]) == len(c['token_ids']) and all(math.isfinite(v) for v in c[k]) for k in ('sc', 'cn', 'lp')), 'native_statistics')
        ck(all(t not in c['token_ids'][:-1] for t in (eos, close)), 'native_terminal_last')
    ck(used == set(events), 'every_retrieval_consumed_by_thinking')
    if row['arm'] != 'direct':
        indices = []
        for logical in row['calls']:
            pieces = [native[i] for i in logical['native_indices']]
            indices += logical['native_indices']
            prefix = list(logical['prefix_ids'])
            produced = 0
            for part in pieces:
                ck(part['logical_request_id'] == logical['request_id'], 'native_request_owner')
                ck(part['logical_prefix_ids'] == prefix, 'append_only_continuation')
                ck(part['seed'] == logical['seed'] and part['uniform_offset'] == logical['uniform_offset']+produced, 'native_rng_offsets')
                length = logical['max_new_tokens']-produced
                if row['arm'] != 'closed_is' and not is_static:
                    next_position = next((p for p in rcfg.positions if p > len(prefix)-len(root)), None)
                    if next_position is not None:
                        length = min(length, next_position-(len(prefix)-len(root)))
                ck(part['max_new_tokens'] == length, 'exact_trigger_boundary')
                prefix += part['token_ids']
                produced += len(part['token_ids'])
            for key in ('token_ids', 'sc', 'cn', 'lp'):
                ck([v for p in pieces for v in p[key]] == logical[key], 'logical_'+key+'_concatenation')
            ck(logical['finish_reason'] == pieces[-1]['finish_reason'], 'logical_stop_reason')
        ck(sorted(indices) == list(range(len(native))), 'all_native_calls_charged_once')
        ck(sum(len(c['token_ids']) for c in native) == row['search_work'], 'native_search_cost')
        ck(row['audit']['steps'][0]['generated_length_before'] == 0, 'IS_starts_at_zero')
        replay = ReplayLedger(row, cfg, eos, close)
        candidates, audit = search(replay, 'is_block_sc')
        ck(next(replay.recorded, None) is None, 'all_IS_calls_replayed')
        ck(canonical(candidates) == row['candidates'], 'selected_thought_replay')
        ck(same_search_record(canonical(audit), canonical(row['audit'])), 'unchanged_IS_full_replay')
    else:
        ck(not native and not events and len(row['calls']) == 1, 'direct_single_thought_call')
        ck(row['calls'][0]['prefix_ids'] == root and row['calls'][0]['token_ids'] == selected, 'direct_context')
    selected_physical = physical(root+selected, row['selected_event_ids'], final=True)
    ck(selected_physical == row['selected_physical_ids'], 'selected_physical_context')
    answer = row['answer']
    ck(answer['eos_inside_thinking'] == (eos in selected), 'thinking_EOS_preserved')
    ck(decode(answer['answer_ids'], True).strip() == answer['answer_text'], 'decoded_answer')
    ck(answer['empty_answer'] == (not answer['answer_text'].strip()), 'empty_flag')
    ck([t for c in answer['calls'] for t in c['token_ids']] == answer['answer_ids'], 'answer_concatenation')
    ck(len(answer['answer_ids']) <= cfg.answer_tokens, 'answer_budget')
    if answer['calls']:
        last = answer['calls'][-1]
        stopped = eos in last['token_ids'] or last['finish_reason'] in {'eos', 'eos_token', 'stop'}
        ck(answer['truncated'] == (not stopped and len(answer['answer_ids']) == cfg.answer_tokens), 'recorded_final_truncation')
    else:
        ck(not answer['truncated'], 'no_readout_truncation_without_calls')
    if eos in selected:
        ck(not answer['calls'] and not answer['answer_ids'], 'no_EOS_repair')
    prefix = selected_physical+([] if selected[-1:] == [close] else [close])
    for c in answer['calls']:
        ck(token_hash(prefix) == c['prefix_sha256'], 'answer_inherits_selected_evidence')
        ck(len(prefix)+len(c['token_ids']) <= cfg.context_tokens, 'answer_context')
        prefix += c['token_ids']
    ck(row['total_generation_tokens'] == row['search_work']+len(answer['answer_ids']), 'complete_generation_cost')
    ck(row['cost']['generated_tokens'] == row['total_generation_tokens'], 'native_counter')
    ck(all(row['cost'].get(k, 0) == 0 for k in ('scored_tokens', 'cancelled_requests', 'preemptions')), 'no_hidden_scoring')
    times = row['stage_seconds']
    ck(times['search_seconds']+times['answer_seconds']+times.get('static_retrieval_seconds', 0.) <= row['seconds']+1e-6, 'pipeline_timing')
    ck(0 <= times['materialization_seconds'] <= times['search_seconds'], 'retrieval_timing_subset')


def audit_snapshot(folder, stage, frozen_path, output):
    rows = read(folder/(stage+'.jsonl'))
    meta = next(r for r in rows if r['kind'] == 'metadata')
    engine = next(r for r in rows if r['kind'] == 'engine')
    cfg = GuidanceConfig(**meta['config'])
    rcfg = OnlineRetrievalConfig(**{**meta['retrieval_config'], 'positions': tuple(meta['retrieval_config']['positions'])})
    prepared_path = folder/('prepared-'+stage+'.json')
    prepared = json.loads(prepared_path.read_text(encoding='utf-8'))
    cases = {c['id']: c for c in prepared['cases']}
    decode = Decoder(json.loads((folder/('vocabulary-'+stage+'.json')).read_text(encoding='utf-8')))
    checks, failures = 0, []
    def check(condition, label):
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(label)
    check(meta['config'] == asdict(GuidanceConfig()), 'unchanged_IS_guidance_parameters')
    check(canonical(meta['retrieval_config']) == canonical(asdict(OnlineRetrievalConfig())), 'frozen_retrieval_config')
    frozen = json.loads(frozen_path.read_text(encoding='utf-8'))
    check(all(meta['source'].get(n) == h for n, h in frozen['files'].items()), 'frozen_17_IS_files')
    check(hashlib.sha256(frozen_path.with_name('settings.json').read_bytes()).hexdigest() == meta['settings_sha256'],
          'unchanged_model_settings')
    check(hashlib.sha256(prepared_path.read_bytes()).hexdigest() == meta['prepared_sha256'], 'prepared_hash')
    measures = [r for r in rows if r['kind'] == 'measurement']
    expected = {(q, s, arm) for q in meta['cases'] for s in meta['seeds'] for arm in meta['arms']}
    actual = [(r['question_id'], r['seed'], r['arm']) for r in measures]
    check(len(actual) == len(set(actual)) and set(actual) == expected, 'exact_complete_cohort')
    check([r['units'] for r in rows if r['kind'] == 'completed'] == [len(expected)], 'completion_marker')
    check(not any(r['kind'] == 'failed' for r in rows), 'native_no_failure')
    finish = json.loads((folder/(stage+'-finish.json')).read_text(encoding='utf-8'))
    check(finish['exit'] == 0 and not finish['state']['Running'], 'own_container_stopped')
    for row in measures:
        check(row['base_ids'] == cases[row['question_id']]['prompt_ids'], row['unit_id']+':prepared_root')
        row = {**row, 'question': cases[row['question_id']]['question']}
        try:
            (audit_smart_row if row['arm'] in VARIANTS else audit_baseline_row)(row, cfg, rcfg, decode, prepared['noise_ids'], engine['eos_id'], engine['close_ids'][0], check)
        except Exception as exc:
            check(False, row['unit_id']+':audit_exception:'+repr(exc))
    result = {'passed': not failures, 'checks': checks, 'failures': failures,
        'units': len(measures), 'stage': stage, 'quality_scored': False,
        'generated_tokens': sum(r['total_generation_tokens'] for r in measures),
        'empty_answers': sum(r['answer']['empty_answer'] for r in measures),
        'truncated_answers': sum(r['answer']['truncated'] for r in measures),
        'retrieval_events': sum(len(r['events']) for r in measures),
        'selected_injections': sum(len(r['selected_event_ids']) for r in measures)}
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    if failures:
        raise ValueError(failures[:20])
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--stage', default='smoke')
    p.add_argument('--frozen', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(audit_snapshot(a.folder, a.stage, a.frozen, a.output)))
