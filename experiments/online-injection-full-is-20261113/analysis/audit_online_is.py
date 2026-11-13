"""Offline replay of logical IS and physical evidence-bearing native calls."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from summarize_cot_guidance import Decoder, read
from inference_scaling.arllm.types import SequenceSample
from inference_scaling.retrieval.cot_guidance import GuidanceConfig, GuidanceLedger, search
from inference_scaling.retrieval.literature import unrelated
from inference_scaling.retrieval.online_is import OnlineRetrievalConfig
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.shared.rng import SeedStream


def canonical(value):
    """Compare semantic search records, excluding hardware-only slot counters."""
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items() if k != 'forward_token_slots'}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    return value


def same_search_record(left, right):
    """Tokens/choices are exact; tolerate platform libm rounding of weights."""
    if isinstance(left, float) and isinstance(right, float):
        return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-10)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(same_search_record(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same_search_record(a, b) for a, b in zip(left, right, strict=True))
    return type(left) is type(right) and left == right


class ReplayLedger(GuidanceLedger):
    def __init__(self, row, cfg, eos, close):
        backend = SimpleNamespace(model_id='offline', tokenizer=SimpleNamespace(eos_token_id=eos))
        seed = SeedStream(SeedStream(row['seed']).derive('question', row['question_id'])).derive('search')
        super().__init__(backend, None, row['root_ids'], (close,), seed, cfg)
        self.recorded = iter(row['calls'])

    def sample_batch(self, requests):
        samples = []
        for r in requests:
            c = next(self.recorded)
            assert (list(r.prefix), r.seed, r.uniform_offset, r.max_new_tokens, r.request_id) == (
                c['prefix_ids'], c['seed'], c['uniform_offset'], c['max_new_tokens'], c['request_id'])
            self.calls.append(c)
            self.work += len(c['token_ids'])
            samples.append(SequenceSample(tuple(r.prefix), tuple(c['token_ids']), tuple(c['lp']),
                r.sampling.policy_id, self.model_id, r.request_id, c['finish_reason']))
        return samples


def audit_row(row, cfg, rcfg, decode, noise, eos, close, check):
    def ck(condition, label):
        check(condition, row['unit_id']+':'+label)
    root, selected = row['root_ids'], row['candidates'][0]
    ck(root == row['base_ids'], 'original_prompt_only')
    ck(row['carried'] == {'sc': [], 'cn': [], 'lp': []}, 'no_fixed_thought_statistics')
    ck(row['presearch_work'] == 0, 'zero_presearch_generation')
    ck(sum(len(c['token_ids']) for c in row['calls']) == row['search_work'] <= cfg.search_tokens,
       'logical_search_cost')
    ck(len(selected) <= cfg.thinking_tokens, 'retained_thought_cap')
    events = {e['event_id']: e for e in row['events']}
    ck(list(events) == list(range(len(events))), 'unique_ordered_events')
    ck(len({e['branch_sha256'] for e in events.values()}) == len(events), 'one_packet_per_prefix')
    for e in events.values():
        ck(len(e['branch_ids']) == e['position'] and e['position'] in rcfg.positions, 'configured_trigger')
        ck(not {eos, close}.intersection(e['branch_ids']), 'retrieval_before_termination')
        ck(token_hash(e['branch_ids']) == e['branch_sha256'], 'branch_hash')
        ck(e['query'] == row['question']+'\n'+decode(e['branch_ids'][-rcfg.query_tail_tokens:], True).strip(), 'branch_query')
        p = e['packet']
        ck(token_hash(p['ids']) == p['sha256'], 'packet_hash')
        ck(len(e['injected_ids']) == len(p['ids']) == rcfg.packet_tokens, 'fixed_packet_size')
        expected = list(unrelated(p, noise)) if row['arm'] == 'online_is_noise' else p['ids']
        ck(e['injected_ids'] == expected, 'factual_or_noise_body')
        ck(not {eos, close}.intersection(e['injected_ids']), 'no_structural_packet_tokens')
        ck(e['retrieval']['query'] == e['query'] and e['retrieval']['selected_ids'] == [p['id']], 'retrieval_identity')

    def physical(logical, event_ids, *, final=False):
        ck(logical[:len(root)] == root, 'logical_root')
        thought = logical[len(root):]
        valid = [e for e in events.values() if thought[:e['position']] == e['branch_ids']
                 and (not final or e['position'] < len(thought))]
        valid.sort(key=lambda e: e['position'])
        ck([e['event_id'] for e in valid] == event_ids, 'exact_branch_evidence')
        # No trigger may be missing after the corresponding thought boundary.
        terminal_at = next((i for i, t in enumerate(thought) if t in {eos, close}), len(thought))
        positions = [p for p in rcfg.positions if p <= terminal_at and (not final or p < len(thought))]
        if row['arm'] in {'direct', 'closed_is'}:
            positions = []
        ck([e['position'] for e in valid] == positions, 'all_trigger_positions')
        ck(len({e['packet']['id'] for e in valid}) == len(valid), 'no_repeated_window')
        result, previous = list(root), 0
        for e in valid:
            result += thought[previous:e['position']]+e['injected_ids']
            previous = e['position']
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
                if row['arm'] != 'closed_is':
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
    ck(not answer['truncated'], 'no_final_truncation')
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
    ck(times['search_seconds']+times['answer_seconds'] <= row['seconds']+1e-6, 'pipeline_timing')
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
            audit_row(row, cfg, rcfg, decode, prepared['noise_ids'], engine['eos_id'], engine['close_ids'][0], check)
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
