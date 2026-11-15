"""Replay causal gate decisions and unchanged IS from immutable native traces."""
from types import SimpleNamespace
import math
import types

from audit_online_is import ReplayLedger, canonical, same_search_record
from inference_scaling.retrieval.cot_guidance import search
from inference_scaling.retrieval.smart_online_is import boundary, clean_question, BRIDGE, RESUME
from inference_scaling.retrieval.tuned_online_is import TuningPolicy, TunedOnlineLedger, overlaps, last_complete_step
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.retrieval.attention_policy import attention_feature


def audit_tuned_row(row, cfg, decode, eos, close, check):
    def ck(ok, label):
        check(ok, row['unit_id']+':'+label)
    p = TuningPolicy(**row['policy'])
    root, selected = row['root_ids'], row['candidates'][0]
    native, events = row['native_calls'], {e['event_id']:e for e in row['events']}
    auxiliary = None
    if 'auxiliary_policy' in row:
        from audit_auxiliary_online_is import prepare_auxiliary_audit
        auxiliary = prepare_auxiliary_audit(row,cfg,decode,eos,close,ck)
    if 'gap_policy' in row:
        from audit_gap_online_is import prepare_gap_audit
        ck(auxiliary is None,'single_paid_controller')
        auxiliary = prepare_gap_audit(row,cfg,decode,eos,close,ck)
    ck(root == row['base_ids'], 'question_only_root')
    ck(not row['static_packets'] and not row['static_injected_ids'], 'no_hidden_static_context')
    ck(row['presearch_work'] == 0 and row['carried'] == {'sc': [], 'cn': [], 'lp': []}, 'from_token_zero')
    ck(list(events) == list(range(len(events))), 'unique_event_ids')
    ck(len({d['branch_sha256'] for d in row['decisions']}) == len(row['decisions']), 'unique_decisions')
    ck({d['event_id'] for d in row['decisions'] if d['event_id'] is not None} == set(events), 'decision_event_bijection')
    for d in row['decisions']:
        branch, position = d['branch_ids'], d['position']
        ck(len(branch) == position and position in p.positions, 'checkpoint')
        ck(not {eos,close}.intersection(branch), 'decision_before_terminal')
        ck(token_hash(branch) == d['branch_sha256'], 'decision_hash')
        n = d['observed_native_calls']
        ck(0 <= n <= len(native), 'causal_call_cutoff')
        previous = sorted((e for e in events.values() if e['position'] < position
                           and branch[:e['position']] == e['branch_ids']), key=lambda e:e['position'])
        tail_size = max(p.query_tail, p.rule_tail) if p.trigger in {'rule', 'hybrid'} else p.query_tail
        text = decode(branch[-tail_size:], True)
        eligible = len(previous) < p.max_insertions and (not previous or position-previous[-1]['position'] >= p.cooldown)
        reason = {}
        proxy = SimpleNamespace(root=tuple(root), native_calls=native[:n], policy=p,
                                backend=SimpleNamespace(decode=lambda ids,skip_special_tokens=True:decode(ids,skip_special_tokens)),
                                tap=SimpleNamespace(history={c['request_id']:c['internal_signals']
                                    for c in native[:n] if 'internal_signals' in c}))
        proxy.recent = types.MethodType(TunedOnlineLedger.recent, proxy)
        if eligible:
            eligible, reason = auxiliary[0](d) if auxiliary else TunedOnlineLedger.signal(proxy, branch, text)
        if p.wait_for_boundary:
            eligible = eligible and (boundary(text) or position in p.boundary_deadlines)
        ck(d['boundary'] == boundary(text), 'boundary_predicate')
        ck(d['triggered'] == eligible and d['signal'] == reason, 'causal_gate_replay')
        ck(('retrieval' in d) == eligible, 'retrieval_only_if_triggered')
        if not eligible:
            ck(d['event_id'] is None, 'skip_has_no_injection')
            continue
        tail = decode(branch[-p.query_tail:], True).strip()
        if p.query == 'uncertain':
            states = proxy.recent(branch, p.query_tail)
            tail = decode([r['token'] for r in states if r['lp'] < math.log(p.low_probability)], True)
        if p.query == 'attention':
            tail = attention_feature(proxy.recent(branch,p.window),proxy.backend.decode,
                query_tokens=p.attention_query_tokens,content_only=p.attention_content_only,expand_words=p.attention_expand_words)['query_text']
        query = clean_question(row['question']) if p.query == 'question' else row['question']+'\n'+tail
        if p.query in {'sentence','sentence_anchored'}:
            step=last_complete_step(tail)
            query=step if p.query=='sentence' else clean_question(row['question'])+'\n'+step
        if auxiliary:
            query = auxiliary[1](d) or query
        retrieval = d['retrieval']
        ck(retrieval['query'] == query, 'causal_query')
        used = {e['packet']['id'] for e in previous}
        if p.query == 'anchored':
            scores = {}
            components = retrieval['components']
            ck(len(components) == 2, 'two_query_fusion')
            ck([c['query'] for c in components] == [clean_question(row['question']), tail or clean_question(row['question'])], 'fusion_queries')
            for weight, component in zip((p.question_weight,1-p.question_weight),components,strict=True):
                for rank, name in enumerate(component['selected_ids'],1):
                    scores[name] = scores.get(name,0)+weight/(20+rank)
            ck(scores == retrieval['rrf_scores'], 'fusion_scores')
            ranked = sorted(scores,key=lambda x:(-scores[x],x))
        else:
            ranked = retrieval['selected_ids']
        ck(not used.intersection(ranked), 'used_windows_excluded')
        chosen = next((k for k in ranked if not any(overlaps(k,old,p.overlap) for old in used)),None)
        if auxiliary:
            chosen = auxiliary[3](d,ranked,used,chosen)
        ck((d['event_id'] is not None) == (chosen is not None), 'inject_iff_evidence_available')
        if chosen is None:
            continue
        e = events[d['event_id']]
        ck(e['branch_ids'] == branch and e['branch_sha256'] == d['branch_sha256'], 'event_prefix')
        ck(e['position'] == position and e['query'] == query and e['retrieval'] == retrieval, 'event_decision_identity')
        payload = e['packet']
        ck(payload['id'] == chosen and token_hash(payload['ids']) == payload['sha256'], 'chosen_packet_integrity')
        ck(len(payload['ids']) == p.packet_tokens, 'packet_budget')
        continuation = {'none':'','resume':RESUME,'bridge':BRIDGE,'custom':p.continuation_text}[p.continuation]
        ck(e['injected_ids'][:p.packet_tokens] == payload['ids'], 'packet_prefix')
        ck(decode(e['injected_ids'][p.packet_tokens:]) == continuation, 'exact_continuation')
        ck(not {eos,close}.intersection(e['injected_ids']), 'no_structural_injection')

    def physical(logical, ids, final=False):
        ck(logical[:len(root)] == root, 'logical_root')
        thought = logical[len(root):]
        terminal = next((i for i,t in enumerate(thought) if t in {eos,close}),len(thought))
        decision_keys = {d['branch_sha256'] for d in row['decisions']}
        for position in p.positions:
            if position <= terminal and (not final or position < len(thought)):
                ck(token_hash(thought[:position]) in decision_keys, 'every_checkpoint_decided')
        relevant = sorted((e for e in events.values() if thought[:e['position']] == e['branch_ids']
                           and (not final or e['position'] < len(thought))),key=lambda e:e['position'])
        ck([e['event_id'] for e in relevant] == ids, 'exact_branch_evidence')
        ck(len(relevant) <= p.max_insertions, 'insertion_limit')
        result, before = list(root), 0
        for e in relevant:
            result += thought[before:e['position']]+e['injected_ids']
            before = e['position']
        return result+thought[before:]

    used = set()
    for call in native:
        ck(physical(call['logical_prefix_ids'],call['event_ids']) == call['prefix_ids'], 'native_physical_prefix')
        used.update(call['event_ids'])
        ck(token_hash(call['prefix_ids']) == call['prefix_sha256'], 'native_hash')
        ck(0 < len(call['token_ids']) <= call['max_new_tokens'], 'native_length')
        ck(len(call['prefix_ids'])+call['max_new_tokens']+1+cfg.answer_tokens <= cfg.context_tokens, 'answer_context_reserved')
        ck(all(len(call[k]) == len(call['token_ids']) and all(math.isfinite(v) for v in call[k]) for k in ('sc','cn','lp')), 'native_statistics')
        ck(not {eos,close}.intersection(call['token_ids'][:-1]), 'terminal_last')
        if 'internal_signals' in call:
            ck(len(call['internal_signals']) == len(call['token_ids']), 'internal_signal_alignment')
    ck(used == set(events), 'all_injections_consumed')
    indices = []
    for logical in row['calls']:
        pieces = [native[i] for i in logical['native_indices']]
        indices += logical['native_indices']
        prefix, produced = list(logical['prefix_ids']),0
        for part in pieces:
            ck(part['logical_request_id'] == logical['request_id'], 'native_owner')
            ck(part['logical_prefix_ids'] == prefix, 'append_only_continuation')
            ck(part['seed'] == logical['seed'] and part['uniform_offset'] == logical['uniform_offset']+produced, 'native_rng_offsets')
            remaining = logical['max_new_tokens']-produced
            checkpoint = next((x for x in p.positions if x > len(prefix)-len(root)),None)
            if checkpoint is not None:
                remaining = min(remaining,checkpoint-(len(prefix)-len(root)))
            ck(part['max_new_tokens'] == remaining, 'exact_checkpoint_segmentation')
            prefix += part['token_ids']
            produced += len(part['token_ids'])
        for key in ('token_ids','sc','cn','lp'):
            ck([v for c in pieces for v in c[key]] == logical[key], 'concatenation_'+key)
        ck(logical['finish_reason'] == pieces[-1]['finish_reason'], 'logical_stop_reason')
    ck(sorted(indices) == list(range(len(native))), 'native_calls_charged_once')
    ck(sum(len(c['token_ids']) for c in native) == row['search_work'] <= cfg.search_tokens, 'search_work')
    replay = ReplayLedger(row,cfg,eos,close)
    candidates, audit = search(replay,'is_block_sc')
    ck(next(replay.recorded,None) is None, 'all_IS_calls_replayed')
    ck(canonical(candidates) == row['candidates'], 'selected_candidates_replayed')
    ck(same_search_record(canonical(audit),canonical(row['audit'])), 'frozen_IS_replayed')
    ck(row['audit']['steps'][0]['generated_length_before'] == 0, 'IS_starts_at_zero')
    selected_physical = physical(root+selected,row['selected_event_ids'],True)
    ck(selected_physical == row['selected_physical_ids'], 'selected_context')
    answer = row['answer']
    ck(answer['eos_inside_thinking'] == (eos in selected), 'thinking_EOS')
    ck(decode(answer['answer_ids'],True).strip() == answer['answer_text'], 'decoded_answer')
    ck(answer['empty_answer'] == (not answer['answer_text'].strip()), 'empty_flag')
    ck([t for c in answer['calls'] for t in c['token_ids']] == answer['answer_ids'], 'answer_concatenation')
    ck(len(answer['answer_ids']) <= cfg.answer_tokens, 'answer_budget')
    if eos in selected:
        ck(not answer['calls'], 'no_EOS_repair')
    prefix = selected_physical+([] if selected[-1:] == [close] else [close])
    for c in answer['calls']:
        ck(token_hash(prefix) == c['prefix_sha256'], 'selected_evidence_inherited')
        prefix += c['token_ids']
    stopped = bool(answer['calls']) and (eos in answer['calls'][-1]['token_ids'] or answer['calls'][-1]['finish_reason'] in {'eos','eos_token','stop'})
    ck(answer['truncated'] == (bool(answer['calls']) and not stopped and len(answer['answer_ids']) == cfg.answer_tokens), 'truncation_flag')
    ck(row['total_generation_tokens'] == row['search_work']+len(answer['answer_ids'])+row.get('auxiliary_work',0)
       == row['cost']['generated_tokens'], 'complete_native_cost')
    ck(all(row['cost'].get(k,0) == 0 for k in ('scored_tokens','cancelled_requests','preemptions')), 'no_hidden_scoring')
    if auxiliary:
        auxiliary[2]()
