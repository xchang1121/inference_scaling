"""Causal, budget, RNG and feature replay for paid candidate-local lookahead."""
from types import SimpleNamespace
import math

from inference_scaling.retrieval.auxiliary_online_is import AuxiliaryPolicy, auxiliary_feature
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.shared.rng import SeedStream
from inference_scaling.retrieval.auxiliary_signals import hidden_disagreement
from inference_scaling.retrieval.tuned_online_is import overlaps
from inference_scaling.retrieval.smart_online_is import BRIDGE, RESUME


def prepare_auxiliary_audit(row, cfg, decode, eos, close, ck):
    p = AuxiliaryPolicy(**row['auxiliary_policy'])
    calls, consumed = row['auxiliary_calls'], []
    ck([c['aux_id'] for c in calls] == list(range(len(calls))), 'auxiliary_unique_ids')
    ck(sum(len(c['token_ids']) for c in calls) == row['auxiliary_work'] <= p.work_tokens, 'auxiliary_total_budget')
    seed = SeedStream(SeedStream(row['seed']).derive('question',row['question_id'])).derive('search')
    backend = SimpleNamespace(decode=lambda ids,skip_special_tokens=True:decode(ids,skip_special_tokens))
    features = {}

    def physical_before(d):
        branch = d['branch_ids']
        events = sorted((e for e in row['events'] if e['position'] < len(branch)
                         and branch[:e['position']] == e['branch_ids']),key=lambda e:e['position'])
        physical, previous = list(row['root_ids']), 0
        for e in events:
            physical += branch[previous:e['position']]+e['injected_ids']
            previous = e['position']
        return physical+branch[previous:]

    def validate_group(group, branch, physical, purpose, tokens, count):
        ck(len(group) == count and [c['batch_index'] for c in group] == list(range(count)), 'auxiliary_full_sample_pool')
        for index,c in enumerate(group):
            req = c['request']
            ck(c['branch_ids'] == branch and c['branch_sha256'] == token_hash(branch), 'auxiliary_same_branch')
            ck(c['prefix_ids'] == req['prefix'] == physical and c['prefix_sha256'] == token_hash(physical), 'auxiliary_exact_physical_prefix')
            ck(req['seed'] == SeedStream(seed).derive('retrieval_aux',token_hash(physical),token_hash(branch),purpose,index), 'auxiliary_independent_rng')
            ck(c['purpose'] == purpose and c['batch_size'] == count, 'auxiliary_purpose_layout')
            ck(req['max_new_tokens'] == tokens and req['uniform_offset'] == 0, 'auxiliary_request_budget')
            ck(req['stop_sequences'] == [[close]] and req['statistic']['top_k'] == 5, 'auxiliary_stop_and_statistics')
            ck(0 < len(c['token_ids']) <= tokens and len(physical)+tokens <= cfg.context_tokens, 'auxiliary_length')
            ck(all(len(c[k]) == len(c['token_ids']) and all(math.isfinite(v) for v in c[k]) for k in ('lp','sc','cn')), 'auxiliary_finite_aligned_statistics')
            ck(not {eos,close}.intersection(c['token_ids'][:-1]), 'auxiliary_terminal_last')
            consumed.append(c['aux_id'])
        if group and 'generated_tokens' in group[0]['batch_cost']:
            ck(group[0]['batch_cost']['generated_tokens'] == sum(len(c['token_ids']) for c in group), 'auxiliary_measured_tokens')

    def resolve(d):
        original = d['signal']
        start, stop = original['auxiliary_start'], original['auxiliary_stop']
        ck(0 <= start <= stop <= len(calls), 'auxiliary_causal_cutoff')
        ck(start == len(consumed), 'auxiliary_decision_order')
        before = sum(len(c['token_ids']) for c in calls[:start])
        count = 1 if p.method == 'flare' else p.samples
        exhausted = before+count*p.draft_tokens > p.work_tokens
        reason = {'auxiliary_start':start,'auxiliary_stop':stop,'budget_exhausted':exhausted}
        if exhausted:
            ck(stop == start, 'auxiliary_exhausted_no_request')
            return False, reason
        group = calls[start:stop]
        branch = d['branch_ids']
        validate_group(group,branch,physical_before(d),p.method,p.draft_tokens,count)
        eligible, feature = auxiliary_feature(p, group, backend)
        reason.update(aux_ids=[c['aux_id'] for c in group],feature=feature)
        features[d['branch_sha256']] = feature
        return eligible, reason

    def query(d):
        if p.method == 'flare' and row['policy']['query'] != 'question':
            return row['question']+'\n'+features[d['branch_sha256']]['masked_query'].strip()
        return None

    def finish():
        ck(consumed == list(range(len(calls))), 'all_auxiliary_calls_charged_once')

    def rank(d, ranked, used, default):
        if not p.evidence_candidates:
            ck('evidence_trials' not in d['retrieval'], 'no_unrequested_evidence_trials')
            return default
        r = d['retrieval']['evidence_trials']
        expected = [k for k in ranked if not any(overlaps(k,old,row['policy']['overlap']) for old in used)][:p.evidence_candidates]
        ck(r['candidate_ids'] == expected,'trial_candidate_pool')
        start, stop = r['auxiliary_start'],r['auxiliary_stop']
        ck(start == len(consumed) and start <= stop <= len(calls),'trial_causal_order')
        before = sum(len(c['token_ids']) for c in calls[:start])
        exhausted = before+len(expected)*p.samples*p.evidence_tokens > p.work_tokens
        ck(r['budget_exhausted'] == exhausted,'trial_full_pool_reservation')
        ck([t['id'] for t in r['trials']] == ([] if exhausted else expected),'trial_complete_coverage')
        base = physical_before(d)
        accepted = []
        for t in r['trials']:
            offset = len(consumed)
            ids = list(range(offset,offset+p.samples))
            ck(t['aux_ids'] == ids,'trial_call_order')
            group = calls[offset:offset+p.samples]
            packet = t['packet']
            budget = row['policy']['packet_tokens']
            ck(packet['id'] == t['id'] and token_hash(packet['ids']) == packet['sha256'],'trial_packet_identity')
            ck(len(packet['ids']) == budget and t['injected_ids'][:budget] == packet['ids'],'trial_equal_packet_budget')
            continuation = {'none':'','resume':RESUME,'bridge':BRIDGE,'custom':row['policy']['continuation_text']}[row['policy']['continuation']]
            ck(decode(t['injected_ids'][budget:],False) == continuation,'trial_exact_continuation')
            ck(not {eos,close}.intersection(t['injected_ids']),'trial_no_structural_injection')
            validate_group(group,d['branch_ids'],base+t['injected_ids'],'seakr_evidence:'+t['id'],p.evidence_tokens,p.samples)
            feature = hidden_disagreement(group,tail=p.hidden_tail,metric=p.metric,ridge=p.ridge)
            ck(t['feature'] == feature,'trial_hidden_feature_replay')
            if feature['value'] <= p.evidence_ceiling:
                accepted.append((feature['value'],t['id']))
        chosen = min(accepted)[1] if accepted else None
        ck(r['chosen_id'] == chosen and stop == len(consumed),'trial_selection_and_budget')
        if d['event_id'] is not None and chosen is not None:
            event = row['events'][d['event_id']]
            chosen_trial = next(t for t in r['trials'] if t['id'] == chosen)
            ck(event['packet'] == chosen_trial['packet'] and event['injected_ids'] == chosen_trial['injected_ids'],'trial_exact_committed_evidence')
        return chosen
    return resolve, query, finish, rank
