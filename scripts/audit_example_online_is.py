"""Replay the candidate-local answer and whole-example selection independently."""
import math

from inference_scaling.retrieval.example_online_is import ExamplePolicy
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.shared.rng import SeedStream


def prepare_example_audit(row,cfg,decode,eos,close,ck):
    p=ExamplePolicy(**row['example_policy'])
    calls,consumed,queries=row['auxiliary_calls'],[],{}
    seed=SeedStream(SeedStream(row['seed']).derive('question',row['question_id'])).derive('search')
    ck([c['aux_id'] for c in calls]==list(range(len(calls))),'example_unique_calls')
    ck(sum(len(c['token_ids']) for c in calls)==row['auxiliary_work']<=p.work_tokens,'example_total_budget')

    def resolve(d):
        s=d['signal'];start,stop=s['auxiliary_start'],s['auxiliary_stop']
        ck(start==len(consumed) and start<=stop<=len(calls),'example_causal_order')
        exhausted=sum(len(c['token_ids']) for c in calls[:start])+p.answer_tokens>p.work_tokens
        reason={'auxiliary_start':start,'auxiliary_stop':stop,'budget_exhausted':exhausted}
        if exhausted:
            ck(start==stop,'example_budget_skip')
            return False,reason
        ck(stop==start+1,'example_single_interim')
        call=calls[start];req=call['request'];branch=d['branch_ids']
        prior=sorted((e for e in row['events'] if e['position']<len(branch) and
                      branch[:e['position']]==e['branch_ids']),key=lambda e:e['position'])
        physical=list(row['root_ids']);previous=0
        for e in prior:
            physical+=branch[previous:e['position']]+e['injected_ids'];previous=e['position']
        physical+=branch[previous:]+[close]
        ck(call['prefix_ids']==req['prefix']==physical and call['prefix_sha256']==token_hash(physical),'example_exact_auxiliary_prefix')
        ck(call['branch_ids']==branch and call['branch_sha256']==token_hash(branch),'example_branch')
        ck(req['seed']==SeedStream(seed).derive('retrieval_aux',token_hash(physical),token_hash(branch),'thinkretrieve_answer',0),'example_rng')
        ck(call['purpose']=='thinkretrieve_answer' and call['batch_size']==1 and call['batch_index']==0,'example_purpose_layout')
        ck(req['max_new_tokens']==p.answer_tokens and req['uniform_offset']==0 and req['stop_sequences']==[[close]],'example_request_limits')
        ck(0<len(call['token_ids'])<=p.answer_tokens and len(physical)+p.answer_tokens<=cfg.context_tokens,'example_context')
        ck(not {eos,close}.intersection(call['token_ids'][:-1]),'example_terminal_last')
        ck(all(len(call[k])==len(call['token_ids']) and all(math.isfinite(v) for v in call[k]) for k in ('lp','sc','cn')),'example_statistics')
        if 'generated_tokens' in call['batch_cost']:
            ck(call['batch_cost']['generated_tokens']==len(call['token_ids']),'example_measured_cost')
        interim=decode([t for t in call['token_ids'] if t not in {eos,close}],True).strip()
        complete=call['finish_reason'] in {'eos','eos_token','stop'} or call['token_ids'][-1]==eos
        reason.update(aux_ids=[call['aux_id']],interim=interim,complete=complete)
        queries[d['branch_sha256']]='query: '+row['question']+'\nsolution: '+interim
        consumed.append(call['aux_id'])
        return bool(interim) and (complete or not p.require_complete_answer),reason

    def rank(d,ranked,used,default):
        r=d['retrieval'];packets=r['candidate_packets'];budget=row['policy']['packet_tokens']
        ck([x['id'] for x in packets]==ranked,'example_ranked_packets')
        eligible=[x['id'] for x in packets if len(x['ids'])<=budget]
        ck(r['complete_packet_ids']==eligible and r['oversized_ids']==[x['id'] for x in packets if len(x['ids'])>budget],'example_intact_size_filter')
        for packet in packets:
            ck(token_hash(packet['ids'])==packet['sha256'],'example_source_packet_hash')
        chosen=eligible[0] if eligible else None
        if chosen is not None and d['event_id'] is not None:
            original=next(x for x in packets if x['id']==chosen)
            actual=row['events'][d['event_id']]['packet']
            ck(actual['ids'][:len(original['ids'])]==original['ids'],'whole_worked_solution_preserved')
            ck(actual['body_start']==original['body_start'] and actual['body_stop']==original['body_stop'],'whole_example_body_bounds')
            if not p.pad_to_budget:
                ck(actual['ids']==original['ids'],'unpadded_example_exact')
        return chosen

    def finish():ck(consumed==list(range(len(calls))),'all_example_auxiliary_charged')
    return resolve,lambda d:queries[d['branch_sha256']],finish,rank
