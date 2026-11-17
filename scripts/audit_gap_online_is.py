"""Offline replay of candidate-local gap planning and supported evidence choice."""
from types import SimpleNamespace
import math

from inference_scaling.retrieval.gap_online_is import GapPolicy, PLAN_SYSTEM, SUPPORT_SYSTEM, checker_text
from inference_scaling.retrieval.auxiliary_signals import draft_confidence
from inference_scaling.retrieval.common import clean_question
from inference_scaling.retrieval.common import parse_plan,parse_support,token_hash
from inference_scaling.retrieval.online_is import overlaps
from inference_scaling.shared.rng import SeedStream


def prepare_gap_audit(row,cfg,decode,eos,close,ck):
    p=GapPolicy(**row['gap_policy'])
    calls,consumed,plans=row['auxiliary_calls'],[],{}
    ck([c['aux_id'] for c in calls]==list(range(len(calls))),'gap_unique_ids')
    ck(sum(len(c['token_ids']) for c in calls)==row['auxiliary_work']<=p.work_tokens,'gap_total_budget')
    seed=SeedStream(SeedStream(row['seed']).derive('question',row['question_id'])).derive('search')
    backend=SimpleNamespace(decode=lambda ids,skip_special_tokens=True:decode(ids,skip_special_tokens))

    def previous(d):
        branch=d['branch_ids']
        return sorted((e for e in row['events'] if e['position']<len(branch)
                       and branch[:e['position']]==e['branch_ids']),key=lambda e:e['position'])

    def call(index,d,purpose,tokens,prefix=None,prompt=None):
        ck(index==len(consumed),'gap_call_order')
        c=calls[index];req=c['request'];branch=d['branch_ids']
        ck(c['branch_ids']==branch and c['branch_sha256']==token_hash(branch),'gap_branch')
        ck(c['prefix_ids']==req['prefix'] and c['prefix_sha256']==token_hash(c['prefix_ids']),'gap_prefix_identity')
        if prefix is not None:ck(c['prefix_ids']==prefix,'gap_exact_physical_prefix')
        if prompt is not None:ck(decode(c['prefix_ids'],False)==prompt,'gap_exact_checker_prompt')
        ck(c['purpose']==purpose and c['batch_size']==1 and c['batch_index']==0,'gap_purpose')
        ck(req['seed']==SeedStream(seed).derive('retrieval_aux',token_hash(c['prefix_ids']),token_hash(branch),purpose,0),'gap_rng')
        ck(req['max_new_tokens']==tokens and req['uniform_offset']==0,'gap_request_budget')
        ck(req['stop_sequences']==[[close]] and req['statistic']['top_k']==5,'gap_stops_statistics')
        ck(0<len(c['token_ids'])<=tokens and len(c['prefix_ids'])+tokens<=cfg.context_tokens,'gap_length')
        ck(all(len(c[k])==len(c['token_ids']) and all(math.isfinite(v) for v in c[k]) for k in ('lp','sc','cn')),'gap_aligned_statistics')
        ck(not {eos,close}.intersection(c['token_ids'][:-1]),'gap_terminal_last')
        if 'generated_tokens' in c['batch_cost']:ck(c['batch_cost']['generated_tokens']==len(c['token_ids']),'gap_measured_cost')
        consumed.append(index)
        return c

    def resolve(d):
        original=d['signal'];start=original['auxiliary_start']
        ck(start==len(consumed),'gap_decision_order')
        remaining=p.work_tokens-sum(len(c['token_ids']) for c in calls[:start])
        needed=p.draft_tokens+p.planner_tokens+(p.support_tokens if p.require_support else 0)
        reason={'auxiliary_start':start,'budget_exhausted':remaining<needed}
        if remaining<needed:
            reason['auxiliary_stop']=start
            return False,reason
        events=previous(d);branch=d['branch_ids']
        physical=list(row['root_ids']);before=0
        for e in events:
            physical+=branch[before:e['position']]+e['injected_ids'];before=e['position']
        physical+=branch[before:]
        draft=call(original['draft_aux_id'],d,'gap_draft',p.draft_tokens,prefix=physical)
        confidence=draft_confidence(draft,backend,low_probability=p.low_probability)
        reason.update(draft_aux_id=draft['aux_id'],draft_confidence=confidence)
        if confidence['low_fraction']<p.minimum_low_fraction:
            reason.update(auxiliary_stop=len(consumed),low_confidence_gate=False)
            return False,reason
        payload={'question':row['question'],'confirmed':decode(branch[-p.prefix_tail:],True),
                 'provisional_step':decode([t for t in draft['token_ids'] if t not in {eos,close}],True),
                 'references':[decode(e['packet']['ids'][e['packet']['body_start']:e['packet']['body_stop']],True) for e in events]}
        planner=call(original['planner_aux_id'],d,'gap_plan',p.planner_tokens,prompt=checker_text(PLAN_SYSTEM,payload))
        raw=decode(planner['token_ids'],True);plan,error=parse_plan(raw)
        reason.update(low_confidence_gate=True,planner_aux_id=planner['aux_id'],planner_payload=payload,
                      raw=raw,plan=plan,error=error,auxiliary_stop=len(consumed))
        plans[d['branch_sha256']]=plan
        return bool(plan and plan['need']),reason

    def query(d):return clean_question(row['question'])+'\n'+plans[d['branch_sha256']]['query']

    def rank(d,ranked,used,default):
        r=d['retrieval']['support_check'];plan=plans[d['branch_sha256']]
        expected=[k for k in ranked if not any(overlaps(k,old,row['policy']['overlap']) for old in used)][:p.support_candidates]
        packets=r['candidate_packets']
        ck([packet['id'] for packet in packets]==expected,'support_exact_candidate_pool')
        ck(r['auxiliary_start']==len(consumed),'support_call_order')
        for packet in packets:
            ck(token_hash(packet['ids'])==packet['sha256'] and len(packet['ids'])==row['policy']['packet_tokens'],'support_packet_integrity')
        bodies=[decode(packet['ids'][packet['body_start']:packet['body_stop']],True) for packet in packets]
        ck(r['bodies']==bodies,'support_exact_exposed_bodies')
        chosen=0 if packets else None
        if packets and p.require_support:
            payload={'question':row['question'],'missing_fact':plan['fact'],
                     'sources':[{'index':i,'body':body} for i,body in enumerate(bodies)]}
            c=call(r['aux_id'],d,'gap_support',p.support_tokens,prompt=checker_text(SUPPORT_SYSTEM,payload))
            raw=decode(c['token_ids'],True)
            support,error=parse_support(raw,{'bodies':[{'text':body} for body in bodies]})
            ck(r['payload']==payload and r['raw']==raw and r['support']==support and r['error']==error,'support_parse_and_verbatim_quote')
            chosen=support['index'] if support and support['use'] else None
        ck(r['auxiliary_stop']==len(consumed),'support_complete_cost')
        name=packets[chosen]['id'] if chosen is not None else None
        ck(r['chosen_id']==name,'support_chosen_evidence')
        if name is not None and d['event_id'] is not None:
            ck(row['events'][d['event_id']]['packet']==packets[chosen],'support_exact_committed_packet')
        return name

    def finish():ck(consumed==list(range(len(calls))),'gap_all_calls_charged_once')
    return resolve,query,finish,rank
