"""Audit shared prefix, causal features, paired RNG and complete local suffix cost."""
from types import SimpleNamespace,MethodType
import math

from inference_scaling.retrieval.probe_features import FEATURE_SCHEMA,prefix_features,EVIDENCE_SCHEMA,evidence_features
from inference_scaling.retrieval.tuned_online_is import TunedOnlineLedger
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.shared.rng import SeedStream


def audit_probe_pair(baseline,pairs,cfg,decode,eos,close,check):
    def ck(ok,label):check(ok,baseline['question_id']+':probe:'+label)
    ck({p['arm'] for p in pairs}=={'none','evidence'} and len(pairs)==2,'complete_intervention_pair')
    decision=baseline['decisions'][0];branch=decision['branch_ids'];n=decision['observed_native_calls']
    proxy=SimpleNamespace(root=tuple(baseline['root_ids']),native_calls=baseline['native_calls'][:n],
        tap=SimpleNamespace(history={c['request_id']:c['internal_signals'] for c in baseline['native_calls'][:n]}))
    proxy.recent=MethodType(TunedOnlineLedger.recent,proxy)
    features=prefix_features(proxy.recent(branch,32))
    unit_seed=SeedStream(baseline['seed']).derive('question',baseline['question_id'])
    paired_seed=SeedStream(unit_seed).derive('probe_pair',token_hash(branch))
    for r in pairs:
        ck(r['question_id']==baseline['question_id'] and r['seed']==baseline['seed'],'same_question_seed')
        ck(r['decision']==decision and r['branch_ids']==branch and r['base_ids']==baseline['root_ids'],'first_observed_prefix')
        ck(r['features']==features and r['feature_schema']==FEATURE_SCHEMA,'causal_hidden_features')
        ck(r['evidence_schema']==EVIDENCE_SCHEMA and r['evidence_features']==evidence_features(
            baseline['question'],r['retrieval']['query'],r['packet'],decode),'causal_evidence_features')
        ck(r['pair_seed']==paired_seed,'same_pair_seed')
        ck(r['packet']==pairs[0]['packet'] and r['retrieval']==pairs[0]['retrieval'],'same_retrieved_candidate')
        injected=[] if r['arm']=='none' else r['packet']['ids']
        ck(r['injected_ids']==injected and not {eos,close}.intersection(injected),'exact_intervention')
        ck(token_hash(r['packet']['ids'])==r['packet']['sha256'],'packet_hash')
        physical=r['base_ids']+branch+injected
        ck(len(r['auxiliary_calls'])==1,'single_local_suffix')
        c=r['auxiliary_calls'][0];req=c['request'];answer=r['answer']
        ck(c['prefix_ids']==req['prefix']==physical,'prefix_plus_intervention_only')
        ck(req['seed']==paired_seed and req['uniform_offset']==0 and req['max_new_tokens']==cfg.thinking_tokens-len(branch),'paired_suffix_rng_budget')
        ck(req['stop_sequences']==[[close]] and req['statistic']['top_k']==5,'native_stops_statistics')
        ck(0<len(c['token_ids'])<=req['max_new_tokens'] and not {eos,close}.intersection(c['token_ids'][:-1]),'suffix_length_and_terminal')
        ck(all(len(c[k])==len(c['token_ids']) and all(math.isfinite(v) for v in c[k]) for k in ('lp','sc','cn')),'suffix_statistics')
        early=eos in c['token_ids'];natural=c['token_ids'][-1:]==[close]
        ck(answer['eos_inside_thinking']==early and answer['forced_thinking_end']==(not early and not natural),'unchanged_answer_transition')
        if early:ck(not answer['calls'] and not answer['answer_ids'],'no_eos_repair')
        prefix=physical+c['token_ids']+([] if natural else [close])
        produced=[];answer_seed=SeedStream(paired_seed).derive('paired_answer')
        for i,call in enumerate(answer['calls']):
            ck(call['prefix_sha256']==token_hash(prefix+produced),'answer_exact_context')
            ck(call['seed']==SeedStream(answer_seed).derive('answer',i),'paired_answer_rng')
            ck(call['max_new_tokens']==min(cfg.answer_chunk,cfg.answer_tokens-len(produced)),'answer_chunk_budget')
            produced+=call['token_ids']
        ck(produced==answer['answer_ids'] and len(produced)<=cfg.answer_tokens,'answer_complete_tokens')
        ck(decode(produced,True).strip()==answer['answer_text'],'answer_decode')
        ck(answer['empty_answer']==(not bool(answer['answer_text'])),'empty_label')
        ck(r['total_generation_tokens']==len(c['token_ids'])+len(produced)==r['cost']['generated_tokens'],'complete_pair_cost')
