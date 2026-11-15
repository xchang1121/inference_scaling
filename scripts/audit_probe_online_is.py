"""Replay learned gate and evidence predictions using only observed prefix rows."""
from types import SimpleNamespace,MethodType
from audit_probe_pairs import evidence_features_match

from inference_scaling.retrieval.probe_online_is import ProbePolicy
from inference_scaling.retrieval.linear_probe import predict_probe
from inference_scaling.retrieval.probe_features import prefix_features,evidence_features
from inference_scaling.retrieval.tuned_online_is import TunedOnlineLedger,overlaps
from inference_scaling.retrieval.stepwise import token_hash


def prepare_probe_audit(row,cfg,decode,eos,close,ck):
    p=ProbePolicy(**row['probe_policy']);features={}
    def resolve(d):
        n=d['observed_native_calls'];native=row['native_calls'][:n]
        proxy=SimpleNamespace(root=tuple(row['root_ids']),native_calls=native,
            tap=SimpleNamespace(history={c['request_id']:c['internal_signals'] for c in native}))
        proxy.recent=MethodType(TunedOnlineLedger.recent,proxy)
        values=prefix_features(proxy.recent(d['branch_ids'],32));features[d['branch_sha256']]=values
        value=predict_probe(p.model,values) if p.model['target']=='knowledge' else None
        return value is None or value>=p.threshold,{'probe_features':values,'prediction':value,'model_sha256':p.model['sha256']}

    def rank(d,ranked,used,default):
        if p.model['target']=='knowledge':
            ck('probe_ranking' not in d['retrieval'],'knowledge_gate_only');return default
        trial=d['retrieval']['probe_ranking'];expected=[name for name in ranked if not any(
            overlaps(name,old,row['policy']['overlap']) for old in used)][:p.evidence_candidates]
        ck([t['id'] for t in trial['trials']]==expected,'probe_complete_candidate_pool')
        ck(trial['model_sha256']==p.model['sha256'],'probe_frozen_artifact')
        accepted=[]
        for t in trial['trials']:
            packet=t['packet']
            ck(packet['id']==t['id'] and token_hash(packet['ids'])==packet['sha256'],'probe_candidate_identity')
            ck(len(packet['ids'])==row['policy']['packet_tokens'],'probe_actual_packet_budget')
            evidence=evidence_features(row['question'],d['retrieval']['query'],packet,decode)
            ck(evidence_features_match(t['evidence_features'],evidence),'probe_causal_evidence_features')
            # Replay the actual validated runtime inputs, preserving its libm rounding.
            value=predict_probe(p.model,features[d['branch_sha256']]+t['evidence_features'])
            keep=value<=p.threshold if p.model['target']=='risk' else value>=p.threshold
            ck(t['prediction']==value and t['accepted']==keep,'probe_prediction_and_gate')
            if keep:accepted.append((value if p.model['target']=='risk' else -value,t['id']))
        chosen=min(accepted)[1] if accepted else None
        ck(trial['chosen_id']==chosen,'probe_chosen_evidence')
        if chosen is not None and d['event_id'] is not None:
            target=next(t['packet'] for t in trial['trials'] if t['id']==chosen)
            ck(row['events'][d['event_id']]['packet']==target,'probe_exact_committed_packet')
        return chosen
    return resolve,lambda d:None,lambda:None,rank
