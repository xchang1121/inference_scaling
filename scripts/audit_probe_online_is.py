"""Replay learned gate and evidence predictions using only observed prefix rows."""
from types import SimpleNamespace,MethodType
import math
from audit_probe_pairs import evidence_features_match

from inference_scaling.retrieval.probe_online_is import ProbePolicy
from inference_scaling.retrieval.linear_probe import predict_probe
from inference_scaling.retrieval.probe_features import prefix_features,evidence_features
from inference_scaling.retrieval.tuned_online_is import TunedOnlineLedger,overlaps
from inference_scaling.retrieval.stepwise import token_hash


def prediction_matches(recorded, replayed, threshold, *, risk=False):
    """Allow at most two FP64 ULPs, and never a changed threshold decision.

    A native ARM64 numpy dot versus Windows replay differed by one ULP with
    identical inputs. Keep that trace; reject larger drift, nonfinite values,
    or even a one-ULP crossing of a gate. Features/artifact hashes remain exact.
    """
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (recorded, replayed, threshold)):
        return False
    if abs(recorded-replayed) > 2*max(math.ulp(recorded), math.ulp(replayed)):
        return False
    return ((recorded <= threshold) == (replayed <= threshold)) if risk else (
        (recorded >= threshold) == (replayed >= threshold))


def prepare_probe_audit(row,cfg,decode,eos,close,ck):
    p=ProbePolicy(**row['probe_policy']);features={}
    layer=p.model['feature_schema'].get('capture_layer')
    if layer is not None:
        from inference_scaling.retrieval.multilayer_capture import SCHEMA
        collection=row.get('signal_collection',{})
        ck(collection == {'requested':'multilayer','hidden_dimensions':513,'attention_topk':0,
            'hidden_source':'multilayer_diagnostic','projection_seed':2026111502,
            'state_offset':-1,'schema':SCHEMA}, 'probe_multilayer_collection_identity')
        for call in row['native_calls']:
            signals=call.get('internal_signals',[])
            ck(len(signals)==len(call['token_ids']), 'probe_multilayer_complete_native_signals')
            for signal in signals:
                v=signal.get('hidden_projection',[])
                ck(len(v)==513 and all(math.isfinite(x) for x in v) and v[512]==0,
                    'probe_multilayer_all_native_alignment')
    def resolve(d):
        n=d['observed_native_calls'];native=row['native_calls'][:n]
        proxy=SimpleNamespace(root=tuple(row['root_ids']),native_calls=native,
            tap=SimpleNamespace(history={c['request_id']:c['internal_signals'] for c in native}))
        proxy.recent=MethodType(TunedOnlineLedger.recent,proxy)
        pool=p.model['feature_schema']['pool']
        values=prefix_features(proxy.recent(d['branch_ids'],pool),pool=pool,capture_layer=layer);features[d['branch_sha256']]=values
        value=predict_probe(p.model,values) if p.model['target']=='knowledge' else None
        if value is not None:
            recorded=d['signal']['prediction']
            ck(prediction_matches(recorded,value,p.threshold),'probe_numeric_prediction_and_same_gate')
            value=recorded
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
            ck(prediction_matches(t['prediction'],value,p.threshold,risk=p.model['target']=='risk')
               and t['accepted']==keep,'probe_prediction_and_gate')
            if keep:accepted.append((value if p.model['target']=='risk' else -value,t['id']))
        chosen=min(accepted)[1] if accepted else None
        ck(trial['chosen_id']==chosen,'probe_chosen_evidence')
        if chosen is not None and d['event_id'] is not None:
            target=next(t['packet'] for t in trial['trials'] if t['id']==chosen)
            ck(row['events'][d['event_id']]['packet']==target,'probe_exact_committed_packet')
        return chosen
    return resolve,lambda d:None,lambda:None,rank
