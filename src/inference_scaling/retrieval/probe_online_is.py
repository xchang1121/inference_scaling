"""Frozen small probes gate retrieval or rank evidence at causal IS prefixes."""
from dataclasses import dataclass,asdict
import math

from inference_scaling.retrieval.linear_probe import predict_probe,fingerprint
from inference_scaling.retrieval.probe_features import prefix_features,evidence_features
from inference_scaling.retrieval.smart_online_is import clean_question
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.retrieval.tuned_online_is import TunedOnlineLedger,overlaps


@dataclass(frozen=True)
class ProbePolicy:
    model: dict
    threshold: float = .5
    evidence_candidates: int = 2

    def __post_init__(self):
        if not math.isfinite(self.threshold) or self.evidence_candidates<1:raise ValueError('invalid probe policy')
        if self.model['sha256']!=fingerprint(self.model):raise ValueError('probe model hash mismatch')
        if self.model['target'] not in {'knowledge','value','risk'}:raise ValueError('unsupported learned target')


class ProbeOnlineLedger(TunedOnlineLedger):
    def __init__(self,*args,probe_policy,**kwargs):
        super().__init__(*args,**kwargs)
        if self.policy.trigger!='fixed' or self.policy.query not in {'concat','question'}:
            raise ValueError('probe controls fixed checkpoints and concat/question retrieval')
        self.probe_policy=probe_policy;self.probe_features={}

    def decision_signal(self,branch,text,physical):
        features=prefix_features(self.recent(branch,32));p=self.probe_policy
        self.probe_features[token_hash(branch)]=features
        value=predict_probe(p.model,features) if p.model['target']=='knowledge' else None
        return value is None or value>=p.threshold,{'probe_features':features,'prediction':value,'model_sha256':p.model['sha256']}

    def retrieve(self,branch,used):
        p=self.probe_policy
        if p.model['target']=='knowledge':return super().retrieve(branch,used)
        tail=self.backend.decode(branch[-self.policy.query_tail:],skip_special_tokens=True).strip()
        query=clean_question(self.question) if self.policy.query=='question' else self.question+'\n'+tail
        found,log=self.facts.retrieve(query,k=self.policy.candidates,exclude=frozenset(used))
        found=[h for h in found if not any(overlaps(h['id'],old,self.policy.overlap) for old in used)][:p.evidence_candidates]
        trials=[]
        for candidate in found:
            packet=self.format_packet(candidate)
            evidence=evidence_features(self.question,query,packet,lambda ids,skip:self.backend.decode(ids,skip_special_tokens=skip))
            value=predict_probe(p.model,self.probe_features[token_hash(branch)]+evidence)
            accepted=value<=p.threshold if p.model['target']=='risk' else value>=p.threshold
            trials.append({'id':candidate['id'],'packet':packet,'evidence_features':evidence,'prediction':value,'accepted':accepted})
        candidates=[t for t in trials if t['accepted']]
        chosen=min(candidates,key=lambda t:(t['prediction'] if p.model['target']=='risk' else -t['prediction'],t['id']))['id'] if candidates else None
        log={**log,'probe_ranking':{'model_sha256':p.model['sha256'],'trials':trials,'chosen_id':chosen}}
        return next((h for h in found if h['id']==chosen),None),log,query

    def auxiliary_result(self):
        return {'probe_policy':asdict(self.probe_policy),
            'adaptation':'trained frozen-feature local suffix utility surrogate; no LLM weights updated; full IS evaluation still required'}
