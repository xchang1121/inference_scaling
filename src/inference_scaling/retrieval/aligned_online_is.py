"""Candidate-local learned hidden retrieval, with optional lexical rank fusion."""
from dataclasses import asdict,dataclass
from types import SimpleNamespace

from inference_scaling.retrieval.corpus import Hit
from inference_scaling.retrieval.hidden_alignment import project_hidden, validate_alignment_model
from inference_scaling.retrieval.probe_features import prefix_features
from inference_scaling.retrieval.smart_online_is import clean_question
from inference_scaling.retrieval.tuned_online_is import TunedOnlineLedger,overlaps
from inference_scaling.retrieval.stepwise import token_hash


@dataclass(frozen=True)
class AlignmentPolicy:
    model: dict
    dense_weight: float = .5
    candidate_documents: int = 64

    def __post_init__(self):
        if not 0<=self.dense_weight<=1 or self.candidate_documents<1:
            raise ValueError('invalid hidden retrieval fusion')
        validate_alignment_model(self.model)


def fuse_documents(dense,lexical,weight):
    scores={}
    for coefficient,names in ((weight,dense),(1-weight,lexical)):
        if not coefficient:continue
        for rank,name in enumerate(names,1):scores[name]=scores.get(name,0)+coefficient/(20+rank)
    return sorted(scores,key=lambda name:(-scores[name],name)),scores


class AlignedOnlineLedger(TunedOnlineLedger):
    def __init__(self,*args,alignment_policy,document_vectors,**kwargs):
        super().__init__(*args,**kwargs)
        if self.policy.trigger!='fixed' or self.policy.query not in {'concat','question'}:
            raise ValueError('hidden retrieval uses fixed checkpoints and concat/question window queries')
        if alignment_policy.model['provenance'].get('vector_sha256')!=document_vectors.sha256:
            raise ValueError('alignment trained against a different document index')
        self.alignment_policy=alignment_policy;self.document_vectors=document_vectors;self.hidden_queries={}
        self.documents_by_id={d.id:d for d in self.facts.index.documents}
        if set(self.documents_by_id)!=set(document_vectors.ids):raise ValueError('corpus/index document set mismatch')

    def decision_signal(self,branch,text,physical):
        p=self.alignment_policy
        schema=p.model['feature_schema'];pool=schema['pool']
        features=prefix_features(self.recent(branch,pool),pool=pool,capture_layer=schema.get('capture_layer'))
        vector=project_hidden(p.model,features[:128])
        self.hidden_queries[token_hash(branch)]=vector
        return True,{'probe_features':features,'projected_query':vector.tolist(),'model_sha256':p.model['sha256']}

    def retrieve(self,branch,used):
        p=self.alignment_policy
        tail=self.backend.decode(branch[-self.policy.query_tail:],skip_special_tokens=True).strip()
        query=clean_question(self.question) if self.policy.query=='question' else self.question+'\n'+tail
        dense=self.document_vectors.search(self.hidden_queries[token_hash(branch)],k=p.candidate_documents)
        original=self.facts.index;lexical=original.search(query,k=p.candidate_documents)
        order,scores=fuse_documents([name for name,_ in dense],[h.document.id for h in lexical],p.dense_weight)
        def search(query,*,k):return [Hit(self.documents_by_id[name],scores[name]) for name in order[:k]]
        self.facts.index=SimpleNamespace(search=search)
        try:found,log=self.facts.retrieve(query,k=self.policy.candidates,exclude=frozenset(used))
        finally:self.facts.index=original
        log['alignment']={'vector_sha256':self.document_vectors.sha256,'dense_hits':dense,
            'lexical_hits':[(h.document.id,h.score) for h in lexical],'fusion_scores':scores,'parent_order':order}
        found=[h for h in found if not any(overlaps(h['id'],old,self.policy.overlap) for old in used)]
        return (found[0] if found else None),log,query

    def auxiliary_result(self):
        schema=self.alignment_policy.model['feature_schema']
        adaptation='supervised final hidden128-to-E5768 ridge; full document heads truncated512tokens; window matching remains lexical'
        if schema.get('capture_layer') is not None or schema['pool'] != 32:
            adaptation=('supervised explicit hidden128-to-E5768 ridge; layer='+str(schema.get('capture_layer','final'))
                        +'; causal pool='+str(schema['pool'])+'; full document heads truncated512tokens; window matching remains lexical')
        return {'alignment_policy':asdict(self.alignment_policy),
            'document_vector_sha256':self.document_vectors.sha256,
            'adaptation':adaptation}
