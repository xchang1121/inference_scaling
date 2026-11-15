"""Experimental parent/window rank fusion without modifying the original bank."""
from dataclasses import dataclass
import math
import time

from inference_scaling.retrieval.controller import _clean_reference
from inference_scaling.retrieval.corpus import Document,Retriever
from inference_scaling.retrieval.literature import FactBank,encode,packet,FACT_HEADER,FACT_FOOTER
from inference_scaling.retrieval.smart_online_is import clean_question
from inference_scaling.retrieval.stepwise import token_hash


@dataclass(frozen=True)
class FactRanking:
    parent_weight: float = .5
    rank_constant: float = 20.
    max_windows_per_parent: int = 0
    clean_query: bool = False

    def __post_init__(self):
        if not 0<=self.parent_weight<=1 or not math.isfinite(self.rank_constant) or self.rank_constant<0:
            raise ValueError('invalid rank fusion')
        if self.max_windows_per_parent<0:raise ValueError('invalid per-parent window cap')


def rank_windows(parents,windows,policy,exclude):
    parent_ranks={name:i+1 for i,name in enumerate(parents)}
    counts={}
    for name in exclude:
        parent=name.rsplit('@',1)[0]
        counts[parent]=counts.get(parent,0)+1
    scores=[]
    for i,name in enumerate(windows,1):
        parent=name.rsplit('@',1)[0]
        if parent not in parent_ranks:raise ValueError('window parent missing')
        score=policy.parent_weight/(policy.rank_constant+parent_ranks[parent])+(1-policy.parent_weight)/(policy.rank_constant+i)
        scores.append((score,i,name,parent))
    ranked=[]
    for score,rank,name,parent in sorted(scores,key=lambda t:(-t[0],t[1],t[2])):
        if name in exclude or (policy.max_windows_per_parent and counts.get(parent,0)>=policy.max_windows_per_parent):continue
        ranked.append(name);counts[parent]=counts.get(parent,0)+1
    return ranked,{name:score for score,_,name,_ in scores}


class RankedFactBank(FactBank):
    def __init__(self,*args,ranking,**kwargs):
        super().__init__(*args,**kwargs);self.ranking=ranking

    def retrieve(self,query,*,k=1,exclude=frozenset()):
        started=time.perf_counter();b,cfg=self.backend,self.cfg
        retrieval_query=clean_question(query) if self.ranking.clean_query else query
        parents=self.index.search(retrieval_query,k=cfg.parent_documents)
        windows=[]
        for hit in parents:
            doc=hit.document;ids=encode(b,_clean_reference(doc.text))
            for start in range(0,len(ids),cfg.window_stride):
                stop=min(start+cfg.window_tokens,len(ids))
                windows.append(Document(f'{doc.id}@{start}-{stop}',doc.title,b.decode(ids[start:stop],skip_special_tokens=False),doc.source))
                if stop==len(ids):break
        hits=Retriever(windows,kind='bm25').search(retrieval_query,k=len(windows)) if windows else []
        by_id={h.document.id:h for h in hits}
        parent_ids=[h.document.id for h in parents];window_ids=[h.document.id for h in hits]
        ranked,scores=rank_windows(parent_ids,window_ids,self.ranking,exclude)
        result=[];overhead=len(encode(b,FACT_HEADER))+len(encode(b,FACT_FOOTER))
        for name in ranked[:k]:
            hit=by_id[name];d=hit.document
            body=encode(b,_clean_reference(d.title)+'\n'+_clean_reference(d.text))[:cfg.fact_tokens-overhead]
            p=packet(b,FACT_HEADER,'',FACT_FOOTER,id=d.id,score=scores[name],source=d.source)
            head,tail=p['ids'][:p['body_start']],p['ids'][p['body_stop']:]
            p.update(ids=head+list(body)+tail,body_stop=len(head)+len(body));p['sha256']=token_hash(p['ids'])
            result.append(p)
        return result,{'query':query,'retrieval_query':retrieval_query,'parent_ids':parent_ids,
            'window_ranked_ids':window_ids,'fusion_scores':scores,'selected_ids':[p['id'] for p in result],
            'seconds':time.perf_counter()-started}
