"""Frozen document and worked-example retrieval for candidate-local IS."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import time

from inference_scaling.retrieval.common import _clean_reference, clean_question, token_hash
from inference_scaling.retrieval.corpus import Document, Retriever

@dataclass(frozen=True)
class EvidenceConfig:
    fact_tokens: int = 256
    parent_documents: int = 8
    window_tokens: int = 192
    window_stride: int = 96
    example_body_tokens: int = 1024


FACT_HEADER = '\n[Retrieved reference; source data]\n'


FACT_FOOTER = '\n[End reference]\n'


# Official ThinkRetrieve ICE_TEMPLATE, copyright (c) 2026 Vaibhav Singh, MIT.
# License: docs/third_party/THINKRETRIEVE-LICENSE.txt.
EXAMPLE_HEADER = ('\n\nWait, let me check again. Here is a similar question and its solution. '
                  'Looking at this example:\n')


EXAMPLE_FOOTER = ('\n\nNow coming back to the original question, '
                  'Here are the key points I can take from example question to solve the '
                  'original question:\n')


def encode(backend, text):
    return tuple(backend.encode(text, add_special_tokens=False))


def packet(backend, header, body, footer, **metadata):
    head, content, tail = encode(backend,header), encode(backend,body), encode(backend,footer)
    ids = head+content+tail
    forbidden = {backend.tokenizer.eos_token_id, backend.tokenizer.convert_tokens_to_ids('<think>'),
                 backend.tokenizer.convert_tokens_to_ids('</think>')}
    if forbidden.intersection(ids):
        raise ValueError('structural token in retrieved data')
    return {'ids':list(ids),'body_start':len(head),'body_stop':len(head)+len(content),
            'sha256':token_hash(ids),**metadata}


def format_example(question,solution):
    return '## Example 1:\nProblem:\n'+_clean_reference(question)+'\n\nSolution:\n'+_clean_reference(solution)


def example_packet(backend,body,example_id):
    """Encode the official template once, recording token-aligned body bounds."""
    text=EXAMPLE_HEADER+body+EXAMPLE_FOOTER
    encoded=backend.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=list(encoded['input_ids']);offsets=encoded['offset_mapping']
    assert tuple(ids)==encode(backend,text)
    left,right=len(EXAMPLE_HEADER),len(EXAMPLE_HEADER)+len(body)
    inside=[i for i,(a,b) in enumerate(offsets) if left<=a<b<=right]
    if not inside:raise ValueError('no token-aligned example body')
    start,stop=inside[0],inside[-1]+1
    # BPE may merge final punctuation with the following newline. Retain only
    # such punctuation outside the body; keep every word/number within it.
    edges=text[left:offsets[start][0]]+text[offsets[stop-1][1]:right]
    if re.search(r'\w',edges):raise ValueError('unmasked lexical example boundary')
    forbidden={backend.tokenizer.eos_token_id,backend.tokenizer.convert_tokens_to_ids('<think>'),
               backend.tokenizer.convert_tokens_to_ids('</think>')}
    if forbidden.intersection(ids):raise ValueError('structural token in example')
    return {'ids':ids,'body_start':start,'body_stop':stop,'sha256':token_hash(ids),'id':example_id}


class ExampleBank:
    """Exact cosine search with E5-base-v2 joint QA embeddings, on CPU.

    Like the official library, both sides use `query: question\nsolution: answer`.
    Matrix search is mathematically the same as an exact normalized FAISS IP index.
    """
    def __init__(self, backend, examples_path, encoder_path, cfg=EvidenceConfig()):
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer
        started=time.perf_counter()
        self.torch,self.np=torch,np
        self.backend,self.cfg=backend,cfg
        self.rows=[]
        for line in Path(examples_path).read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            body=format_example(row['question'],row['solution'])
            if len(encode(backend,body)) <= cfg.example_body_tokens:
                self.rows.append({**row,'packet':example_packet(backend,body,row['id'])})
        if len(self.rows)<30: raise ValueError('too few complete worked examples')
        self.tokenizer=AutoTokenizer.from_pretrained(encoder_path,local_files_only=True)
        self.model=AutoModel.from_pretrained(encoder_path,local_files_only=True).to('cpu').eval()
        self.embeddings=self.embed([f"query: {r['question']}\nsolution: {r['solution']}" for r in self.rows])
        self.build_seconds=time.perf_counter()-started
        self.manifest={'entries':len(self.rows),'ids':[r['id'] for r in self.rows],
            'encoder':json.loads((Path(encoder_path)/'MANIFEST.json').read_text()),
            'build_seconds':self.build_seconds,'retrieval':'joint-QA E5-base-v2 exact cosine; CPU; max encoder length 512'}

    def embed(self, texts):
        torch=self.torch
        output=[]
        with torch.inference_mode():
            for offset in range(0,len(texts),8):
                batch=self.tokenizer(texts[offset:offset+8],max_length=512,padding=True,truncation=True,return_tensors='pt')
                hidden=self.model(**batch).last_hidden_state
                hidden=hidden.masked_fill(~batch['attention_mask'][...,None].bool(),0.)
                pooled=hidden.sum(1)/batch['attention_mask'].sum(1)[...,None]
                output.append(torch.nn.functional.normalize(pooled,p=2,dim=1).cpu().numpy())
        return self.np.concatenate(output,axis=0)

    def retrieve(self, question, interim='', *, k=1, exclude=frozenset()):
        started=time.perf_counter()
        query=f'query: {question}\nsolution: {interim}'
        vector=self.embed([query])[0]
        scores=self.embeddings@vector
        ranking=sorted(range(len(scores)),key=lambda i:(-float(scores[i]),self.rows[i]['id']))
        chosen=[i for i in ranking if self.rows[i]['id'] not in exclude][:k]
        packets=[{**self.rows[i]['packet'],'score':float(scores[i])} for i in chosen]
        return packets,{'query':query,'selected_ids':[p['id'] for p in packets],
            'top_scores':[p['score'] for p in packets],
            'encoder_query_tokens':len(self.tokenizer.encode(query)),
            'seconds':time.perf_counter()-started}


@dataclass(frozen=True)
class FactRanking:
    parent_weight: float = .5
    rank_constant: float = 20.
    max_windows_per_parent: int = 0
    clean_query: bool = False
    window_kind: str = 'bm25'

    def __post_init__(self):
        if not 0<=self.parent_weight<=1 or not math.isfinite(self.rank_constant) or self.rank_constant<0:
            raise ValueError('invalid rank fusion')
        if self.max_windows_per_parent<0:raise ValueError('invalid per-parent window cap')
        if self.window_kind not in {'bm25','char','hybrid'}:raise ValueError('invalid window ranker')


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


class FactBank:
    """Retrieve parent documents, rank their token windows and encode evidence."""
    def __init__(self, backend, index, cfg=EvidenceConfig(), *, ranking=None):
        self.backend, self.index, self.cfg, self.ranking = backend, index, cfg, ranking

    def retrieve(self, query, *, k=1, exclude=frozenset()):
        started = time.perf_counter()
        b, cfg, ranking = self.backend, self.cfg, self.ranking
        retrieval_query = clean_question(query) if ranking and ranking.clean_query else query
        parents = self.index.search(retrieval_query, k=cfg.parent_documents)
        windows = []
        for hit in parents:
            doc = hit.document
            ids = encode(b, _clean_reference(doc.text))
            for start in range(0, len(ids), cfg.window_stride):
                stop = min(start+cfg.window_tokens, len(ids))
                windows.append(Document(f'{doc.id}@{start}-{stop}', doc.title,
                    b.decode(ids[start:stop], skip_special_tokens=False), doc.source))
                if stop == len(ids):
                    break
        hits = Retriever(windows, kind=ranking.window_kind if ranking else 'bm25').search(
            retrieval_query, k=len(windows) if ranking else k,
            exclude=frozenset() if ranking else exclude) if windows else []
        parent_ids = [h.document.id for h in parents]
        metadata = {'query': query, 'parent_ids': parent_ids}
        if ranking:
            window_ids = [h.document.id for h in hits]
            ranked, scores = rank_windows(parent_ids, window_ids, ranking, exclude)
            by_id = {h.document.id: h for h in hits}
            selected = [(by_id[name].document, scores[name]) for name in ranked[:k]]
            metadata.update(retrieval_query=retrieval_query, window_ranker=ranking.window_kind,
                            window_ranked_ids=window_ids, fusion_scores=scores)
        else:
            selected = [(h.document, h.score) for h in hits]
        result = []
        overhead = len(encode(b, FACT_HEADER))+len(encode(b, FACT_FOOTER))
        for doc, score in selected:
            body = encode(b, _clean_reference(doc.title)+'\n'+_clean_reference(doc.text))[:cfg.fact_tokens-overhead]
            payload = packet(b, FACT_HEADER, '', FACT_FOOTER, id=doc.id, score=score, source=doc.source)
            head, tail = payload['ids'][:payload['body_start']], payload['ids'][payload['body_stop']:]
            payload.update(ids=head+list(body)+tail, body_stop=len(head)+len(body))
            payload['sha256'] = token_hash(payload['ids'])
            result.append(payload)
        return result, {**metadata, 'selected_ids': [p['id'] for p in result],
                        'seconds': time.perf_counter()-started}
