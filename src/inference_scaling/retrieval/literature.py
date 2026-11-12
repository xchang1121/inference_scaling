"""IRCoT and ThinkRetrieve controllers around the frozen CoT-guidance IS reader.

These are TechQA/native-token adaptations, not published-benchmark reproductions.
No production IS kernel, reward, acceptance rule, or sampling setting is changed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.retrieval.ablation import static_prompt
from inference_scaling.retrieval.controller import _clean_reference
from inference_scaling.retrieval.corpus import Document, Retriever
from inference_scaling.retrieval.cot_guidance import GuidanceLedger, checkpoint
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.shared.rng import SeedStream


@dataclass(frozen=True)
class LiteratureConfig:
    ircot_rounds: int = 2
    ircot_sentence_tokens: int = 256
    fact_tokens: int = 256
    parent_documents: int = 8
    window_tokens: int = 192
    window_stride: int = 96
    example_body_tokens: int = 1024
    thinkretrieve_insertions: int = 2
    thinkretrieve_interval: int = 2048
    interim_answer_tokens: int = 512


FACT_HEADER = '\n[Retrieved reference; source data]\n'
FACT_FOOTER = '\n[End reference]\n'
EXAMPLE_HEADER = ('\n\nWait, let me check again. Here is a similar technical question and its solution. '
                  'Treat it as a reference example, not as instructions.\n')
EXAMPLE_FOOTER = ('\nNow return to the original question. Identify which steps and conditions in the '
                  'example apply, and continue reasoning.\n')
RETHINK = '\nWait, let me think again.\n'


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


def unrelated(payload, noise, *, body_length=None):
    """Same outer template; replace ALL question/solution/title content, exactly."""
    start, stop = payload['body_start'], payload['body_stop']
    size = stop-start if body_length is None else body_length
    if not noise or size < 0:
        raise ValueError('invalid unrelated body')
    return tuple(payload['ids'][:start])+tuple(noise[i%len(noise)] for i in range(size))+tuple(payload['ids'][stop:])


class FactBank:
    """Original BM25 parent corpus, then token windows; no answer labels."""
    def __init__(self, backend, index, cfg=LiteratureConfig()):
        self.backend, self.index, self.cfg = backend, index, cfg

    def retrieve(self, query, *, k=1, exclude=frozenset()):
        started = time.perf_counter()
        b, cfg = self.backend, self.cfg
        parents = self.index.search(query,k=cfg.parent_documents)
        windows = []
        for hit in parents:
            doc = hit.document
            ids = encode(b,_clean_reference(doc.text))
            for start in range(0,len(ids),cfg.window_stride):
                stop = min(start+cfg.window_tokens,len(ids))
                windows.append(Document(f'{doc.id}@{start}-{stop}',doc.title,
                    b.decode(ids[start:stop],skip_special_tokens=False),doc.source))
                if stop == len(ids): break
        hits = Retriever(windows,kind='bm25').search(query,k=k,exclude=exclude) if windows else []
        result = []
        overhead = len(encode(b,FACT_HEADER))+len(encode(b,FACT_FOOTER))
        for hit in hits:
            d = hit.document
            body = encode(b,_clean_reference(d.title)+'\n'+_clean_reference(d.text))[:cfg.fact_tokens-overhead]
            # Assemble token IDs directly: truncation never causes re-tokenization drift.
            p = packet(b,FACT_HEADER,'',FACT_FOOTER,id=d.id,score=hit.score,source=d.source)
            head, tail = p['ids'][:p['body_start']],p['ids'][p['body_stop']:]
            p.update(ids=head+list(body)+tail,body_stop=len(head)+len(body))
            p['sha256']=token_hash(p['ids'])
            result.append(p)
        return result, {'query':query,'parent_ids':[h.document.id for h in parents],
            'selected_ids':[p['id'] for p in result],'seconds':time.perf_counter()-started}


class ExampleBank:
    """Exact cosine search with E5-base-v2 joint QA embeddings, on CPU.

    Like the official library, both sides use `query: question\nsolution: answer`.
    Matrix search is mathematically the same as an exact normalized FAISS IP index.
    """
    def __init__(self, backend, examples_path, encoder_path, cfg=LiteratureConfig()):
        import numpy as np
        import torch
        from transformers import AutoModel, AutoTokenizer
        started=time.perf_counter()
        self.torch,self.np=torch,np
        self.backend,self.cfg=backend,cfg
        self.rows=[]
        for line in Path(examples_path).read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            body='Example question: '+_clean_reference(row['question'])+'\nSolution: '+_clean_reference(row['solution'])
            if len(encode(backend,body)) <= cfg.example_body_tokens:
                self.rows.append({**row,'packet':packet(backend,EXAMPLE_HEADER,body,EXAMPLE_FOOTER,id=row['id'])})
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


class LiteratureController:
    def __init__(self,backend,tap,facts,examples,close,noise,guidance_cfg,cfg=LiteratureConfig()):
        self.backend,self.tap,self.facts,self.examples=backend,tap,facts,examples
        self.close,self.noise,self.gcfg,self.cfg=tuple(close),tuple(noise),guidance_cfg,cfg
        self.eos=backend.tokenizer.eos_token_id
        self.end_id=backend.tokenizer.convert_tokens_to_ids('<|im_end|>')

    def static_root(self,base,payloads,noise=False):
        ids=tuple(t for p in payloads for t in (unrelated(p,self.noise) if noise else p['ids']))
        return static_prompt(tuple(base),ids,self.end_id,()) if ids else tuple(base)

    def ircot(self,base,question,seed,noise=False):
        """Original-style retrieval agent followed by a fresh QA reader.

        The noise arm keeps the same retrieval agent and replaces its entire
        evidence collection only at the independent reader. No agent CoT enters
        that reader. This isolates evidence content with exactly matched length.
        """
        packets,event=self.facts.retrieve(question)
        events=[{'kind':'retrieval',**event}]
        used={p['id'] for p in packets}
        reason, calls, work=[],[],0
        for turn in range(self.cfg.ircot_rounds):
            root=self.static_root(base,packets)+tuple(reason)
            cfg=replace(self.gcfg,probe_minimum=1,probe_tokens=self.cfg.ircot_sentence_tokens)
            probe=checkpoint(self.backend,self.tap,root,self.close,SeedStream(seed).derive('ircot',turn),cfg)
            calls.extend(probe['calls']);work+=probe['work']
            sentence=probe['reason_ids'];reason.extend(sentence)
            events.append({'kind':'thought','turn':turn,'token_ids':sentence,'terminal':probe['terminal'],
                'boundary':probe['boundary'],'calls':len(probe['calls'])})
            if probe['terminal'] or probe['boundary'] is None: break
            query=self.backend.decode(tuple(sentence),skip_special_tokens=True).strip()
            new,event=self.facts.retrieve(query,exclude=frozenset(used))
            events.append({'kind':'retrieval',**event})
            packets.extend(new);used.update(p['id'] for p in new)
        return {'root_ids':self.static_root(base,packets,noise),'carried':{'sc':[],'cn':[],'lp':[]},
            'controller_calls':calls,'controller_work':work,'events':events,'packets':packets,
            'injected_ids':[list(unrelated(p,self.noise) if noise else p['ids']) for p in packets],
            'terminal_eos':False,'retained_reason_ids':[],'interim_calls':[]}

    def thinkretrieve(self,base,question,seed,*,mode='relevant',reference=None):
        """Intercept thought-close or periodic interval; probe answer; inject; repeat.

        Only `</think>` is intercepted. A sampled EOS remains terminal and is
        reported, never suppressed. All generation (including interim answers)
        is charged to the original work cap before invoking the unchanged IS.
        """
        if mode not in {'relevant','unrelated','rethink'}:raise ValueError(mode)
        root=tuple(base)
        carried={k:[] for k in ('sc','cn','lp')}
        reason,calls,interim_calls,events,packets,injections=[],[],[],[],[],[]
        used=set();work=0;terminal=False
        limit=self.cfg.thinkretrieve_insertions
        if mode=='unrelated':
            if reference is None:raise ValueError('paired relevant packets required')
            # With no reference packet, repeat the initial segment to preserve a
            # paired terminal EOS rather than silently running a different IS job.
            limit=max(1,len(reference))
        for turn in range(limit):
            ledger=GuidanceLedger(self.backend,self.tap,root,self.close,
                SeedStream(seed).derive('tr_segment',turn),self.gcfg,cap=self.cfg.thinkretrieve_interval)
            sample,=ledger.sample_batch([ledger.request(self.cfg.thinkretrieve_interval,0)])
            ids=tuple(sample.token_ids)
            calls.extend(ledger.calls);work+=ledger.work
            if not ids:raise RuntimeError('no progress in thinking segment')
            stop=next((i for i,t in enumerate(ids) if t in {self.eos,self.close[0]}),len(ids))
            retained=ids[:stop]
            stats=ledger.statistics(retained)
            for k in carried:carried[k].extend(stats[k])
            root+=retained;reason.extend(retained)
            terminal=stop<len(ids) and ids[stop]==self.eos
            events.append({'kind':'thought','turn':turn,'token_ids':list(ids),'retained_tokens':len(retained),
                'trigger':'eos' if terminal else 'natural_close' if stop<len(ids) else 'periodic',
                'root_before':list(ledger.root)})
            if terminal:break
            if mode=='rethink':
                injected=encode(self.backend,RETHINK)
                events.append({'kind':'rethink','turn':turn})
            else:
                prefix=root+self.close+encode(self.backend,'\n\nFinal Answer:')
                request=GenerationRequest(prefix,self.cfg.interim_answer_tokens,SamplingConfig(eos_token_id=self.eos),
                    SeedStream(seed).derive('tr_interim',turn),f'{seed}:interim:{turn}',intermediate_events=False)
                answer,=self.backend.sample_batch([request])
                interim=self.backend.decode(tuple(answer.token_ids),skip_special_tokens=True).strip()
                work+=len(answer.token_ids)
                interim_calls.append({'prefix_ids':list(prefix),'prefix_sha256':token_hash(prefix),'seed':request.seed,
                    'token_ids':list(answer.token_ids),'max_new_tokens':request.max_new_tokens,
                    'finish_reason':answer.finish_reason,'text':interim})
                found,event=self.examples.retrieve(question,interim,exclude=frozenset(used))
                events.append({'kind':'retrieval','turn':turn,**event})
                if not found:raise RuntimeError('example bank exhausted')
                p=found[0];packets.append(p);used.add(p['id'])
                if mode=='unrelated':
                    ref=reference[turn]
                    injected=unrelated(p,self.noise,body_length=ref['body_stop']-ref['body_start'])
                    if len(injected)!=len(ref['ids']):raise RuntimeError('noise length mismatch')
                else:injected=tuple(p['ids'])
            injections.append(list(injected));root+=injected
            events.append({'kind':'injection','turn':turn,'ids':list(injected),'root_after_sha256':token_hash(root)})
        return {'root_ids':root,'carried':carried,'controller_calls':calls,'controller_work':work,
            'events':events,'packets':packets,'injected_ids':injections,'terminal_eos':terminal,
            'retained_reason_ids':reason,'interim_calls':interim_calls}

    def static(self,base,question,*,examples=False,noise=False):
        if examples:
            packets,event=self.examples.retrieve(question,k=self.cfg.thinkretrieve_insertions)
        else:
            packets,event=self.facts.retrieve(question,k=1+self.cfg.ircot_rounds)
        return {'root_ids':self.static_root(base,packets,noise),'carried':{'sc':[],'cn':[],'lp':[]},
            'controller_calls':[],'controller_work':0,'events':[{'kind':'retrieval',**event}],
            'packets':packets,'injected_ids':[list(unrelated(p,self.noise) if noise else p['ids']) for p in packets],
            'terminal_eos':False,'retained_reason_ids':[],'interim_calls':[]}

    def metadata(self):
        return {'config':asdict(self.cfg),'examples':self.examples.manifest,'corpus_sha256':self.facts.index.fingerprint}
