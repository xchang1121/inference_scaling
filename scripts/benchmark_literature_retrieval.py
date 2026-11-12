"""Same-question IRCoT/ThinkRetrieve study with frozen Block-Metropolized SC."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

from benchmark_flare_gap import clean, prepare
from benchmark_is_gpu import GraphMeasurements, engine_settings, reset_cache
from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.backends.loader import close_backend, load_backend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.output import output_settings_from_config, thinking_format_from_backend
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.retrieval.controller import _snapshot
from inference_scaling.retrieval.corpus import Retriever
from inference_scaling.retrieval.cot_guidance import (
    DualStatistics, GuidanceConfig, GuidanceLedger, checkpoint, finish_answer, search,
)
from inference_scaling.retrieval.literature import ExampleBank, FactBank, LiteratureController
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.shared.rng import SeedStream

ARMS=('direct','closed_is','static_facts','static_facts_noise','ircot','ircot_noise',
      'static_examples','static_examples_noise','thinkretrieve','thinkretrieve_noise','rethink')


def empty_handoff(base):
    return {'root_ids':tuple(base),'carried':{'sc':[],'cn':[],'lp':[]},'controller_calls':[],
        'controller_work':0,'events':[],'packets':[],'injected_ids':[],'terminal_eos':False,
        'retained_reason_ids':[],'interim_calls':[]}


def run_arm(raw,tap,controller,case,run_seed,arm,cfg,reference=None):
    base=tuple(case['prompt_ids']);close=controller.close
    seed=SeedStream(run_seed).derive('question',case['id'])
    stages={}
    before=time.perf_counter()
    if arm.startswith('ircot'):
        handoff=controller.ircot(base,case['question'],seed,noise=arm.endswith('_noise'))
    elif arm.startswith('static_'):
        handoff=controller.static(base,case['question'],examples='examples' in arm,noise=arm.endswith('_noise'))
    elif arm in {'thinkretrieve','thinkretrieve_noise','rethink'}:
        handoff=controller.thinkretrieve(base,case['question'],seed,
            mode='rethink' if arm=='rethink' else 'unrelated' if arm.endswith('_noise') else 'relevant',reference=reference)
    else:handoff=empty_handoff(base)
    stages['controller_seconds']=time.perf_counter()-before
    root=tuple(handoff['root_ids']);carried=handoff['carried'];aux_work=handoff['controller_work']
    probe=None
    before=time.perf_counter()
    if arm=='direct':
        request=GenerationRequest(base,cfg.thinking_tokens,SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id),
            SeedStream(seed).derive('direct'),f'{seed}:direct',stop_sequences=(close,),intermediate_events=False)
        sample,=raw.sample_batch([request])
        candidates=[tuple(sample.token_ids)]
        calls=[{'prefix_ids':list(base),'prefix_sha256':token_hash(base),'seed':request.seed,
            'max_new_tokens':request.max_new_tokens,'token_ids':list(sample.token_ids),'finish_reason':sample.finish_reason}]
        work=len(sample.token_ids);audit={'selection':{'direct':0},'method':'direct'}
    elif handoff['terminal_eos']:
        candidates=[(raw.tokenizer.eos_token_id,)]
        calls=[];work=0;audit={'selection':{arm:0},'terminal_controller_eos':True}
    else:
        if arm not in {'thinkretrieve','thinkretrieve_noise','rethink'}:
            probe=checkpoint(raw,tap,root,close,SeedStream(seed).derive('probe'),cfg)
            aux_work+=probe['work'];carried=probe['statistics']
            draft=tuple(probe['reason_ids'])
            if probe['terminal'] and probe['boundary'] is None:
                candidates=[draft];calls=[];work=0
                audit={'selection':{arm:0},'terminal_checkpoint':True}
            else:
                root+=draft
        if probe is None or not (probe['terminal'] and probe['boundary'] is None):
            ledger=GuidanceLedger(raw,tap,root,close,SeedStream(seed).derive('search'),cfg,
                carried=carried,cap=cfg.search_tokens-aux_work)
            candidates,audit=search(ledger,'is_block_sc')
            calls,work=ledger.calls,ledger.work
    stages['search_seconds']=time.perf_counter()-before
    assert len(candidates)==1
    assert work+aux_work<=cfg.search_tokens
    before=time.perf_counter()
    answer=finish_answer(raw,root,candidates[0],close,cfg,
        SeedStream(seed).derive('answer_suffix',token_hash(candidates[0])))
    stages['answer_seconds']=time.perf_counter()-before
    return {'arm':arm,'question_id':case['id'],'seed':run_seed,'base_ids':list(base),
        'handoff':handoff,'probe':probe,'root_ids':list(root),'carried':carried,
        'candidates':candidates,'calls':calls,'search_work':work,'presearch_work':aux_work,
        'total_generation_tokens':aux_work+work+len(answer['answer_ids']),
        'answer':answer,'audit':audit,'stage_seconds':stages}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('settings','prepared','corpus','questions','noise','output','examples','encoder'):
        p.add_argument('--'+name,type=Path,required=name in {'settings','prepared','corpus'})
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--case-ids',nargs='+')
    p.add_argument('--seeds',nargs='+',type=int,default=[20261112,20261113])
    p.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    p.add_argument('--worker',default='unassigned')
    a=p.parse_args();settings=load_settings(a.settings);cfg=GuidanceConfig()
    if a.prepare_only:
        if a.prepared.exists():raise ValueError('prepared data exists')
        prepare(a,settings);return
    if a.output is None or a.output.exists():raise ValueError('fresh output required')
    if a.examples is None or a.encoder is None:raise ValueError('example bank and encoder required')
    prepared=json.loads(a.prepared.read_text(encoding='utf-8'))
    cases=[c for c in prepared['cases'] if not a.case_ids or c['id'] in a.case_ids]
    if not cases or a.case_ids and {c['id'] for c in cases}!=set(a.case_ids):raise ValueError('invalid case selection')
    if 'thinkretrieve_noise' in a.arms and 'thinkretrieve' not in a.arms:raise ValueError('paired reference arm required')
    import fcntl
    with a.output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with a.output.open('w',encoding='utf-8') as stream:
            def write(row):
                stream.write(json.dumps(clean(row),ensure_ascii=False,allow_nan=False)+'\n')
                stream.flush();os.fsync(stream.fileno())
            source=Path(__file__).resolve().parents[1]
            hashes={f.relative_to(source).as_posix():hashlib.sha256(f.read_bytes()).hexdigest()
                for f in [*source.joinpath('src').rglob('*.py'),Path(__file__)]}
            write({'kind':'metadata','schema':1,'config':asdict(cfg),'worker':a.worker,
                'source':hashes,'cases':[c['id'] for c in cases],'seeds':a.seeds,'arms':a.arms,
                'prepared_sha256':hashlib.sha256(a.prepared.read_bytes()).hexdigest(),
                'settings_sha256':hashlib.sha256(a.settings.read_bytes()).hexdigest(),
                'examples_sha256':hashlib.sha256(a.examples.read_bytes()).hexdigest()})
            raw=graph=tap=None
            try:
                raw=load_backend(settings['ar']['model'],settings['ar']['engine'],seed=a.seeds[0],logprobs=6)
                graph,tap=GraphMeasurements(raw),DualStatistics(raw)
                fmt,=thinking_format_from_backend(raw,output_settings_from_config(settings['ar'])).formats
                close=tuple(fmt.end_token_ids)
                started=time.perf_counter()
                index=Retriever.from_jsonl(a.corpus,kind='bm25')
                assert index.fingerprint==prepared['corpus_sha256']
                fact_build_seconds=time.perf_counter()-started
                examples=ExampleBank(raw,a.examples,a.encoder)
                controller=LiteratureController(raw,tap,FactBank(raw,index),examples,close,prepared['noise_ids'],cfg)
                write({'kind':'engine','model_id':raw.model_id,'config':engine_settings(raw),
                    'retrieval':controller.metadata(),'fact_build_seconds':fact_build_seconds})
                # Graph/CPU warm-up is outside all per-question measurements.
                warm=GenerationRequest(tuple(cases[0]['prompt_ids']),64,SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id),
                    20261112,'literature:warmup',stop_sequences=(close,),intermediate_events=False)
                raw.sample_batch([warm]);examples.retrieve('How do I restart a server?')
                write({'kind':'warmup','complete':True})
                completed=0
                for qi,case in enumerate(cases):
                    for si,run_seed in enumerate(a.seeds):
                        offset=(qi+si)%len(a.arms)
                        arms=list(a.arms[offset:])+list(a.arms[:offset])
                        if 'thinkretrieve_noise' in arms:
                            arms.remove('thinkretrieve_noise');arms.insert(arms.index('thinkretrieve')+1,'thinkretrieve_noise')
                        reference=None
                        for arm in arms:
                            uid=f"{case['id']}:{run_seed}:{arm}"
                            print(json.dumps({'unit':uid,'completed':completed}),flush=True)
                            reset_cache(raw)
                            b0,g0=_snapshot(raw),graph.snapshot()
                            started=time.perf_counter()
                            result=run_arm(raw,tap,controller,case,run_seed,arm,cfg,reference)
                            elapsed=time.perf_counter()-started
                            b1=_snapshot(raw)
                            write({'kind':'measurement','unit_id':uid,'worker':a.worker,'seconds':elapsed,
                                'cost':{k:b1[k]-b0[k] for k in b0.keys()&b1.keys()
                                    if type(b0[k]) in (int,float) and type(b1[k]) in (int,float)},
                                **graph.difference(g0),**result})
                            if arm=='thinkretrieve':reference=result['handoff']['packets']
                            if result['answer']['truncated']:
                                raise RuntimeError('final answer truncated: stop and preserve the failed study')
                            if tap.rows:raise RuntimeError('unconsumed statistic rows')
                            completed+=1
                write({'kind':'completed','units':completed})
            except BaseException as error:
                write({'kind':'failed','error':repr(error)})
                raise
            finally:
                if tap is not None:tap.close()
                if graph is not None:graph.close()
                if raw is not None:close_backend(raw)


if __name__=='__main__':main()
