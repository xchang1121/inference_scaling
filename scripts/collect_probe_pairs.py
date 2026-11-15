"""Tiny TRAIN-only, same-prefix intervention pairs for local-utility labels.

Prefixes come from the first observed original IS checkpoint, before selection.
Paired suffixes use the base sampler, not a second full IS search. Their utility
is explicitly a local surrogate; actual learned policies need full-IS evaluation.
"""
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace,MethodType
import argparse
import hashlib
import json
import os
import time

from benchmark_flare_gap import clean,render
from benchmark_is_gpu import GraphMeasurements,engine_settings,reset_cache
from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.backends.loader import load_backend,close_backend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest,TokenStatistic
from inference_scaling.retrieval.ablation import SYSTEM
from inference_scaling.retrieval.controller import _snapshot
from inference_scaling.retrieval.corpus import Retriever
from inference_scaling.retrieval.cot_guidance import GuidanceConfig,DualStatistics,finish_answer
from inference_scaling.retrieval.literature import FactBank,LiteratureConfig
from inference_scaling.retrieval.internal_signals import InternalSignals,select_signal_worker
from inference_scaling.retrieval.native_journal import NativeJournal
from inference_scaling.retrieval.online_is import fixed_packet
from inference_scaling.retrieval.probe_features import prefix_features,FEATURE_SCHEMA,evidence_features,EVIDENCE_SCHEMA
from inference_scaling.retrieval.tuned_online_is import TuningTap,TuningPolicy,TunedOnlineLedger,run_tuned_is
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.shared.rng import SeedStream


def main():
    p=argparse.ArgumentParser()
    for name in ('settings','prepared','output','batch','corpus'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();batch=json.loads(a.batch.read_text());cfg=GuidanceConfig()
    assert len(batch['configs'])==1 and len(batch['seeds'])==1 and 1<=len(batch['cases'])<=4
    assert all(c['id'].startswith('TRAIN_') and c['cohort']=='probe_train' for c in batch['cases'])
    config=batch['configs'][0];position=config.get('position',128)
    assert position in (128,256,512) and config.get('packet_tokens',256) in (128,256,384,512)
    settings=load_settings(a.settings);select_signal_worker(hidden=True)
    settings['ar']['engine']['vllm']['engine_kwargs']['max_logprobs']=160
    raw=signals=dual=journal=graph=None
    with a.output.open('x',encoding='utf-8') as stream:
        def write(row):
            stream.write(json.dumps(clean(row),ensure_ascii=False,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        write({'kind':'metadata','batch':batch,'config':asdict(cfg),'probe_schema':FEATURE_SCHEMA,
            'label_scope':'paired base-model suffix utility at first observed IS checkpoint, not full-IS causal utility',
            'source':{str(f.relative_to(Path(__file__).resolve().parents[1])):hashlib.sha256(f.read_bytes()).hexdigest()
                      for f in Path(__file__).resolve().parents[1].joinpath('src').rglob('*.py')}})
        try:
            seed=batch['seeds'][0]
            raw=load_backend(settings['ar']['model'],settings['ar']['engine'],seed=seed,logprobs=6)
            signals=InternalSignals(raw,hidden_dimensions=128);dual=DualStatistics(raw);tap=TuningTap(dual,signals)
            graph=GraphMeasurements(raw);journal=NativeJournal(raw,dual,a.output.with_name(a.output.stem+'-native.jsonl'))
            close=(raw.tokenizer.convert_tokens_to_ids('</think>'),)
            facts=FactBank(raw,Retriever.from_jsonl(a.corpus,kind='bm25'),LiteratureConfig(**config.get('fact_config',{})))
            write({'kind':'engine','model_id':raw.model_id,'config':engine_settings(raw),'corpus_sha256':facts.index.fingerprint})
            warm=render(raw.tokenizer,settings['ar'],SYSTEM,batch['cases'][0]['question'],True)
            samples=raw.sample_batch([GenerationRequest(warm,64,SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id),seed,'probe:warmup',stop_sequences=(close,),intermediate_events=False)])
            write({'kind':'warmup','generated_tokens':len(samples[0].token_ids)})
            units=0
            for case in batch['cases']:
                root=tuple(render(raw.tokenizer,settings['ar'],SYSTEM,case['question'],True))
                unit_seed=SeedStream(seed).derive('question',case['id'])
                journal.unit=f'{case["id"]}:{seed}:prefix-baseline';reset_cache(raw)
                signals.enabled=True;tap.history.clear();before=_snapshot(raw);started=time.perf_counter()
                policy=TuningPolicy(trigger='never',positions=(position,))
                baseline=run_tuned_is(raw,tap,root,close,unit_seed,cfg,question=case['question'],facts=facts,policy=policy)
                after=_snapshot(raw)
                write({'kind':'measurement','question_id':case['id'],'seed':seed,'config_id':'prefix-baseline','arm':'tuned_online',
                    'seconds':time.perf_counter()-started,'base_ids':list(root),'static_packets':[],'static_retrievals':[],'static_injected_ids':[],
                    'cost':{k:after[k]-before[k] for k in before.keys()&after.keys() if type(before[k]) in (int,float)},**baseline})
                units+=1
                eligible=[d for d in baseline['decisions'] if d['position']==position]
                if not eligible:
                    write({'kind':'probe_skip','question_id':case['id'],'reason':'no native checkpoint before terminal'})
                    continue
                decision=eligible[0];branch=tuple(decision['branch_ids']);n=decision['observed_native_calls']
                proxy=SimpleNamespace(root=root,native_calls=baseline['native_calls'][:n],tap=tap)
                proxy.recent=MethodType(TunedOnlineLedger.recent,proxy)
                features=prefix_features(proxy.recent(branch,32))
                query=case['question']+'\n'+raw.decode(branch[-256:],skip_special_tokens=True).strip()
                found,retrieval=facts.retrieve(query,k=1)
                if not found:
                    write({'kind':'probe_skip','question_id':case['id'],'reason':'no retrieved candidate'});continue
                packet=fixed_packet(raw,found[0],config.get('packet_tokens',256))
                evidence=evidence_features(case['question'],query,packet,
                    lambda ids,skip:raw.decode(ids,skip_special_tokens=skip))
                signals.enabled=False
                # Alternate execution order, while retaining paired RNG keys and singleton layouts.
                order=['none','evidence'] if units%2 else ['evidence','none']
                for arm in order:
                    journal.unit=f'{case["id"]}:{seed}:pair-{arm}';reset_cache(raw)
                    injected=() if arm=='none' else tuple(packet['ids'])
                    physical=root+branch+injected
                    pair_seed=SeedStream(unit_seed).derive('probe_pair',token_hash(branch))
                    before=_snapshot(raw);started=time.perf_counter()
                    # Identical seed, singleton layout and continuation budgets in both arms.
                    sampling=SamplingConfig(eos_token_id=raw.tokenizer.eos_token_id)
                    request=GenerationRequest(physical,cfg.thinking_tokens-position,sampling,pair_seed,
                        f'pair:{case["id"]}:{arm}',stop_sequences=(close,),statistic=TokenStatistic(sampling,top_k=5),intermediate_events=False)
                    sample,=raw.sample_batch([request]);sc,cn=tap.take(request,sample)
                    call={'request':asdict(request),'prefix_ids':list(physical),'token_ids':list(sample.token_ids),
                          'lp':list(sample.token_logprobs),'sc':list(sc),'cn':list(cn),'finish_reason':sample.finish_reason}
                    answer=finish_answer(raw,physical,sample.token_ids,close,cfg,SeedStream(pair_seed).derive('paired_answer'))
                    after=_snapshot(raw)
                    write({'kind':'probe_pair','question_id':case['id'],'seed':seed,'config_id':'pair-'+arm,
                        'arm':arm,'feature_schema':FEATURE_SCHEMA,'features':features,'decision':decision,
                        'evidence_features':evidence,'evidence_schema':EVIDENCE_SCHEMA,
                        'branch_ids':list(branch),'base_ids':list(root),'injected_ids':list(injected),'packet':packet,'retrieval':retrieval,
                        'native_calls':[],'auxiliary_calls':[call],'answer':answer,'pair_seed':pair_seed,
                        'total_generation_tokens':len(sample.token_ids)+len(answer['answer_ids']),
                        'cost':{k:after[k]-before[k] for k in before.keys()&after.keys() if type(before[k]) in (int,float)},
                        'seconds':time.perf_counter()-started})
                    units+=1
                    print(json.dumps({'completed':units,'case':case['id'],'pair':arm}),flush=True)
                assert not signals.rows and not signals.pending and not dual.rows
            write({'kind':'completed','units':units})
        except BaseException as error:
            write({'kind':'failed','error':repr(error),'unit':journal.unit if journal else 'setup'});raise
        finally:
            if journal is not None:journal.close()
            if dual is not None:dual.close()
            if signals is not None:signals.close()
            if graph is not None:graph.close()
            if raw is not None:close_backend(raw)


if __name__=='__main__':main()
