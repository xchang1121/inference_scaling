"""TRAIN-only continuation diagnostic: fixed prefix/seed, cache reset, templates."""
from pathlib import Path
import hashlib
import json
import os
import time

from benchmark_is_gpu import reset_cache
from inference_scaling.app.settings import load_settings
from inference_scaling.arllm.backends.loader import load_backend,close_backend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest,TokenStatistic
from inference_scaling.retrieval.controller import _clean_reference
from inference_scaling.retrieval.cot_guidance import DualStatistics
from inference_scaling.retrieval.stepwise import token_hash

def main():
    dep=Path('/deployment/online-injection-literature-20261112');out=Path('/results')
    settings=load_settings(dep/'settings.json')
    rows=[json.loads(s) for s in (out/'smoke.jsonl').read_text().splitlines()]
    cases=[r for r in rows if r['kind']=='measurement' and r['arm']=='thinkretrieve']
    examples={r['id']:r for r in map(json.loads,(dep/'data/examples.jsonl').read_text().splitlines())}
    assert len(cases)==2
    assert not (out/'diagnostic.jsonl').exists()
    
    # Official ThinkRetrieve ICE template and Example.format layout, MIT licensed:
    # Copyright (c) 2026 Vaibhav Singh. See packaged upstream LICENSE.
    head='\n\nWait, let me check again. Here is a similar question and its solution. Looking at this example:\n'
    tail=('\n\nNow coming back to the original question, Here are the key points I can take from example question '
          'to solve the original question:\n')
    raw=tap=None
    with (out/'diagnostic.jsonl').open('w') as stream:
        def write(row):
            stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
        write({'kind':'metadata','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'smoke_sha256':hashlib.sha256((out/'smoke.jsonl').read_bytes()).hexdigest(),
            'settings_sha256':hashlib.sha256((dep/'settings.json').read_bytes()).hexdigest(),
            'purpose':'No F1 grading; replay first post-insertion continuation with unchanged sampling and seed.'})
        try:
            raw=load_backend(settings['ar']['model'],settings['ar']['engine'],seed=20261112,logprobs=6)
            tap=DualStatistics(raw)
            for row in cases:
                h=row['handoff'];first=h['controller_calls'][0];following=h['controller_calls'][1]
                close=248069;eos=raw.tokenizer.eos_token_id
                stop=next((i for i,t in enumerate(first['token_ids']) if t in {eos,close}),len(first['token_ids']))
                stem=list(first['prefix_ids'])+first['token_ids'][:stop]
                current=stem+h['injected_ids'][0]
                assert current==following['prefix_ids']
                ex=examples[h['packets'][0]['id']]
                body='## Example 1:\nProblem:\n'+_clean_reference(ex['question'])+'\n\nSolution:\n'+_clean_reference(ex['solution'])
                official=list(raw.encode(head+body+tail,add_special_tokens=False))
                rethink=list(raw.encode('\nWait, let me think again.\n',add_special_tokens=False))
                variants={'cold_current':current,'cold_official':stem+official,'cold_rethink':stem+rethink}
                write({'kind':'recorded','question_id':row['question_id'],'call':following,
                    'example_id':ex['id'],'first_injected_ids':h['injected_ids'][0]})
                for variant,prefix in variants.items():
                    reset_cache(raw)
                    sampling=SamplingConfig(eos_token_id=eos)
                    request=GenerationRequest(tuple(prefix),following['max_new_tokens'],sampling,following['seed'],
                        row['question_id']+':'+variant,statistic=TokenStatistic(sampling,top_k=5),
                        stop_sequences=((close,),),intermediate_events=False)
                    started=time.perf_counter();sample,=raw.sample_batch([request]);elapsed=time.perf_counter()-started
                    sc,cn=tap.take(request,sample)
                    write({'kind':'diagnosis','question_id':row['question_id'],'variant':variant,
                        'prefix_ids':prefix,'prefix_sha256':token_hash(prefix),'seed':request.seed,
                        'max_new_tokens':request.max_new_tokens,'token_ids':list(sample.token_ids),
                        'finish_reason':sample.finish_reason,'sc':list(sc),'cn':list(cn),
                        'seconds':elapsed,'text':raw.decode(tuple(sample.token_ids),skip_special_tokens=False)})
                    print(json.dumps({'question':row['question_id'],'variant':variant,'tokens':len(sample.token_ids),
                        'first':list(sample.token_ids[:1]),'seconds':elapsed}),flush=True)
            write({'kind':'completed','units':len(cases)*3})
        except BaseException as e:
            write({'kind':'failed','error':repr(e)});raise
        finally:
            if tap is not None:tap.close()
            if raw is not None:close_backend(raw)


if __name__=='__main__':main()
