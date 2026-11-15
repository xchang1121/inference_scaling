import copy
from dataclasses import asdict,replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace,MethodType

from test_online_is import Facts
from test_smart_online_is import ReadableNative
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest,TokenStatistic
from inference_scaling.retrieval.cot_guidance import GuidanceConfig,finish_answer
from inference_scaling.retrieval.probe_features import prefix_features,FEATURE_SCHEMA,evidence_features,EVIDENCE_SCHEMA
from inference_scaling.retrieval.tuned_online_is import TunedOnlineLedger,TuningPolicy,run_tuned_is
from inference_scaling.retrieval.stepwise import token_hash
from inference_scaling.shared.rng import SeedStream


class FeaturesNative(ReadableNative):
    def __init__(self):super().__init__(stop=180,eos=True);self.history={}
    def take(self,request,sample):
        result=super().take(request,sample)
        self.history[request.request_id]=[{'hidden_projection':[t/100]*128,'entropy':.2,'top2_gap':.7,
            'p_eos':.01,'p_think_end':.02} for t in sample.token_ids]
        return result


def test_prefix_features_exclude_future_and_pair_audit_rejects_seed_or_feature_leak():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_probe_pairs import audit_probe_pair
    b=FeaturesNative();cfg=replace(GuidanceConfig(),thinking_tokens=640,answer_tokens=1024,context_tokens=8192)
    unit=SeedStream(1).derive('question','synthetic')
    baseline=run_tuned_is(b,b,(1,2),(901,),unit,cfg,question='Q',facts=Facts(),policy=TuningPolicy(trigger='never',positions=(128,)))
    baseline.update(question_id='synthetic',seed=1,question='Q')
    d=baseline['decisions'][0];branch=tuple(d['branch_ids'])
    proxy=SimpleNamespace(root=(1,2),native_calls=baseline['native_calls'][:d['observed_native_calls']],tap=b)
    proxy.recent=MethodType(TunedOnlineLedger.recent,proxy)
    features=prefix_features(proxy.recent(branch,32));paired_seed=SeedStream(unit).derive('probe_pair',token_hash(branch))
    packet={'ids':[73,74],'sha256':token_hash([73,74]),'body_start':0,'body_stop':2};pairs=[]
    evidence=evidence_features('Q','Q',packet,lambda ids,skip:b.decode(ids,skip_special_tokens=skip))
    for arm in ('none','evidence'):
        injected=[] if arm=='none' else packet['ids'];physical=(1,2)+branch+tuple(injected)
        sampling=SamplingConfig(eos_token_id=900)
        request=GenerationRequest(physical,cfg.thinking_tokens-len(branch),sampling,paired_seed,arm,
            stop_sequences=((901,),),statistic=TokenStatistic(sampling,top_k=5),intermediate_events=False)
        sample,=b.sample_batch([request]);sc,cn=b.take(request,sample)
        answer=finish_answer(b,physical,sample.token_ids,(901,),cfg,SeedStream(paired_seed).derive('paired_answer'))
        count=len(sample.token_ids)+len(answer['answer_ids'])
        pairs.append({'question_id':'synthetic','seed':1,'arm':arm,'features':features,'feature_schema':FEATURE_SCHEMA,
            'decision':d,'branch_ids':list(branch),'base_ids':[1,2],'pair_seed':paired_seed,'packet':packet,'retrieval':{'query':'Q'},
            'evidence_features':evidence,'evidence_schema':EVIDENCE_SCHEMA,
            'injected_ids':injected,'auxiliary_calls':[{'request':asdict(request),'prefix_ids':list(physical),
                'token_ids':list(sample.token_ids),'lp':list(sample.token_logprobs),'sc':list(sc),'cn':list(cn),'finish_reason':sample.finish_reason}],
            'answer':answer,'total_generation_tokens':count,'cost':{'generated_tokens':count}})
    baseline,pairs=json.loads(json.dumps([baseline,pairs]))
    def audit(rows):
        failures=[]
        audit_probe_pair(baseline,rows,cfg,lambda ids,skip:b.decode(ids,skip_special_tokens=skip),900,901,
                         lambda ok,label:failures.append(label) if not ok else None)
        return failures
    assert not audit(pairs)
    bad=copy.deepcopy(pairs);bad[0]['features'][0]+=1
    assert any('causal_hidden_features' in f for f in audit(bad))
    bad=copy.deepcopy(pairs);bad[1]['auxiliary_calls'][0]['request']['seed']+=1
    assert any('paired_suffix_rng_budget' in f for f in audit(bad))
