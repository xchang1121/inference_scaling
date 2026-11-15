import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from test_probe_pairs import FeaturesNative
from test_gap_online_is import TextFacts
from inference_scaling.retrieval.cot_guidance import GuidanceConfig
from inference_scaling.retrieval.linear_probe import fit_probe
from inference_scaling.retrieval.probe_online_is import ProbePolicy
from inference_scaling.retrieval.tuned_online_is import TuningPolicy,run_tuned_is
from inference_scaling.shared.rng import SeedStream


@pytest.mark.parametrize(('target','threshold','injected'),[('knowledge',.2,True),('knowledge',.8,False),
    ('value',.2,True),('value',.8,False),('risk',.8,True),('risk',.2,False)])
@pytest.mark.parametrize('pool',[8,32,128])
def test_learned_gate_and_evidence_policy_remain_causal_replayable(target,threshold,injected,pool):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_tuned_online_is import audit_tuned_row
    model=fit_probe(np.zeros((4,134 if target=='knowledge' else 142)),np.full(4,.5),target=target,pool=pool)
    b=FeaturesNative();cfg=replace(GuidanceConfig(),thinking_tokens=640,answer_tokens=1024,context_tokens=8192)
    result=run_tuned_is(b,b,(1,2),(901,),SeedStream(1).derive('question','synthetic'),cfg,question='Q',
        facts=TextFacts(b),policy=TuningPolicy(),probe_policy=ProbePolicy(model,threshold))
    row=json.loads(json.dumps({'unit_id':'synthetic','question':'Q','question_id':'synthetic','seed':1,
        'base_ids':[1,2],'static_packets':[],'static_injected_ids':[],
        'cost':{'generated_tokens':result['total_generation_tokens']},**result}))
    assert bool(row['events'])==injected
    assert b.requests[0].prefix==(1,2)
    def audit(r):
        failures=[]
        audit_tuned_row(r,cfg,lambda ids,skip=False:b.decode(ids,skip_special_tokens=skip),900,901,
                        lambda ok,label:failures.append(label) if not ok else None)
        return failures
    assert not audit(row)
    broken=copy.deepcopy(row);broken['decisions'][0]['signal']['probe_features'][0]+=1
    assert any('causal_gate_replay' in f for f in audit(broken))
    if target!='knowledge':
        broken=copy.deepcopy(row)
        d=next(d for d in broken['decisions'] if 'retrieval' in d)
        d['retrieval']['probe_ranking']['trials'][0]['prediction']+=1
        assert any('probe_prediction_and_gate' in f for f in audit(broken))
