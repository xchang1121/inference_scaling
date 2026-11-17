import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from retrieval_fixtures import ReadableNative
from inference_scaling.retrieval.cot_guidance import GuidanceConfig
from inference_scaling.retrieval.example_online_is import ExamplePolicy
from inference_scaling.retrieval.evidence import packet
from inference_scaling.retrieval.online_is import TuningPolicy,run_online_is
from inference_scaling.shared.rng import SeedStream


class Examples:
    def __init__(self,b,oversized=False):self.b,self.oversized=b,oversized
    def retrieve(self,question,interim,*,k,exclude):
        body='Problem: Q\nSolution: complete supported explanation.'
        if self.oversized:body*=100
        name='example'+str(len(exclude))
        p=packet(self.b,'\nEXAMPLE\n',body,'\nRETURN\n',id=name)
        return [p],{'query':'query: '+question+'\nsolution: '+interim,'selected_ids':[name],'seconds':0.}


@pytest.mark.parametrize('mode',['valid','unpadded','oversized','budget','require_complete'])
def test_example_causal_answer_whole_solution_and_cost(mode):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_tuned_online_is import audit_tuned_row
    b=ReadableNative(stop=180,eos=True)
    cfg=replace(GuidanceConfig(),thinking_tokens=640,answer_tokens=1024,context_tokens=8192)
    result=run_online_is(b,b,(1,2),(901,),SeedStream(1).derive('question','synthetic'),cfg,
        question='Q',facts=Examples(b,mode=='oversized'),policy=TuningPolicy(),
        example_policy=ExamplePolicy(answer_tokens=16,work_tokens=8 if mode=='budget' else 2000,
                                     require_complete_answer=mode=='require_complete',pad_to_budget=mode!='unpadded'))
    row=json.loads(json.dumps({'unit_id':'synthetic','question':'Q','question_id':'synthetic','seed':1,
        'base_ids':[1,2],'static_packets':[],'static_injected_ids':[],
        'cost':{'generated_tokens':result['total_generation_tokens']},**result}))
    assert b.requests[0].prefix==(1,2)
    assert bool(row['events'])==(mode in {'valid','unpadded'})
    def audit(target):
        failures=[]
        audit_tuned_row(target,cfg,lambda ids,skip=False:b.decode(ids,skip_special_tokens=skip),900,901,
                        lambda ok,label:failures.append(label) if not ok else None)
        return failures
    assert not audit(row)
    if mode in {'valid','unpadded'}:
        broken=copy.deepcopy(row)
        broken['auxiliary_calls'][0]['prefix_ids'][-1]=123
        assert any('exact_auxiliary_prefix' in f for f in audit(broken))
        broken=copy.deepcopy(row)
        broken['events'][0]['packet']['ids'][15]=123
        assert any('whole_worked_solution' in f for f in audit(broken))
