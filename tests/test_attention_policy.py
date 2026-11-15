import copy
import json
from dataclasses import replace
from pathlib import Path
import sys

import pytest

from test_online_is import Facts
from test_smart_online_is import ReadableNative
from inference_scaling.retrieval.attention_policy import attention_feature
from inference_scaling.retrieval.attention_policy import query_text
from inference_scaling.retrieval.cot_guidance import GuidanceConfig
from inference_scaling.retrieval.tuned_online_is import TuningPolicy, run_tuned_is
from inference_scaling.shared.rng import SeedStream


def test_need_uses_observed_later_attention_and_sparse_zero_lower_bound():
    def decode(ids, **kwargs):
        return ' '.join({1:'the',2:'DB2',3:'error',4:'fix'}[i] for i in ids)
    def row(token,index,entropy,indices,weights,ids):
        return {'token':token,'physical_index':index,'entropy':entropy,
                'attention':{'prefix_tokens':index,'indices':indices,'weights':weights},
                'attention_token_ids':ids}
    rows=[row(2,10,2.,[0,2],[.7,.2],[1,3]),row(4,11,9.,[10,1],[.3,.2],[2,1])]
    assert attention_feature(rows[:1],decode)['value'] == 0.
    feature=attention_feature(rows,decode)
    assert feature['value'] == pytest.approx(.6)
    assert feature['uncertain_physical_index'] == 10
    assert feature['query_indices'] == [2]  # stopword with greater attention is filtered
    assert feature['query_text'] == 'error'
    broken=copy.deepcopy(rows)
    broken[0]['attention']['prefix_tokens']=12
    with pytest.raises(ValueError,match='precede'):
        attention_feature(broken,decode)


def test_attention_query_expands_subword_without_joining_unrelated_gaps():
    def decode(ids,**kwargs):return ''.join({1:'The ',2:'DB',3:'2',4:' returns ',5:'error'}[i] for i in ids)
    assert query_text((1,2,3,4,5),[(2,.8,3),(4,.1,5)],decode,True)=='DB2 error'
    assert query_text(None,[(1,.8,2),(2,.7,3),(4,.1,5)],decode,False)=='DB2 error'


class AttentionNative(ReadableNative):
    def __init__(self):
        super().__init__(stop=180,eos=True)
        self.history={}

    def take(self,request,sample):
        result=super().take(request,sample)
        states=[]
        for i in range(len(sample.token_ids)):
            n=len(request.prefix)+i
            states.append({'entropy':2.,'attention':{'prefix_tokens':n,
                'indices':[n-1,0], 'weights':[.5,.1], 'layer':63,
                'relative_output_error':0.,'normalization_error':0.}})
        self.history[request.request_id]=states
        return result


def test_attention_controller_replays_full_IS_and_rejects_changed_attention():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_tuned_online_is import audit_tuned_row
    backend=AttentionNative()
    cfg=replace(GuidanceConfig(),thinking_tokens=640,answer_tokens=1024,context_tokens=8192)
    policy=TuningPolicy(trigger='dragin',query='attention',threshold=.1,attention_content_only=False)
    result=run_tuned_is(backend,backend,(1,2),(901,),SeedStream(1).derive('question','synthetic'),cfg,
                        question='Q',facts=Facts(),policy=policy)
    row=json.loads(json.dumps({'unit_id':'synthetic','arm':'tuned_online','question':'Q',
        'question_id':'synthetic','seed':1,'base_ids':[1,2],'static_packets':[],
        'static_injected_ids':[],'cost':{'generated_tokens':result['total_generation_tokens']},**result}))
    def audit(target):
        failures=[]
        audit_tuned_row(target,cfg,lambda ids,skip=False:backend.decode(ids,skip_special_tokens=skip),
                        900,901,lambda ok,label:failures.append(label) if not ok else None)
        return failures
    assert row['events'] and not audit(row)
    broken=copy.deepcopy(row)
    for call in broken['native_calls']:
        for state in call.get('internal_signals',[]):
            state['attention']['weights']=[0.,0.]
    assert any('gate_replay' in f for f in audit(broken))
