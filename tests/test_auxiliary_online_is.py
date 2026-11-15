import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from test_online_is import Facts
from test_smart_online_is import ReadableNative
from inference_scaling.retrieval.auxiliary_online_is import AuxiliaryPolicy
from inference_scaling.retrieval.cot_guidance import GuidanceConfig
from inference_scaling.retrieval.tuned_online_is import TuningPolicy, run_tuned_is
from inference_scaling.shared.rng import SeedStream


class HiddenNative(ReadableNative):
    def __init__(self):
        super().__init__(stop=180,eos=True)
        self.history = {}

    def take(self, request, sample):
        result = super().take(request,sample)
        self.history[request.request_id] = [{'hidden_projection':[float(t%7),float(t%11)]}
                                           for t in sample.token_ids]
        return result


class TwoFacts(Facts):
    def retrieve(self,query,*,k=1,exclude=frozenset()):
        found,log=super().retrieve(query,k=k,exclude=exclude)
        other={**found[0],'id':found[0]['id']+'b','ids':[20,72,21]}
        found=(found+[other])[:k]
        return found,{**log,'selected_ids':[p['id'] for p in found]}


@pytest.mark.parametrize(('method','trials','work'), [('flare',0,300),('seakr',0,300),
                                                    ('seakr',2,1000),('seakr',2,60)])
def test_paid_lookahead_is_separate_from_frozen_IS_and_fully_replayable(method,trials,work):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from audit_tuned_online_is import audit_tuned_row
    backend = HiddenNative()
    cfg = replace(GuidanceConfig(),thinking_tokens=640,answer_tokens=1024,context_tokens=8192)
    result = run_tuned_is(backend,backend,(1,2),(901,),SeedStream(1).derive('question','synthetic'),cfg,
        question='Q',facts=TwoFacts(),policy=TuningPolicy(),
        auxiliary_policy=AuxiliaryPolicy(method=method,draft_tokens=16,threshold=0.,work_tokens=work,
                                         evidence_candidates=trials,evidence_tokens=16))
    row = json.loads(json.dumps({'unit_id':'synthetic','arm':'tuned_online','question':'Q',
        'question_id':'synthetic','seed':1,'base_ids':[1,2],'static_packets':[],
        'static_injected_ids':[],'cost':{'generated_tokens':result['total_generation_tokens']},**result}))
    assert backend.requests[0].prefix == (1,2)
    assert result['auxiliary_work'] > 0
    assert result['search_work'] == sum(len(c['token_ids']) for c in result['native_calls'])
    assert result['total_generation_tokens'] == sum(len(c['token_ids']) for c in result['native_calls']+result['auxiliary_calls'])+len(result['answer']['answer_ids'])
    def audit(target):
        failures = []
        audit_tuned_row(target,cfg,lambda ids,skip=False:backend.decode(ids,skip_special_tokens=skip),
                        900,901,lambda ok,label:failures.append(label) if not ok else None)
        return failures
    assert not audit(row)
    if trials:
        trial_rows=[d['retrieval']['evidence_trials'] for d in row['decisions'] if 'retrieval' in d]
        assert trial_rows
        if work == 60:
            assert all(r['budget_exhausted'] and not r['trials'] for r in trial_rows)
            assert not row['events']
        else:
            completed=[r for r in trial_rows if not r['budget_exhausted']]
            assert completed and all(len(r['trials']) == 2 for r in completed)
            broken=copy.deepcopy(row)
            target=next(d['retrieval']['evidence_trials'] for d in broken['decisions']
                        if 'retrieval' in d and d['retrieval']['evidence_trials']['trials'])
            target['trials'][0]['feature']['value']+=1
            assert any('trial_hidden_feature_replay' in x for x in audit(broken))
    broken = copy.deepcopy(row)
    broken['auxiliary_calls'][0]['prefix_ids'][0] += 1
    assert any('auxiliary_exact_physical_prefix' in x for x in audit(broken))
    broken = copy.deepcopy(row)
    broken['auxiliary_work'] = 0
    assert any('complete_native_cost' in x for x in audit(broken))
