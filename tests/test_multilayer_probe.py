"""Captured layers must be explicit, causal, aligned and independently replayable."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from audit_tuned_online_is import audit_tuned_row
from test_probe_pairs import FeaturesNative
from test_gap_online_is import TextFacts
from inference_scaling.retrieval.cot_guidance import GuidanceConfig
from inference_scaling.retrieval.linear_probe import fit_probe, fingerprint, predict_probe
from inference_scaling.retrieval.multilayer_capture import SCHEMA, validate_multilayer_batch, validate_diagnostic_batch
from inference_scaling.retrieval.probe_features import prefix_features, pooled_feature_schema, validate_feature_schema
from inference_scaling.retrieval.probe_online_is import ProbePolicy
from inference_scaling.retrieval.tuned_online_is import TuningPolicy, run_tuned_is
from inference_scaling.shared.rng import SeedStream


def model(layer=16, target='knowledge'):
    value = fit_probe(np.zeros((4,134 if target == 'knowledge' else 142)),np.full(4,.5),target=target)
    value['feature_schema'] = pooled_feature_schema(capture_layer=layer)
    value['sha256'] = fingerprint(value)
    return value


def collection():
    return {'requested':'multilayer','hidden_dimensions':513,'attention_topk':0,
        'hidden_source':'multilayer_diagnostic','projection_seed':2026111502,'state_offset':-1,'schema':deepcopy(SCHEMA)}


def batch():
    return {'multilayer_policy':True,'diagnostic_worker':'multilayer','configs':[
        {'arm':'tuned_online','collect_signals':'multilayer','probe_policy':{'model':model()}},
        {'arm':'tuned_online','collect_signals':'multilayer','policy':{'trigger':'never'}}]}


class MultiNative(FeaturesNative):
    def take(self,request,sample):
        result = super().take(request,sample)
        for row in self.history[request.request_id]:
            vector = row['hidden_projection']
            row['hidden_projection'] = [v+layer for layer in (16,32,48,0) for v in vector]+[0.]
        return result


@pytest.mark.parametrize('layer,offset',[(16,0),(32,128),(48,256),('final',384)])
def test_exact_block_and_old_schema(layer,offset):
    rows = [{'hidden_projection':[float(i+n) for i in range(512)]+[0.],
        'entropy':.2,'top2_gap':.3,'lp':-.2,'sc':-.3,'p_eos':.01,'p_think_end':.02} for n in range(40)]
    old = [{**r,'hidden_projection':r['hidden_projection'][offset:offset+128]} for r in rows]
    assert prefix_features(rows,capture_layer=layer) == prefix_features(old)
    assert pooled_feature_schema()['version'] == 1
    validate_feature_schema(pooled_feature_schema(capture_layer=layer))
    with pytest.raises(ValueError):
        prefix_features(rows)
    for index,value in ((512,.01),(3,float('nan'))):
        bad = deepcopy(rows);bad[-1]['hidden_projection'][index] = value
        with pytest.raises(ValueError):
            prefix_features(bad,capture_layer=layer)


@pytest.mark.parametrize('key,value',[('capture_layer',17),('capture_layer',True),('capture_block_start',128),
    ('capture_dimensions',512),('state_offset',0),('projection_seed',1),('hidden_source','wrong')])
def test_reject_corrupt_layer_identity(key,value):
    m = model();m['feature_schema'][key] = value;m['sha256'] = fingerprint(m)
    with pytest.raises(ValueError):
        ProbePolicy(m)
    with pytest.raises(ValueError):
        predict_probe(m,[0.]*134)


def test_policy_marker_does_not_weaken_diagnostic_guard():
    validate_multilayer_batch(batch())
    with pytest.raises(ValueError):
        validate_diagnostic_batch(batch())
    for key,value in [('multilayer_policy',False),('diagnostic_worker','hidden')]:
        b = batch();b[key] = value
        with pytest.raises(ValueError):
            validate_multilayer_batch(b)
    for key,value in [('collect_signals','none'),('auxiliary_policy',{}),('alignment_policy',{}),
            ('policy',{'query':'attention'}),('policy',{'trigger':'never'})]:
        b = batch();b['configs'][0][key] = value
        with pytest.raises(ValueError):
            validate_multilayer_batch(b)
    b = batch();b['configs'][0]['probe_policy']['model'] = model(None)
    with pytest.raises(ValueError):
        validate_multilayer_batch(b)


@pytest.mark.parametrize('layer',[16,32,48,'final'])
@pytest.mark.parametrize('target,threshold,injected',[('knowledge',.2,True),('knowledge',.8,False),
    ('value',.2,True),('value',.8,False),('risk',.8,True),('risk',.2,False)])
def test_native_policy_causal_audit_and_mutation_rejection(layer,target,threshold,injected):
    b = MultiNative();cfg = replace(GuidanceConfig(),thinking_tokens=640,answer_tokens=1024,context_tokens=8192)
    result = run_tuned_is(b,b,(1,2),(901,),SeedStream(1).derive('question','synthetic'),cfg,question='Q',
        facts=TextFacts(b),policy=TuningPolicy(),probe_policy=ProbePolicy(model(layer,target),threshold))
    row = json.loads(json.dumps({'unit_id':'synthetic','question':'Q','question_id':'synthetic','seed':1,
        'base_ids':[1,2],'static_packets':[],'static_injected_ids':[],
        'cost':{'generated_tokens':result['total_generation_tokens']},'signal_collection':collection(),**result}))
    assert bool(row['events']) == injected
    assert b.requests[0].prefix == (1,2)
    def audit(value):
        failures = []
        audit_tuned_row(value,cfg,lambda ids,skip=False:b.decode(ids,skip_special_tokens=skip),900,901,
            lambda ok,label:failures.append(label) if not ok else None)
        return failures
    assert not audit(row)
    bad = deepcopy(row);bad['signal_collection']['hidden_dimensions'] = 128
    assert any('collection_identity' in f for f in audit(bad))
    bad = deepcopy(row);bad['decisions'][0]['signal']['probe_features'][0] += 1
    assert any('causal_gate_replay' in f for f in audit(bad))
    bad = deepcopy(row);bad['native_calls'][-1]['internal_signals'][-1]['hidden_projection'][512] = .01
    assert any('all_native_alignment' in f for f in audit(bad))
