"""Captured retrieval must preserve layer identity, pooling and causal audit."""
from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pytest

from audit_tuned_online_is import audit_tuned_row
from test_multilayer_probe import collection
from test_probe_pairs import FeaturesNative
from inference_scaling.retrieval.aligned_online_is import AlignmentPolicy
from inference_scaling.retrieval.corpus import Document, Retriever
from inference_scaling.retrieval.cot_guidance import GuidanceConfig
from inference_scaling.retrieval.hidden_alignment import DocumentVectors, fit_alignment, project_hidden
from inference_scaling.retrieval.linear_probe import fingerprint
from inference_scaling.retrieval.multilayer_capture import validate_diagnostic_batch, validate_multilayer_batch
from inference_scaling.retrieval.literature import FactBank
from inference_scaling.retrieval.tuned_online_is import TuningPolicy, run_tuned_is
from inference_scaling.shared.rng import SeedStream


def model(layer=48, pool=32):
    targets = np.zeros((2, 768));targets[:, 0] = 1.
    return fit_alignment(np.zeros((2, 128)), targets, capture_layer=layer, pool=pool,
                         provenance={'vector_sha256':'synthetic'})


def batch():
    return {'multilayer_alignment':True, 'diagnostic_worker':'multilayer', 'configs':[
        {'arm':'tuned_online', 'collect_signals':'multilayer', 'alignment_policy':{'model':model()}}]}


class LayersNative(FeaturesNative):
    def take(self, request, sample):
        result = super().take(request, sample)
        for i, row in enumerate(self.history[request.request_id]):
            position = len(request.prefix)+i
            row['hidden_projection'] = [float(position+layer) for layer in (16,32,48,0) for _ in range(128)]+[0.]
        return result


def test_default_model_schema_and_projection_remain_exact():
    rng = np.random.default_rng(77)
    x, y = rng.normal(size=(8,128)), rng.normal(size=(8,768))
    old = fit_alignment(x,y)
    assert old == fit_alignment(x,y,pool=32,capture_layer=None)
    captured = fit_alignment(x,y,pool=16,capture_layer=48)
    assert np.array_equal(project_hidden(old,x[0]),project_hidden(captured,x[0]))
    for key, value in [('capture_layer',16),('capture_block_start',0),('pool',True),('state_offset',0)]:
        bad = deepcopy(captured);bad['feature_schema'][key] = value;bad['sha256'] = fingerprint(bad)
        with pytest.raises(ValueError):
            AlignmentPolicy(bad)
        with pytest.raises(ValueError):
            project_hidden(bad,x[0])


def test_captured_alignment_requires_separate_batch_and_matching_worker():
    validate_multilayer_batch(batch())
    with pytest.raises(ValueError):
        validate_diagnostic_batch(batch())
    for key, value in [('multilayer_alignment',False),('multilayer_policy',True),
                       ('multilayer_auxiliary',True),('diagnostic_worker','hidden'),('attention_geometry','layer63-all')]:
        b = batch();b[key] = value
        with pytest.raises(ValueError):
            validate_multilayer_batch(b)
    for key, value in [('collect_signals','hidden'),('probe_policy',{}),('auxiliary_policy',{}),
                       ('policy',{'query':'attention'}),('policy',{'trigger':'never'})]:
        b = batch();b['configs'][0][key] = value
        with pytest.raises(ValueError):
            validate_multilayer_batch(b)
    b = batch();b['configs'][0]['alignment_policy']['model'] = model(None)
    with pytest.raises(ValueError):
        validate_multilayer_batch(b)


@pytest.mark.parametrize('layer', [48, 'final'])
@pytest.mark.parametrize('pool', [16, 32])
def test_full_causal_alignment_and_rejected_corrupt_capture(layer,pool):
    vectors = object.__new__(DocumentVectors);vectors.ids = ['a','b'];vectors.sha256 = 'synthetic'
    vectors.vectors = np.zeros((2,768));vectors.vectors[0,0] = 1.;vectors.vectors[1,1] = 1.
    backend = LayersNative()
    cfg = replace(GuidanceConfig(),thinking_tokens=640,answer_tokens=1024,context_tokens=8192)
    index = Retriever([Document('a','Q support','Q facts.'),Document('b','Q related','Q alternatives.')],kind='bm25')
    result = run_tuned_is(backend,backend,(1,2),(901,),SeedStream(1).derive('question','synthetic'),cfg,
        question='Q',facts=FactBank(backend,index),policy=TuningPolicy(),alignment_policy=AlignmentPolicy(model(layer,pool)),
        document_vectors=vectors)
    row = json.loads(json.dumps({'unit_id':'synthetic','question':'Q','question_id':'synthetic','seed':1,
        'base_ids':[1,2],'static_packets':[],'static_injected_ids':[],
        'cost':{'generated_tokens':result['total_generation_tokens']},'signal_collection':collection(),**result}))
    assert backend.requests[0].prefix == (1,2) and row['events']
    assert row['decisions'][0]['signal']['probe_features'][0] == (0 if layer == 'final' else layer)+2+127-(pool-1)/2
    def audit(value):
        failures = []
        audit_tuned_row(value,cfg,lambda ids,skip=False:backend.decode(ids,skip_special_tokens=skip),900,901,
            lambda ok,label:failures.append(label) if not ok else None,document_vectors=vectors)
        return failures
    assert not audit(row)
    bad = deepcopy(row);bad['signal_collection']['schema']['state_offset'] = 0
    assert any('alignment_multilayer_collection_identity' in f for f in audit(bad))
    bad = deepcopy(row);bad['decisions'][0]['signal']['probe_features'][0] += 1.
    assert any('alignment_causal_features' in f for f in audit(bad))
    bad = deepcopy(row);bad['native_calls'][-1]['internal_signals'][-1]['hidden_projection'][512] = .01
    assert any('alignment_multilayer_all_native_alignment' in f for f in audit(bad))
