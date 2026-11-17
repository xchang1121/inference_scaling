import copy
from dataclasses import replace
import json

import pytest

from inference_scaling.retrieval.auxiliary_capture import policy_disagreement, validate_auxiliary_batch
from inference_scaling.retrieval.auxiliary_online_is import AuxiliaryPolicy
from inference_scaling.retrieval.auxiliary_signals import hidden_disagreement
from inference_scaling.retrieval.cot_guidance import GuidanceConfig
from inference_scaling.retrieval.multilayer_capture import SCHEMA
from inference_scaling.retrieval.tuned_online_is import TuningPolicy, run_tuned_is
from inference_scaling.shared.rng import SeedStream
from test_auxiliary_multilayer import calls
from test_auxiliary_online_is import HiddenNative, TwoFacts


@pytest.mark.parametrize('layer,start', [(16, 0), (32, 128), (48, 256), ('final', 384)])
def test_explicit_layers_match_independent_existing_score_and_preserve_input(layer, start):
    values = calls()
    original = copy.deepcopy(values)
    policy = AuxiliaryPolicy(method='seakr', capture_layer=layer)
    actual = policy_disagreement(policy, values)
    view = copy.deepcopy(values)
    for call in view:
        for row in call['internal_signals']:
            row['hidden_projection'] = row['hidden_projection'][start:start+128]
    assert {k: v for k, v in actual.items() if k != 'capture'} == hidden_disagreement(view)
    assert actual['capture']['schema'] == SCHEMA and actual['capture']['layer'] == layer
    assert values == original
    assert policy_disagreement(AuxiliaryPolicy(method='seakr'), view) == hidden_disagreement(view)


@pytest.mark.parametrize('layer', [True, 16., '16', 0, 15, 63, 'last', [], {}])
def test_reject_invalid_layer_types(layer):
    with pytest.raises(ValueError):
        AuxiliaryPolicy(method='seakr', capture_layer=layer)


@pytest.mark.parametrize('defect', ['legacy513', 'layer128', 'alignment', 'unused_nan', 'missing', 'flare'])
def test_captured_rows_reject_invalid_schema_even_outside_tail(defect):
    values = calls()
    policy = AuxiliaryPolicy(method='seakr', capture_layer=16, hidden_tail=1)
    if defect == 'legacy513':
        policy = replace(policy, capture_layer=None)
    elif defect == 'layer128':
        values[0]['internal_signals'][0]['hidden_projection'] = [1.]*128
    elif defect == 'alignment':
        values[0]['internal_signals'][0]['hidden_projection'][512] = .001
    elif defect == 'unused_nan':
        values[0]['internal_signals'][0]['hidden_projection'][300] = float('nan')
    elif defect == 'missing':
        values[0]['internal_signals'] = None
    if defect == 'flare':
        with pytest.raises(ValueError):
            replace(policy, method='flare')
    else:
        with pytest.raises(ValueError):
            policy_disagreement(policy, values)


def batch():
    return {'family': 'seakr', 'multilayer_auxiliary': True, 'diagnostic_worker': 'multilayer',
        'cases': [{'id': 'TRAIN_Q334'}, {'id': 'TRAIN_Q417'}], 'seeds': [1],
        'configs': [{'id': str(layer), 'arm': 'tuned_online', 'collect_signals': 'multilayer',
            'auxiliary_policy': {'method': 'seakr', 'capture_layer': layer},
            'policy': {'trigger': 'fixed'}} for layer in (16, 32, 'final')]}


def test_explicit_batch_has_separate_guard():
    value = batch()
    validate_auxiliary_batch(value)
    from inference_scaling.retrieval.multilayer_capture import validate_diagnostic_batch
    with pytest.raises(ValueError):
        validate_diagnostic_batch(value)


@pytest.mark.parametrize('defect', ['flag', 'worker', 'legacy_layer', 'legacy_mode', 'trigger', 'query'])
def test_reject_mixed_batch_routes(defect):
    value = batch()
    if defect == 'flag':
        value.pop('multilayer_auxiliary')
    elif defect == 'worker':
        value['diagnostic_worker'] = 'hidden'
    elif defect == 'legacy_layer':
        value['configs'][0]['auxiliary_policy'].pop('capture_layer')
    elif defect == 'legacy_mode':
        value['configs'][0]['collect_signals'] = 'hidden'
    elif defect == 'trigger':
        value['configs'][0]['policy']['trigger'] = 'entropy'
    elif defect == 'query':
        value['configs'][0]['policy']['query'] = 'attention'
    with pytest.raises(ValueError):
        validate_auxiliary_batch(value)


class CapturedNative(HiddenNative):
    def take(self, request, sample):
        result = super().take(request, sample)
        for row in self.history[request.request_id]:
            pair = row['hidden_projection']
            row['hidden_projection'] = [pair[i % 2]+block*.2 for block in range(4) for i in range(128)]+[0.]
        return result


def test_full_candidate_local_layer_gate_trials_cost_and_audit():
    from audit_tuned_online_is import audit_tuned_row
    backend = CapturedNative()
    cfg = replace(GuidanceConfig(), thinking_tokens=640, answer_tokens=1024, context_tokens=8192)
    result = run_tuned_is(backend, backend, (1, 2), (901,), SeedStream(1).derive('question', 'synthetic'), cfg,
        question='Q', facts=TwoFacts(), policy=TuningPolicy(),
        auxiliary_policy=AuxiliaryPolicy(method='seakr', capture_layer=32, draft_tokens=16,
            threshold=0., work_tokens=1000, evidence_candidates=2, evidence_tokens=16))
    for call in result['native_calls']:
        call['internal_signals'] = backend.history[call['request_id']]
    row = json.loads(json.dumps({'unit_id': 'synthetic', 'arm': 'tuned_online', 'question': 'Q',
        'question_id': 'synthetic', 'seed': 1, 'base_ids': [1, 2], 'static_packets': [],
        'static_injected_ids': [], 'cost': {'generated_tokens': result['total_generation_tokens']},
        'signal_collection': {'requested': 'multilayer', 'hidden_dimensions': 513, 'attention_topk': 0,
            'hidden_source': 'multilayer_diagnostic', 'projection_seed': 2026111502, 'state_offset': -1, 'schema': SCHEMA},
        **result}))
    def audit(value):
        failures = []
        audit_tuned_row(value, cfg, lambda ids, skip=False: backend.decode(ids, skip_special_tokens=skip),
            900, 901, lambda ok, label: failures.append(label) if not ok else None)
        return failures
    assert not audit(row)
    trials = [t for d in row['decisions'] if 'retrieval' in d for t in d['retrieval']['evidence_trials']['trials']]
    assert trials and all(t['feature']['capture']['layer'] == 32 for t in trials)
    assert row['auxiliary_work'] == sum(len(c['token_ids']) for c in row['auxiliary_calls'])
    broken = copy.deepcopy(row)
    broken['signal_collection']['state_offset'] = 0
    assert any('auxiliary_capture_schema' in x for x in audit(broken))
    broken = copy.deepcopy(row)
    broken['auxiliary_calls'][0]['internal_signals'][0]['hidden_projection'][512] = .1
    with pytest.raises(ValueError):
        audit(broken)
