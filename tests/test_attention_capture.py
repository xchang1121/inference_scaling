import copy

import pytest

from inference_scaling.retrieval.attention_capture import validate_attention_batch


def batch():
    config = {'arm': 'tuned_online', 'policy': {'trigger': 'fixed'}}
    return {'family': 'engineering', 'diagnostic_worker': 'attention',
            'attention_geometry': 'layer31-all', 'cases': ['a', 'b'], 'seeds': [1],
            'configs': [{**copy.deepcopy(config), 'id': 'off'},
                        {**copy.deepcopy(config), 'id': 'on', 'collect_signals': 'attention'}]}


def test_only_matched_injecting_engineering_pair():
    assert validate_attention_batch(batch())['layer'] == 31
    for key, value in [('family', 'dragin'), ('diagnostic_worker', 'multilayer'),
                       ('multilayer_policy', True), ('cases', ['a']), ('seeds', [1, 2])]:
        altered = batch()
        altered[key] = value
        with pytest.raises(ValueError):
            validate_attention_batch(altered)


@pytest.mark.parametrize('policy', [{'trigger': 'dragin'}, {'trigger': 'never'}, {'query': 'attention'}])
def test_signal_consuming_or_noninjecting_policies_rejected(policy):
    altered = batch()
    for config in altered['configs']:
        config['policy'] = policy
    with pytest.raises(ValueError):
        validate_attention_batch(altered)


def test_mismatched_settings_and_wrong_signal_schema_rejected():
    altered = batch()
    altered['configs'][1]['policy']['positions'] = [32]
    with pytest.raises(ValueError):
        validate_attention_batch(altered)
    altered = batch()
    altered['configs'][1]['collect_signals'] = 'hidden'
    with pytest.raises(ValueError):
        validate_attention_batch(altered)
    altered = batch()
    for config in altered['configs']:
        config['auxiliary_policy'] = {'method': 'seakr'}
    with pytest.raises(ValueError):
        validate_attention_batch(altered)
