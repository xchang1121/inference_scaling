import copy

import pytest

from calibrate_auxiliary_multilayer import validate_four_pool
from test_auxiliary_multilayer import calls


def four_calls():
    values = calls() + calls()
    for index, call in enumerate(values):
        call['request']['seed'] = index
        call['token_ids'] = [index + 1, index + 2]
        for row in call['internal_signals']:
            row['hidden_projection'][index] += .1 * index
    return values


def test_four_real_paths_have_separate_layer_scores_and_preserve_inputs():
    values = four_calls()
    saved = copy.deepcopy(values)
    metrics = validate_four_pool(values)
    assert values == saved
    assert set(metrics) == {'16', '32', '48', 'final'}
    assert all(m['samples'] == m['distinct_paths'] == 4 and m['feature_dimensions'] == 128
               for tails in metrics.values() for m in tails.values())


@pytest.mark.parametrize('defect', ['two_samples', 'repeated_seed', 'different_prefix', 'alignment', 'nan', 'copied_layer'])
def test_four_pool_rejects_invalid_samples(defect):
    values = four_calls()
    if defect == 'two_samples':
        values = values[:2]
    elif defect == 'repeated_seed':
        values[3]['request']['seed'] = values[0]['request']['seed']
    elif defect == 'different_prefix':
        values[3]['prefix_sha256'] = 'other'
    elif defect == 'alignment':
        values[3]['internal_signals'][0]['hidden_projection'][512] = .01
    elif defect == 'nan':
        values[3]['internal_signals'][0]['hidden_projection'][17] = float('nan')
    elif defect == 'copied_layer':
        for row in values[3]['internal_signals']:
            row['hidden_projection'][:128] = row['hidden_projection'][384:512]
    with pytest.raises(AssertionError):
        validate_four_pool(values)
