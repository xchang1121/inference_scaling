import copy

import pytest

from check_auxiliary_multilayer import validate_pool


def calls():
    return [{'prefix_sha256': 'same', 'request': {'seed': seed}, 'token_ids': [seed+1, seed+2],
             'internal_signals': [{'hidden_projection': [
                 (1+block)*(1+position)*float((i+seed+block) % 7+1) for block in range(4) for i in range(128)]
                 + [0.0]} for position in range(2)]} for seed in range(2)]


def test_auxiliary_capture_scores_explicit_real_blocks_without_mutating_calls():
    original = calls()
    saved = copy.deepcopy(original)
    results = validate_pool(original)
    assert original == saved
    assert set(results) == {'16', '32', '48', 'final'}
    assert all(v['feature_dimensions'] == 128 and v['samples'] == 2 and v['distinct_paths'] == 2
               for group in results.values() for v in group.values())
    assert len({group['1']['value'] for group in results.values()}) > 1


@pytest.mark.parametrize('defect', ['alignment', 'dimension', 'nan', 'copied_layer', 'static_rows', 'seed', 'prefix', 'missing'])
def test_auxiliary_capture_rejects_invalid_native_identity_and_geometry(defect):
    values = calls()
    vector = values[0]['internal_signals'][0]['hidden_projection']
    if defect == 'alignment':
        vector[-1] = .01
    elif defect == 'dimension':
        vector.pop()
    elif defect == 'nan':
        vector[17] = float('nan')
    elif defect == 'copied_layer':
        for signal in values[0]['internal_signals']:
            signal['hidden_projection'][:128] = signal['hidden_projection'][384:512]
    elif defect == 'static_rows':
        values[0]['internal_signals'][1] = copy.deepcopy(values[0]['internal_signals'][0])
    elif defect == 'seed':
        values[1]['request']['seed'] = values[0]['request']['seed']
    elif defect == 'prefix':
        values[1]['prefix_sha256'] = 'other'
    elif defect == 'missing':
        values[0]['internal_signals'].pop()
    with pytest.raises(AssertionError):
        validate_pool(values)
