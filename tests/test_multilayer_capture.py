from copy import deepcopy
import pytest
from inference_scaling.retrieval.multilayer_capture import validate_diagnostic_batch


def batch():
    return {'diagnostic_worker': 'multilayer', 'configs': [
        {'arm': 'tuned_online', 'policy': {'trigger': 'never'}},
        {'arm': 'tuned_online', 'policy': {'trigger': 'never'}, 'collect_signals': 'multilayer'}]}


def test_diagnostic_pair():
    value = batch()
    before = deepcopy(value)
    validate_diagnostic_batch(value)
    assert value == before


@pytest.mark.parametrize('key,value', [('collect_signals','hidden'), ('collect_signals','attention'),
    ('auxiliary_policy',{}), ('arm','closed_is'),
    ('policy',{'trigger':'seakr'}), ('policy',{'query':'attention'})])
def test_reject_incompatible_consumer(key, value):
    config = batch()
    config['configs'][1][key] = value
    with pytest.raises(ValueError):
        validate_diagnostic_batch(config)
