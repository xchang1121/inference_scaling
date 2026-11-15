from copy import deepcopy
import pytest
from inference_scaling.retrieval.multilayer_capture import validate_diagnostic_batch, recapture_batches
from inference_scaling.retrieval.stepwise import token_hash


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
    ('probe_policy',{}), ('alignment_policy',{}), ('auxiliary_policy',{}), ('arm','closed_is'),
    ('policy',{'trigger':'seakr'}), ('policy',{'query':'attention'})])
def test_reject_incompatible_consumer(key, value):
    config = batch()
    config['configs'][1][key] = value
    with pytest.raises(ValueError):
        validate_diagnostic_batch(config)


def recapture_plan():
    return {'kind':'multilayer_prefix_recapture_pool', 'version':3, 'pool':32, 'batch_size':4,
            'repeats':2, 'labels_added':0, 'independent_groups_added':0,
            'items':[{'question_id':'TRAIN_Q010', 'prefix_ids':[1,n],
                      'prefix_sha256':token_hash([1,n])} for n in range(64)]}


def test_recapture_repeat_order_and_cost_identity():
    plan = recapture_plan()
    batches = recapture_batches(plan)
    assert len(batches) == 32 and len({(r,b) for r,b,_ in batches}) == 32
    for repeat in (0,1):
        flat = [item for r,_,items in batches if r == repeat for item in items]
        assert flat == (plan['items'] if repeat == 0 else plan['items'][::-1])
    assert all(len(items) == 4 for _,_,items in batches)


@pytest.mark.parametrize('field,value', [('repeats',3), ('batch_size',64), ('pool',128),
                                       ('labels_added',1), ('version',4)])
def test_recapture_refuses_unbounded_or_new_labels(field, value):
    plan = recapture_plan()
    plan[field] = value
    with pytest.raises(ValueError):
        recapture_batches(plan)


def test_recapture_refuses_duplicate_or_corrupt_prefix():
    plan = recapture_plan()
    plan['items'][1] = deepcopy(plan['items'][0])
    with pytest.raises(ValueError):
        recapture_batches(plan)
    plan = recapture_plan()
    plan['items'][0]['prefix_ids'].append(99)
    with pytest.raises(ValueError):
        recapture_batches(plan)
