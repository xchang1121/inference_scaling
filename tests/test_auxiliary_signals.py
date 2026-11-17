import copy

import pytest

from retrieval_fixtures import Native
from inference_scaling.retrieval.auxiliary_signals import AuxiliarySampler, hidden_disagreement, draft_confidence


def test_paid_auxiliary_samples_are_prefix_scoped_cached_and_budgeted():
    backend = Native(stop=600)
    aux = AuxiliarySampler(backend, backend, 7, (901,), context_tokens=1000, work_tokens=24)
    calls = aux.sample((1,2,105,30), (105,), purpose='lookahead', count=2, tokens=4)
    assert aux.work == 8 and len(backend.requests) == 2
    assert calls[0]['request']['seed'] != calls[1]['request']['seed']
    assert all(c['prefix_ids'] == [1,2,105,30] for c in calls)
    assert aux.sample((1,2,105,30), (105,), purpose='lookahead', count=2, tokens=4) == calls
    assert aux.work == 8 and len(backend.requests) == 2
    # The same logical prefix with different evidence is a distinct intervention.
    other = aux.sample((1,2,105,31), (105,), purpose='lookahead', count=2, tokens=4)
    assert other[0]['request']['seed'] != calls[0]['request']['seed']
    assert aux.sample((1,2,106,30), (106,), purpose='lookahead', count=3, tokens=4) is None
    assert aux.work == 16 and len(backend.requests) == 4


def examples():
    return [{'prefix_sha256':'same', 'request': {'seed': i}, 'token_ids':[i+1],
             'internal_signals':[{'hidden_projection':v}]} for i,v in enumerate([[1.,0.],[0.,1.]])]


@pytest.mark.parametrize('metric', ['cosine','variance','logvolume'])
def test_hidden_disagreement_requires_real_paths_and_independent_samples(metric):
    calls = examples()
    assert hidden_disagreement(calls, metric=metric)['value'] > 0
    calls[1]['token_ids'] = calls[0]['token_ids']
    assert hidden_disagreement(calls, metric=metric)['value'] == 0
    calls[1]['request']['seed'] = 0
    with pytest.raises(ValueError, match='independent'):
        hidden_disagreement(calls, metric=metric)


def test_hidden_uncertainty_rejects_mixed_contexts_and_missing_states():
    calls = examples()
    broken = copy.deepcopy(calls)
    broken[1]['prefix_sha256'] = 'different-evidence'
    with pytest.raises(ValueError, match='physical prefix'):
        hidden_disagreement(broken)
    calls[0]['internal_signals'] = None
    with pytest.raises(ValueError, match='aligned hidden'):
        hidden_disagreement(calls)


def test_flare_query_masks_low_confidence_draft_tokens():
    value = draft_confidence({'token_ids':[100,101,102], 'lp':[-.1,-5.,-.2]}, Native())
    assert value['low_fraction'] == pytest.approx(1/3)
    assert value['retained_token_ids'] == [100,102]
    assert value['masked_query'] == '100 102'
