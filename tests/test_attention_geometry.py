import math
from collections import namedtuple
from types import SimpleNamespace

import pytest

from inference_scaling.arllm.backends.vllm_statistics import install_scalar_statistics
from inference_scaling.retrieval.attention_geometry import decode_geometry, geometry, head_indices
from inference_scaling.retrieval.internal_signals import install_distribution_signals, unpack_signals


def tagged_row(layer=31, code=0, count=4, selected=4):
    row = {1: -.3, -1: 5., -2: 1., -3: .4, -4: .2, -5: .1}
    row.update({-2000-i: v for i, v in enumerate([0, 2, .6, .2, 3, .001, 0., layer, code, count, selected])})
    return row


def test_geometry_tags_bind_real_layer_heads_and_leave_legacy_unchanged():
    row = tagged_row()
    clean, signals = unpack_signals([row], [1], 7, attention_topk=2, attention_geometry='layer31-all')
    assert clean == [{1: -.3, -1: 5.}]
    assert signals[0]['attention']['layer'] == 31
    assert signals[0]['attention']['geometry']['selected_heads'] == [0, 1, 2, 3]
    legacy = {k: v for k, v in row.items() if k >= -2006}
    _, old = unpack_signals([legacy], [1], 7, attention_topk=2)
    assert old[0]['attention']['layer'] == 63 and 'geometry' not in old[0]['attention']
    with pytest.raises(RuntimeError, match='unknown'):
        unpack_signals([row], [1], 7, attention_topk=2)
    with pytest.raises(KeyError):
        unpack_signals([legacy], [1], 7, attention_topk=2, attention_geometry='layer31-all')


@pytest.mark.parametrize('index,value', [(7, 63), (8, 1), (9, 0), (10, 3), (7, 31.5), (9, math.nan)])
def test_reject_mutated_geometry_metadata(index, value):
    row = tagged_row()
    row[-2000-index] = value
    with pytest.raises((ValueError, RuntimeError)):
        unpack_signals([row], [1], 7, attention_topk=2, attention_geometry='layer31-all')


def test_head_geometry_and_invalid_mixed_inputs():
    assert head_indices(5, 'even') == (0, 2, 4)
    assert head_indices(5, 'odd') == (1, 3)
    assert decode_geometry([63, 1, 5, 3], 'layer63-even')['selected_heads'] == [0, 2, 4]
    with pytest.raises(ValueError):
        geometry('layer32-all')
    with pytest.raises(ValueError):
        unpack_signals([], [], 7, attention_geometry='layer31-all')
    with pytest.raises(ValueError):
        unpack_signals([], [], 7, hidden_dimensions=128, attention_topk=2, attention_geometry='layer31-all')


def check_attention_geometry_truth():
    import torch
    from inference_scaling.retrieval.attention_signals import reconstruct_attention
    q = torch.tensor([[1., 0.], [0., 1.], [2., -1.], [-1., 2.]])
    k = torch.tensor([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]], [[1., 1.], [-1., 1.]]])
    v = torch.tensor([[[2., 1.], [1., 2.]], [[3., 4.], [4., 3.]], [[5., 6.], [6., 5.]]])
    probabilities = torch.stack([(q[h] @ k[:, h//2].T / math.sqrt(2)).softmax(-1) for h in range(4)])
    output = torch.stack([probabilities[h] @ v[:, h//2] for h in range(4)])
    saved = [x.clone() for x in (q, k, v, output)]
    for mode in ('all', 'even', 'odd'):
        result = reconstruct_attention(q, k, v, output, topk=3, head_pooling=mode)
        expected = probabilities[list(head_indices(4, mode))].mean(0)
        assert torch.allclose(result[3:6], expected[result[:3].long()], atol=1e-6)
        assert result[-2] < 1e-6 and result[-1] < 1e-6
    assert all(torch.equal(a, b) for a, b in zip((q, k, v, output), saved, strict=True))
    # An unselected odd head still participates in the P@V reconstruction validation.
    broken = output.clone()
    broken[1] += 100
    assert reconstruct_attention(q, k, v, broken, topk=3, head_pooling='even')[-2] > .03
    # Legacy all-head operation remains exactly the original reduction/topk expression.
    legacy = reconstruct_attention(q, k, v, output, topk=3)
    grouped = q.reshape(2, 2, 2)
    weights = (torch.einsum('hgd,lhd->hgl', grouped, k)/math.sqrt(2)).softmax(-1)
    scores, ids = weights.mean(dim=(0, 1)).topk(3)
    assert torch.equal(legacy[:3], ids.float()) and torch.equal(legacy[3:6], scores)

    payload = namedtuple('Payload', 'logprob_token_ids logprobs selected_token_ranks')
    def gather(lp, n, tokens):
        values, ids = lp.topk(n, dim=-1)
        return payload(torch.cat((tokens[:, None], ids), dim=-1),
                       torch.cat((lp.gather(-1, tokens[:, None]), values), dim=-1), None)
    sampler = SimpleNamespace(gather_logprobs=gather)
    install_scalar_statistics(sampler, 7)
    lp = torch.tensor([[3., 2., 1., 0., -1., -2., -3.]]).log_softmax(-1)
    chosen = torch.tensor([0])
    baseline = sampler.gather_logprobs(lp, 6, chosen)
    install_distribution_signals(sampler, 7, 6, 5, attention_topk=3, attention_geometry='layer63-even')
    attention = reconstruct_attention(q, k, v, output, topk=3, head_pooling='even')
    sampler._retrieval_attention = torch.cat((attention, attention.new_tensor([63, 1, 4, 2])))[None]
    result = sampler.gather_logprobs(lp, 23, chosen)
    rows = [dict(zip(ids.tolist(), values.tolist(), strict=True))
            for ids, values in zip(result.logprob_token_ids, result.logprobs, strict=True)]
    clean, signals = unpack_signals(rows, [0], 7, attention_topk=3, attention_geometry='layer63-even')
    expected = [dict(zip(ids.tolist(), values.tolist(), strict=True))
                for ids, values in zip(baseline.logprob_token_ids, baseline.logprobs, strict=True)]
    assert clean == expected and signals[0]['attention']['geometry']['selected_heads'] == [0, 2]


def test_tensor_geometry_truth():
    pytest.importorskip('torch')
    check_attention_geometry_truth()
