from __future__ import annotations

import pytest
import torch

from online_speculation.attention_partition import merge_attention_partitions


@pytest.mark.parametrize("cut", [0, 1, 7, 19])
def test_partition_identity_equals_dense_softmax_in_float64(cut):
    generator = torch.Generator().manual_seed(616)
    logits = torch.randn(2, 3, 19, dtype=torch.float64, generator=generator) * 10
    values = torch.randn(2, 3, 19, 7, dtype=torch.float64, generator=generator)

    def partition(start, stop):
        if start == stop:
            return torch.zeros(2, 3, 7, dtype=torch.float64), torch.full((2, 3), -torch.inf, dtype=torch.float64)
        scores = logits[..., start:stop]
        output = (scores.softmax(-1).unsqueeze(-1) * values[..., start:stop, :]).sum(-2)
        return output, scores.logsumexp(-1)

    a, la = partition(0, cut)
    b, lb = partition(cut, 19)
    expected = (logits.softmax(-1).unsqueeze(-1) * values).sum(-2)
    assert torch.allclose(merge_attention_partitions(a, la, b, lb), expected, atol=1e-12, rtol=1e-12)


def test_empty_total_visibility_is_rejected():
    output = torch.zeros(2, 4)
    lse = torch.full((2,), -torch.inf)
    with pytest.raises(ValueError):
        merge_attention_partitions(output, lse, output, lse)
