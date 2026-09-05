"""CPU/GPU mathematical reference for partitioned softmax attention merging."""

from __future__ import annotations

import torch
from torch import Tensor


def merge_attention_partitions(out_a: Tensor, lse_a: Tensor, out_b: Tensor, lse_b: Tensor) -> Tensor:
    """Merge disjoint normalized partitions; empty partitions must have zero output.

    LSE shapes equal output shapes without the final value dimension. At least
    one partition per query must be nonempty. This is not a production kernel.
    """
    if out_a.shape != out_b.shape or lse_a.shape != out_a.shape[:-1] or lse_b.shape != lse_a.shape:
        raise ValueError("attention partition shapes differ")
    total = torch.logaddexp(lse_a, lse_b)
    if not torch.isfinite(total).all():
        raise ValueError("combined attention partition must have a finite normalizer")
    return (lse_a - total).exp().unsqueeze(-1) * out_a + (lse_b - total).exp().unsqueeze(-1) * out_b
