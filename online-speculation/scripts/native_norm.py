"""One-kernel XLLM grouped RMSNorm, preserving explicit BF16 round points.

Inference only. The differentiable last-layer replay lives in native_fast_weights.
"""
from functools import partial

import torch
import triton
import triton.language as tl


@triton.jit
def _norm(X, R, W, Y, S, H: tl.constexpr, G: tl.constexpr,
          EPS: tl.constexpr, HAS_R: tl.constexpr, BLOCK: tl.constexpr):
    group = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    offset = group * G + col
    valid = col < G
    x = tl.load(X + offset, valid, 0).to(tl.float32)
    if HAS_R:
        r = tl.load(R + offset, valid, 0).to(tl.float32)
        # Upstream adds in FP32, rounds to input dtype, THEN computes variance.
        x = (x + r).to(X.dtype.element_ty).to(tl.float32)
        tl.store(S + offset, x, valid)
    square = tl.where(valid, x * x, 0.)
    inv = tl.rsqrt(tl.sum(square, 0) / G + EPS)
    unit = (x * inv).to(X.dtype.element_ty).to(tl.float32)
    weight = tl.load(W + (group * G) % H + col, valid, 0).to(tl.float32)
    tl.store(Y + offset, unit * weight, valid)


def fused_grouped_rms(module, x, residual=None):
    if not x.is_cuda or not x.is_contiguous() or x.shape[-1] != module.hidden_size:
        raise ValueError("fused norm requires contiguous CUDA rows of hidden_size")
    if residual is not None and (residual.shape != x.shape or not residual.is_contiguous()):
        raise ValueError("residual shape/stride mismatch")
    if torch.is_grad_enabled() and (x.requires_grad or (residual is not None and residual.requires_grad)):
        raise RuntimeError("inference norm has no backward; use differentiable replay")
    y = torch.empty_like(x)
    summed = torch.empty_like(x) if residual is not None else y
    _norm[(x.numel() // module.group_size,)](
        x, residual if residual is not None else x, module.weight, y, summed,
        module.hidden_size, module.group_size, module.eps, residual is not None,
        triton.next_power_of_2(module.group_size), enable_fp_fusion=False,
    )
    return (y, summed) if residual is not None else y


def install_fused_norms(model):
    from nano_vllm_uno.models.xllm import XllmGroupedRMSNorm

    count = 0
    for module in model.modules():
        if isinstance(module, XllmGroupedRMSNorm):
            module.forward = partial(fused_grouped_rms, module)
            count += 1
    if count == 0:
        raise ValueError("this optimization supports XLLM only")
    return count
