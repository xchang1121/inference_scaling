"""Opt-in scalar statistics transported with native asynchronous logprob rows.

The private wire format is ``[sampled logprob, KL(U||p), top-K logprobs]``.
The KL slot has token ID ``-1``, which is never a generated token. The backend
requests one additional slot and removes it before interpreting probabilities.
Native request slicing, DELTA aggregation and async device copies remain in
charge of alignment. Only the small row, never the full vocabulary, leaves
the accelerator.

Platform-agnostic: works identically on Ascend NPU (``AscendLogprobWorker``)
and NVIDIA CUDA GPU (``GPULogprobWorker``). When top-k or top-p truncation is
active, the pre-mask logits (after ``logit_bias`` and temperature scaling,
right before hard masking) are reduced in O(1) extra memory via
``logsumexp(z) - mean(z) - log(V)``, keeping full-vocabulary Self-Certainty
and top-K Consilience finite and exact under any global sampling policy.
"""

from __future__ import annotations

from math import fsum, isfinite, log


SCALAR_STATISTICS_KEY = "inference_scaling_scalar_self_certainty"
SELF_CERTAINTY_SLOT = -1


def install_scalar_statistics(sampler, vocabulary_size: int) -> None:
    """Reduce the full-vocabulary distribution on device; leave native sampling intact."""
    import torch

    if getattr(sampler, "_scalar_self_certainty_installed", False):
        return
    original_gather = sampler.gather_logprobs
    sampler._pending_pre_mask = None
    sampler._active_pre_mask_topk = 0

    original_forward = getattr(sampler, "forward", None)
    topk_topp = getattr(sampler, "topk_topp_sampler", None)
    original_topk_topp = getattr(topk_topp, "forward", None) if topk_topp is not None else None

    if callable(original_forward):
        def tracked_forward(logits, sampling_metadata, *args, **kwargs):
            max_lp = getattr(sampling_metadata, "max_num_logprobs", None)
            sampler._active_pre_mask_topk = max(0, int(max_lp) - 1) if max_lp is not None and max_lp > 1 else 0
            sampler._pending_pre_mask = None
            try:
                return original_forward(logits, sampling_metadata, *args, **kwargs)
            finally:
                sampler._pending_pre_mask = None
                sampler._active_pre_mask_topk = 0

        sampler.forward = tracked_forward

    if topk_topp is not None and callable(original_topk_topp):
        def pre_mask_topk_topp(logits, generators, k, p):
            # Only capture pre-mask statistics when hard truncation (top-k / top-p) may mask entries to -inf.
            if (k is not None or p is not None) and logits.shape[-1] == vocabulary_size:
                f = logits.float()
                lse = torch.logsumexp(f, dim=-1, keepdim=True)
                certainty = (lse - f.mean(dim=-1, keepdim=True) - log(vocabulary_size)).clamp_min(0.0)
                pre_k = getattr(sampler, "_active_pre_mask_topk", 0)
                if pre_k > 0:
                    top_vals, top_ids = f.topk(min(pre_k, vocabulary_size), dim=-1)
                    top_lps = top_vals - lse
                else:
                    top_ids = top_lps = None
                sampler._pending_pre_mask = (certainty, top_ids, top_lps)
            else:
                sampler._pending_pre_mask = None
            return original_topk_topp(logits, generators, k, p)

        topk_topp.forward = pre_mask_topk_topp

    def gather(logprobs, num_logprobs, token_ids):
        pending = getattr(sampler, "_pending_pre_mask", None)
        sampler._pending_pre_mask = None
        if num_logprobs == 0:
            return original_gather(logprobs, num_logprobs, token_ids)
        if num_logprobs < 1 or logprobs.shape[-1] != vocabulary_size:
            raise RuntimeError("scalar Self-Certainty requires the complete, unpadded vocabulary")
        if pending is not None and pending[0].shape[0] == logprobs.shape[0]:
            certainty, top_ids, top_lps = pending
            k_needed = num_logprobs - 1
            if k_needed > 0 and top_ids is not None and top_ids.shape[-1] >= k_needed:
                base = original_gather(logprobs, 0, token_ids)
                ids0, vals0 = base.logprob_token_ids[:, :1], base.logprobs[:, :1]
                slot = torch.full_like(ids0, SELF_CERTAINTY_SLOT)
                return base._replace(
                    logprob_token_ids=torch.cat((ids0, slot, top_ids[:, :k_needed].to(dtype=ids0.dtype)), dim=-1),
                    logprobs=torch.cat((vals0, certainty.to(dtype=vals0.dtype), top_lps[:, :k_needed].to(dtype=vals0.dtype)), dim=-1),
                )
            result = original_gather(logprobs, k_needed, token_ids)
            ids, values = result.logprob_token_ids, result.logprobs
            return result._replace(
                logprob_token_ids=torch.cat((ids[:, :1], torch.full_like(ids[:, :1], SELF_CERTAINTY_SLOT), ids[:, 1:]), dim=-1),
                logprobs=torch.cat((values[:, :1], certainty.to(dtype=values.dtype), values[:, 1:]), dim=-1),
            )
        result = original_gather(logprobs, num_logprobs - 1, token_ids)
        certainty = -logprobs.float().mean(dim=-1, keepdim=True) - log(vocabulary_size)
        ids, values = result.logprob_token_ids, result.logprobs
        return result._replace(
            logprob_token_ids=torch.cat((ids[:, :1], torch.full_like(ids[:, :1], SELF_CERTAINTY_SLOT), ids[:, 1:]), dim=-1),
            logprobs=torch.cat((values[:, :1], certainty.to(dtype=values.dtype), values[:, 1:]), dim=-1),
        )

    sampler.gather_logprobs = gather
    sampler._scalar_self_certainty_installed = True


def scalar_generated_logprobs(positions, tokens, top_k):
    """Decode the private slot, failing closed on an incompatible worker output."""
    if positions is None or len(positions) != len(tokens):
        raise RuntimeError("invalid scalar-statistic output shape")
    flat = hasattr(positions, "start_indices")
    logs, statistics = [], []
    for index, token in enumerate(tokens):
        if flat:
            start, end = positions.start_indices[index], positions.end_indices[index]
            row_ids = positions.token_ids[start:end]
            row_lps = positions.logprobs[start:end]
            if not row_ids:
                raise RuntimeError("scalar Self-Certainty worker omitted its tagged statistic or sampled token")
            # Native flat wire layout: [sampled_token, SELF_CERTAINTY_SLOT, *top_k_tokens].
            if len(row_ids) >= 2 and row_ids[0] == token and row_ids[1] == SELF_CERTAINTY_SLOT:
                sampled_lp = float(row_lps[0])
                certainty = float(row_lps[1])
                extra_ids = row_ids[2:]
                extra_lps = row_lps[2:]
                if not isfinite(certainty) or certainty < -1e-5:
                    raise RuntimeError("Self-Certainty needs finite full-vocabulary probabilities (no hard masks)")
                if not isfinite(sampled_lp) or any(key < 0 for key in extra_ids) or not all(isfinite(float(v)) for v in extra_lps):
                    raise RuntimeError("invalid scalar-statistic probability payload")
                logs.append(sampled_lp)
                if top_k is None:
                    statistics.append(max(0.0, certainty))
                else:
                    if len(extra_lps) == top_k:
                        statistics.append(-fsum(float(v) for v in extra_lps) / top_k)
                    else:
                        distinct = dict(zip(extra_ids, (float(v) for v in extra_lps), strict=True))
                        distinct.setdefault(token, sampled_lp)
                        if len(distinct) < top_k:
                            raise RuntimeError("scalar-statistic payload omitted required top-K probabilities")
                        statistics.append(-fsum(sorted(distinct.values(), reverse=True)[:top_k]) / top_k)
                continue
            values = dict(zip(row_ids, row_lps, strict=True))
        else:
            values = {key: float(getattr(value, "logprob", value)) for key, value in positions[index].items()}
        if token not in values or SELF_CERTAINTY_SLOT not in values:
            raise RuntimeError("scalar Self-Certainty worker omitted its tagged statistic or sampled token")
        certainty = float(values.pop(SELF_CERTAINTY_SLOT))
        if not isfinite(certainty) or certainty < -1e-5:
            raise RuntimeError("Self-Certainty needs finite full-vocabulary probabilities (no hard masks)")
        if any(key < 0 for key in values) or not all(isfinite(value) for value in values.values()):
            raise RuntimeError("invalid scalar-statistic probability payload")
        logs.append(float(values[token]))
        if top_k is None:
            statistics.append(max(0.0, certainty))
        else:
            if len(values) < top_k:
                raise RuntimeError("scalar-statistic payload omitted required top-K probabilities")
            statistics.append(-fsum(sorted(values.values(), reverse=True)[:top_k]) / top_k)
    return tuple(logs), tuple(statistics)


__all__ = ["SCALAR_STATISTICS_KEY", "SELF_CERTAINTY_SLOT", "install_scalar_statistics", "scalar_generated_logprobs"]
