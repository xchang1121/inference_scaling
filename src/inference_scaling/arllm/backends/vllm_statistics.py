"""Opt-in scalar statistics transported with native asynchronous logprob rows.

The private wire format is [sampled logprob, KL(U||p), top-K logprobs].
The KL slot has token ID -1, which is never a generated token. The backend
requests one additional slot and removes it before interpreting probabilities.
Native request slicing, DELTA aggregation and async device copies remain in
charge of alignment. Only the small row, never the full vocabulary, leaves
the accelerator. This protocol is restricted to our Ascend worker/backend.
"""

from __future__ import annotations

from math import fsum, isfinite, log


SCALAR_STATISTICS_KEY = "inference_scaling_scalar_self_certainty"
SELF_CERTAINTY_SLOT = -1


def install_scalar_statistics(sampler, vocabulary_size: int) -> None:
    """Reduce the processed full-support distribution; leave native sampling intact."""
    import torch

    if getattr(sampler, "_scalar_self_certainty_installed", False):
        return
    original = sampler.gather_logprobs

    def gather(logprobs, num_logprobs, token_ids):
        if num_logprobs == 0:
            return original(logprobs, num_logprobs, token_ids)
        if num_logprobs < 1 or logprobs.shape[-1] != vocabulary_size:
            raise RuntimeError("scalar Self-Certainty requires the complete, unpadded vocabulary")
        result = original(logprobs, num_logprobs - 1, token_ids)
        certainty = -logprobs.float().mean(dim=-1, keepdim=True) - log(vocabulary_size)
        ids, values = result.logprob_token_ids, result.logprobs
        return result._replace(
            logprob_token_ids=torch.cat((ids[:, :1], torch.full_like(ids[:, :1], SELF_CERTAINTY_SLOT), ids[:, 1:]), dim=-1),
            logprobs=torch.cat((values[:, :1], certainty, values[:, 1:]), dim=-1),
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
            values = dict(zip(positions.token_ids[start:end], positions.logprobs[start:end], strict=True))
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
