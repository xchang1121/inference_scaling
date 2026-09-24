"""Candidate proposal and support checks shared by the AR importance samplers.

Every conditional-IS variant draws candidate blocks from the base policy with
the same seeds and request ids, requires full-support policies, and rescores
off-policy rollouts under the base policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
    TokenSequence,
)
from inference_scaling.shared.rng import SeedStream


def validate_base_sampling(sampling: SamplingConfig) -> None:
    if sampling.top_p < 1 or sampling.top_k is not None:
        raise ValueError(
            "base candidates must use a full-support autoregressive policy: "
            "top_p=1 and top_k=None"
        )


def validate_rollout_sampling(sampling: SamplingConfig) -> None:
    if sampling.top_p < 1 or sampling.top_k is not None:
        raise ValueError(
            "off-policy IS requires proposal support wherever the base weighted target is positive; "
            "hard top-k/top-p truncation is not accepted"
        )


def score_samples(
    base_backend: AutoregressiveBackend,
    prefixes: Sequence[TokenSequence],
    samples: Sequence[SequenceSample],
    base_sampling: SamplingConfig,
) -> list[float]:
    requests = [
        ScoreRequest(prefix, (sample.token_ids,), base_sampling)
        for prefix, sample in zip(prefixes, samples, strict=True)
    ]
    token_scores = base_backend.score_batch(requests)
    if len(token_scores) != len(samples):
        raise RuntimeError("backend returned an invalid number of base scores")
    totals: list[float] = []
    for sample, scores in zip(samples, token_scores, strict=True):
        if len(scores) != len(sample.token_ids):
            raise RuntimeError("backend returned an invalid base token score shape")
        total = float(sum(scores))
        if not isfinite(total):
            raise ValueError(
                "rollout proposal generated a completion outside base-model support"
            )
        totals.append(total)
    return totals


def sample_candidates(
    base_backend: AutoregressiveBackend,
    prefix: TokenSequence,
    count: int,
    block_length: int,
    sampling: SamplingConfig,
    seeds: SeedStream,
    step_index: int,
) -> list[SequenceSample]:
    requests = [
        GenerationRequest(
            prefix=prefix,
            max_new_tokens=block_length,
            sampling=sampling,
            seed=seeds.derive(
                "conditional_is", step_index, "candidate", candidate_index
            ),
            request_id=f"conditional-is:step:{step_index}:candidate:{candidate_index}",
        )
        for candidate_index in range(count)
    ]
    candidates = base_backend.sample_batch(requests)
    if len(candidates) != count:
        raise RuntimeError("backend returned an invalid number of candidates")
    for candidate in candidates:
        if not candidate.token_ids:
            raise RuntimeError("a candidate block must contain at least one token")
        if (
            candidate.model_id != base_backend.model_id
            or candidate.policy_id != sampling.policy_id
        ):
            raise RuntimeError(
                "candidate was not sampled and scored by the requested base policy"
            )
    return candidates


__all__ = [
    "sample_candidates",
    "score_samples",
    "validate_base_sampling",
    "validate_rollout_sampling",
]
