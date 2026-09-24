"""Candidate proposal and support check shared by the AR importance samplers.

Fixed and budgeted conditional IS draw candidate blocks from the full-support
base policy with the same seeds and request ids.
"""

from __future__ import annotations

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, GenerationRequest, SequenceSample, TokenSequence
from inference_scaling.shared.rng import SeedStream


def validate_base_sampling(sampling: SamplingConfig) -> None:
    if sampling.top_p < 1 or sampling.top_k is not None:
        raise ValueError(
            "base candidates must use a full-support autoregressive policy: "
            "top_p=1 and top_k=None"
        )


def sample_candidates(
    base_backend: AutoregressiveBackend,
    prefix: TokenSequence,
    count: int,
    block_length: int,
    sampling: SamplingConfig,
    seeds: SeedStream,
    step_index: int,
    first_index: int = 0,
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
        for candidate_index in range(first_index, first_index + count)
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


__all__ = ["sample_candidates", "validate_base_sampling"]
