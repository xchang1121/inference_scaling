"""Candidate proposal and support check shared by the AR importance samplers.

A fresh candidate is drawn as a complete output and cut at the block boundary:
the block is the candidate and the rest is its first completion, a draw from
the base policy given the block. Fixed and budgeted conditional IS use the
same seeds and request ids. Drawn block-first, the candidate is the block alone
and its first completion continues the same request stream later.
"""

from __future__ import annotations

from dataclasses import replace

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, GenerationRequest, SequenceSample, TokenSequence
from inference_scaling.shared.rng import SeedStream

# A generated completion: its tokens and their base-policy log-probabilities.
Completion = tuple[TokenSequence, tuple[float, ...]]


class OwnStream:
    """Marks a block-first candidate: its first completion continues the candidate's own request.

    The completion is requested with the candidate's seed and ``uniform_offset`` at
    the block's length, so a backend with position-indexed uniform streams returns
    exactly the rest of the complete output the candidate would have been cut from.
    """


OWN_STREAM = OwnStream()


def validate_base_sampling(sampling: SamplingConfig) -> None:
    if sampling.top_p < 1 or sampling.top_k is not None:
        raise ValueError(
            "base candidates must use a full-support autoregressive policy: "
            "top_p=1 and top_k=None"
        )


def sample_outputs(
    base_backend: AutoregressiveBackend,
    prefix: TokenSequence,
    count: int,
    horizon: int,
    sampling: SamplingConfig,
    seeds: SeedStream,
    step_index: int,
    first_index: int = 0,
) -> list[SequenceSample]:
    """Complete outputs of up to ``horizon`` tokens after ``prefix``, one per candidate."""

    requests = [
        GenerationRequest(
            prefix=prefix,
            max_new_tokens=horizon,
            sampling=sampling,
            seed=seeds.derive("conditional_is", step_index, "candidate", candidate_index),
            request_id=f"conditional-is:step:{step_index}:candidate:{candidate_index}",
        )
        for candidate_index in range(first_index, first_index + count)
    ]
    outputs = base_backend.sample_batch(requests)
    if len(outputs) != count:
        raise RuntimeError("backend returned an invalid number of candidates")
    for output in outputs:
        if not output.token_ids:
            raise RuntimeError("a candidate block must contain at least one token")
        if output.model_id != base_backend.model_id or output.policy_id != sampling.policy_id:
            raise RuntimeError("candidate was not sampled and scored by the requested base policy")
    return outputs


def cut_block(output: SequenceSample, length: int) -> tuple[SequenceSample, Completion | None]:
    """A complete output as a candidate block and, when it continues, the block's first completion."""

    if len(output.token_ids) <= length:
        return output, None
    reference, bounds = output.reference_token_logprobs, output.token_cdf_bounds
    block = replace(output, token_ids=output.token_ids[:length], token_logprobs=output.token_logprobs[:length],
                    reference_token_logprobs=None if reference is None else reference[:length],
                    token_cdf_bounds=None if bounds is None else bounds[:length], finish_reason="length")
    return block, (output.token_ids[length:], output.token_logprobs[length:])


__all__ = ["Completion", "OWN_STREAM", "OwnStream", "cut_block", "sample_outputs", "validate_base_sampling"]
