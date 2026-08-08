"""Suffix-resampling Metropolis--Hastings for sequence power targets.

For a fixed generated length ``L``, the target is proportional to
``p_base(x | prompt) ** alpha``.  A move retains a uniformly selected prefix,
regenerates the suffix from an autoregressive proposal, and uses the full
forward/reverse proposal correction.  The implementation caches per-token
base and proposal log-probabilities for the current state; this is an
algorithmic reproduction, not the later paged-KV runtime optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log

import numpy as np

from inference_scaling.config import MHConfig, SamplingConfig
from inference_scaling.rng import SeedStream
from inference_scaling.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    TokenSequence,
)


@dataclass(frozen=True, slots=True)
class MHStep:
    stage_length: int
    step: int
    cut: int
    proposed_suffix_length: int
    log_acceptance: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class MHChainResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    proposal_token_logprobs: tuple[float, ...]
    trace: tuple[MHStep, ...]
    chain_id: int

    @property
    def attempts(self) -> int:
        return len(self.trace)

    @property
    def accepted(self) -> int:
        return sum(step.accepted for step in self.trace)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0


def _stage_lengths(total_length: int, block_size: int) -> tuple[int, ...]:
    stages = list(range(block_size, total_length + 1, block_size))
    if not stages or stages[-1] != total_length:
        stages.append(total_length)
    return tuple(stages)


def _validate_proposal(sampling: SamplingConfig) -> None:
    if sampling.eos_token_id is not None:
        raise ValueError("fixed-length MH treats every position as a token; eos_token_id must be None")
    if sampling.top_k is not None or sampling.top_p < 1:
        raise ValueError(
            "hard top-k/top-p truncation normally violates MH's equal-support condition; "
            "use a full-support proposal"
        )


def _score_one(
    backend: AutoregressiveBackend,
    prefix: TokenSequence,
    continuation: TokenSequence,
    sampling: SamplingConfig | None,
) -> tuple[float, ...]:
    scored = backend.score_batch([ScoreRequest(prefix, (continuation,), sampling)])
    if len(scored) != 1 or len(scored[0]) != len(continuation):
        raise RuntimeError("backend returned an invalid token score shape")
    return scored[0]


def _sample_exact_length(
    backend: AutoregressiveBackend,
    *,
    prefix: TokenSequence,
    length: int,
    sampling: SamplingConfig,
    seed: int,
    request_id: str,
) -> tuple[TokenSequence, tuple[float, ...]]:
    sample = backend.sample_batch(
        [GenerationRequest(prefix, length, sampling, seed, request_id)]
    )[0]
    if len(sample.token_ids) != length:
        raise RuntimeError(
            f"MH requires a fixed-length suffix of {length} tokens, "
            f"but backend returned {len(sample.token_ids)}"
        )
    if sample.policy_id != sampling.policy_id:
        raise RuntimeError("backend did not score tokens under the requested proposal policy")
    if any(not isfinite(value) for value in sample.token_logprobs):
        raise RuntimeError("a sampled proposal token must have finite proposal log-probability")
    return sample.token_ids, sample.token_logprobs


def run_mh_chain(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: MHConfig,
    proposal: SamplingConfig,
    seeds: SeedStream,
    *,
    chain_id: int = 0,
) -> MHChainResult:
    """Run the staged MH algorithm from the article for one independent chain."""

    _validate_proposal(proposal)
    tokens: list[int] = []
    base_logs: list[float] = []
    proposal_logs: list[float] = []
    trace: list[MHStep] = []

    for stage_index, stage_length in enumerate(
        _stage_lengths(config.total_length, config.block_size)
    ):
        extension_length = stage_length - len(tokens)
        if extension_length > 0:
            extension_prefix = prompt + tuple(tokens)
            extension, extension_q = _sample_exact_length(
                backend,
                prefix=extension_prefix,
                length=extension_length,
                sampling=proposal,
                seed=seeds.derive("mh", chain_id, stage_index, "extend"),
                request_id=f"mh:{chain_id}:stage:{stage_index}:extend",
            )
            extension_p = _score_one(backend, extension_prefix, extension, None)
            if any(not isfinite(value) for value in extension_p):
                raise ValueError("proposal generated a sequence outside the base model support")
            tokens.extend(extension)
            base_logs.extend(extension_p)
            proposal_logs.extend(extension_q)

        for step_index in range(config.steps_per_block):
            cut_rng = seeds.generator("mh", chain_id, stage_index, step_index, "cut")
            cut = int(cut_rng.integers(0, stage_length))
            shared_prefix = prompt + tuple(tokens[:cut])
            suffix_length = stage_length - cut
            proposed_tokens, proposed_q = _sample_exact_length(
                backend,
                prefix=shared_prefix,
                length=suffix_length,
                sampling=proposal,
                seed=seeds.derive("mh", chain_id, stage_index, step_index, "proposal"),
                request_id=f"mh:{chain_id}:stage:{stage_index}:step:{step_index}",
            )
            proposed_p = _score_one(backend, shared_prefix, proposed_tokens, None)
            if any(not isfinite(value) for value in proposed_p):
                raise ValueError("proposal generated a sequence outside the base model support")

            old_p = float(sum(base_logs[cut:stage_length]))
            old_q = float(sum(proposal_logs[cut:stage_length]))
            new_p = float(sum(proposed_p))
            new_q = float(sum(proposed_q))
            log_acceptance = min(
                0.0,
                config.alpha * (new_p - old_p) + old_q - new_q,
            )
            accept_rng = seeds.generator("mh", chain_id, stage_index, step_index, "accept")
            uniform = max(float(accept_rng.random()), np.finfo(np.float64).tiny)
            accepted = log(uniform) <= log_acceptance
            if accepted:
                tokens[cut:stage_length] = proposed_tokens
                base_logs[cut:stage_length] = proposed_p
                proposal_logs[cut:stage_length] = proposed_q
            trace.append(
                MHStep(
                    stage_length=stage_length,
                    step=step_index,
                    cut=cut,
                    proposed_suffix_length=suffix_length,
                    log_acceptance=log_acceptance,
                    accepted=accepted,
                )
            )

    return MHChainResult(
        prompt=prompt,
        token_ids=tuple(tokens),
        base_token_logprobs=tuple(base_logs),
        proposal_token_logprobs=tuple(proposal_logs),
        trace=tuple(trace),
        chain_id=chain_id,
    )


def run_mh_chains(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: MHConfig,
    proposal: SamplingConfig,
    seeds: SeedStream,
) -> tuple[MHChainResult, ...]:
    """Run independent chains with order-independent random streams."""

    return tuple(
        run_mh_chain(backend, prompt, config, proposal, seeds, chain_id=chain_id)
        for chain_id in range(config.chains)
    )

