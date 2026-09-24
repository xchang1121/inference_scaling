"""Suffix-resampling Metropolis--Hastings for sequence power targets.

For a fixed generated length ``L``, the target is proportional to
``p_base(x | prompt) ** alpha``.  A move draws a suffix length from a configured
full-support schedule, retains the corresponding prefix, regenerates the suffix
from an autoregressive proposal, and uses the full forward/reverse proposal
correction.  The implementation caches per-token
base and proposal log-probabilities for the current state; this is an
algorithmic reproduction, not the later paged-KV runtime optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from inference_scaling.arllm.algorithms.config import MHConfig, RewardMHConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.shared.sampling.mh import decide_metropolis_hastings
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenReward
from inference_scaling.arllm.types import (
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
    suffix_schedule: str
    suffix_probability: float
    proposed_token_changes: int
    accepted_token_changes: int


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

    @property
    def mean_proposed_suffix_length(self) -> float:
        return (
            sum(step.proposed_suffix_length for step in self.trace) / self.attempts
            if self.attempts
            else 0.0
        )

    @property
    def mean_proposed_token_changes(self) -> float:
        return (
            sum(step.proposed_token_changes for step in self.trace) / self.attempts
            if self.attempts
            else 0.0
        )

    @property
    def mean_accepted_token_changes(self) -> float:
        return (
            sum(step.accepted_token_changes for step in self.trace) / self.attempts
            if self.attempts
            else 0.0
        )


@dataclass(frozen=True, slots=True)
class RewardMHStep:
    step: int
    cut: int
    proposed_suffix_length: int
    current_reward: float
    proposed_reward: float
    log_acceptance: float
    accepted: bool
    suffix_schedule: str
    suffix_probability: float
    proposed_token_changes: int
    accepted_token_changes: int


@dataclass(frozen=True, slots=True)
class RewardMHChainResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    reward: float
    base_token_logprobs: tuple[float, ...]
    proposal_token_logprobs: tuple[float, ...]
    trace: tuple[RewardMHStep, ...]
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

    @property
    def mean_proposed_suffix_length(self) -> float:
        return (
            sum(step.proposed_suffix_length for step in self.trace) / self.attempts
            if self.attempts
            else 0.0
        )

    @property
    def mean_proposed_token_changes(self) -> float:
        return (
            sum(step.proposed_token_changes for step in self.trace) / self.attempts
            if self.attempts
            else 0.0
        )

    @property
    def mean_accepted_token_changes(self) -> float:
        return (
            sum(step.accepted_token_changes for step in self.trace) / self.attempts
            if self.attempts
            else 0.0
        )


def suffix_length_probabilities(
    stage_length: int,
    schedule: str,
) -> tuple[float, ...]:
    """Return full-support probabilities for suffix lengths 1 through ``L``."""

    if stage_length <= 0:
        raise ValueError("stage_length must be positive")
    if schedule == "uniform":
        values = np.ones(stage_length, dtype=np.float64)
    elif schedule == "inverse_length":
        values = 1.0 / np.arange(1, stage_length + 1, dtype=np.float64)
    elif schedule == "multiscale":
        # Ten percent uniform mass keeps every suffix length reachable.  The
        # remaining mass is uniform over unique powers of two and the full
        # length, which supplies both local and global proposals.
        values = np.full(stage_length, 0.1 / stage_length, dtype=np.float64)
        favored = {1, stage_length}
        length = 1
        while length < stage_length:
            favored.add(length)
            length *= 2
        bonus = 0.9 / len(favored)
        for suffix_length in favored:
            values[suffix_length - 1] += bonus
    else:
        raise ValueError(f"unknown suffix schedule {schedule!r}")
    values /= values.sum()
    return tuple(float(value) for value in values)


def _draw_suffix(
    *,
    stage_length: int,
    schedule: str,
    rng: np.random.Generator,
) -> tuple[int, int, float]:
    probabilities = suffix_length_probabilities(stage_length, schedule)
    if schedule == "uniform":
        # Preserve the established baseline's random stream exactly.
        cut = int(rng.integers(0, stage_length))
        suffix_length = stage_length - cut
    else:
        suffix_length = int(
            rng.choice(
                np.arange(1, stage_length + 1, dtype=np.int64),
                p=probabilities,
            )
        )
        cut = stage_length - suffix_length
    return cut, suffix_length, probabilities[suffix_length - 1]


def _validate_proposal(sampling: SamplingConfig) -> None:
    if sampling.eos_token_id is not None:
        raise ValueError("fixed-length MH treats every position as a token; eos_token_id must be None")
    if sampling.top_k is not None or sampling.top_p < 1:
        raise ValueError(
            "hard top-k/top-p truncation normally violates MH's equal-support condition; "
            "use a full-support proposal"
        )


def _is_base_proposal(sampling: SamplingConfig) -> bool:
    return sampling.temperature == 1 and sampling.top_p == 1 and sampling.top_k is None


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
) -> tuple[TokenSequence, tuple[float, ...], tuple[float, ...] | None]:
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
    cached_base = None
    if sample.reference_policy_id == SamplingConfig().policy_id:
        cached_base = sample.reference_token_logprobs
    return sample.token_ids, sample.token_logprobs, cached_base


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
        config.stages
    ):
        extension_length = stage_length - len(tokens)
        if extension_length > 0:
            extension_prefix = prompt + tuple(tokens)
            extension, extension_q, extension_cached_p = _sample_exact_length(
                backend,
                prefix=extension_prefix,
                length=extension_length,
                sampling=proposal,
                seed=seeds.derive("mh", chain_id, stage_index, "extend"),
                request_id=f"mh:{chain_id}:stage:{stage_index}:extend",
            )
            extension_p = extension_cached_p or _score_one(
                backend, extension_prefix, extension, None
            )
            if any(not isfinite(value) for value in extension_p):
                raise ValueError("proposal generated a sequence outside the base model support")
            tokens.extend(extension)
            base_logs.extend(extension_p)
            proposal_logs.extend(extension_q)

        for step_index in range(config.stage_updates):
            cut_rng = seeds.generator("mh", chain_id, stage_index, step_index, "cut")
            cut, suffix_length, suffix_probability = _draw_suffix(
                stage_length=stage_length,
                schedule=config.suffix_schedule,
                rng=cut_rng,
            )
            shared_prefix = prompt + tuple(tokens[:cut])
            proposed_tokens, proposed_q, proposed_cached_p = _sample_exact_length(
                backend,
                prefix=shared_prefix,
                length=suffix_length,
                sampling=proposal,
                seed=seeds.derive("mh", chain_id, stage_index, step_index, "proposal"),
                request_id=f"mh:{chain_id}:stage:{stage_index}:step:{step_index}",
            )
            proposed_p = proposed_cached_p or _score_one(
                backend, shared_prefix, proposed_tokens, None
            )
            if any(not isfinite(value) for value in proposed_p):
                raise ValueError("proposal generated a sequence outside the base model support")

            old_p = float(sum(base_logs[cut:stage_length]))
            old_q = float(sum(proposal_logs[cut:stage_length]))
            new_p = float(sum(proposed_p))
            new_q = float(sum(proposed_q))
            accept_rng = seeds.generator("mh", chain_id, stage_index, step_index, "accept")
            decision = decide_metropolis_hastings(
                current_target_log_density=config.alpha * old_p,
                proposed_target_log_density=config.alpha * new_p,
                forward_proposal_log_probability=new_q,
                reverse_proposal_log_probability=old_q,
                uniform=float(accept_rng.random()),
            )
            log_acceptance = decision.log_acceptance
            accepted = decision.accepted
            proposed_token_changes = sum(
                old != new
                for old, new in zip(
                    tokens[cut:stage_length], proposed_tokens, strict=True
                )
            )
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
                    suffix_schedule=config.suffix_schedule,
                    suffix_probability=suffix_probability,
                    proposed_token_changes=proposed_token_changes,
                    accepted_token_changes=(proposed_token_changes if accepted else 0),
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


def run_reward_mh_chain(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: RewardMHConfig,
    proposal: SamplingConfig,
    reward: TokenReward,
    seeds: SeedStream,
    *,
    chain_id: int = 0,
) -> RewardMHChainResult:
    """Sample ``p_base(x) exp(reward(x) / temperature)`` with suffix MH.

    The chain is initialized at full length and every update draws one of all
    suffix starts uniformly.  For a base-model proposal the likelihood terms
    cancel, leaving only the reward difference; the expanded ratio below also
    remains correct for any full-support temperature proposal.
    """

    _validate_proposal(proposal)
    tokens, proposal_logs, cached_base_logs = _sample_exact_length(
        backend,
        prefix=prompt,
        length=config.total_length,
        sampling=proposal,
        seed=seeds.derive("reward_mh", chain_id, "initialize"),
        request_id=f"reward-mh:{chain_id}:initialize",
    )
    base_logs = (
        proposal_logs
        if _is_base_proposal(proposal)
        else cached_base_logs or _score_one(backend, prompt, tokens, None)
    )
    if any(not isfinite(value) for value in base_logs):
        raise ValueError("proposal generated a sequence outside the base model support")
    current_reward = float(reward(prompt, tokens))
    if not isfinite(current_reward):
        raise ValueError("reward must be finite")
    mutable_tokens = list(tokens)
    mutable_base_logs = list(base_logs)
    mutable_proposal_logs = list(proposal_logs)
    trace: list[RewardMHStep] = []

    for step_index in range(config.updates):
        cut, suffix_length, suffix_probability = _draw_suffix(
            stage_length=config.total_length,
            schedule=config.suffix_schedule,
            rng=seeds.generator("reward_mh", chain_id, step_index, "cut"),
        )
        shared_prefix = prompt + tuple(mutable_tokens[:cut])
        proposed_tokens, proposed_q, proposed_cached_p = _sample_exact_length(
            backend,
            prefix=shared_prefix,
            length=suffix_length,
            sampling=proposal,
            seed=seeds.derive("reward_mh", chain_id, step_index, "proposal"),
            request_id=f"reward-mh:{chain_id}:step:{step_index}",
        )
        proposed_p = (
            proposed_q
            if _is_base_proposal(proposal)
            else proposed_cached_p
            or _score_one(backend, shared_prefix, proposed_tokens, None)
        )
        if any(not isfinite(value) for value in proposed_p):
            raise ValueError("proposal generated a sequence outside the base model support")
        proposed_sequence = tuple(mutable_tokens[:cut]) + proposed_tokens
        proposed_reward = float(reward(prompt, proposed_sequence))
        if not isfinite(proposed_reward):
            raise ValueError("reward must be finite")

        old_p = float(sum(mutable_base_logs[cut:]))
        old_q = float(sum(mutable_proposal_logs[cut:]))
        new_p = float(sum(proposed_p))
        new_q = float(sum(proposed_q))
        decision = decide_metropolis_hastings(
            current_target_log_density=(
                old_p + current_reward / config.reward_temperature
            ),
            proposed_target_log_density=(
                new_p + proposed_reward / config.reward_temperature
            ),
            forward_proposal_log_probability=new_q,
            reverse_proposal_log_probability=old_q,
            uniform=float(
                seeds.generator("reward_mh", chain_id, step_index, "accept").random()
            ),
        )
        log_acceptance = decision.log_acceptance
        accepted = decision.accepted
        proposed_token_changes = sum(
            old != new
            for old, new in zip(mutable_tokens[cut:], proposed_tokens, strict=True)
        )
        previous_reward = current_reward
        if accepted:
            mutable_tokens[cut:] = proposed_tokens
            mutable_base_logs[cut:] = proposed_p
            mutable_proposal_logs[cut:] = proposed_q
            current_reward = proposed_reward
        trace.append(
            RewardMHStep(
                step=step_index,
                cut=cut,
                proposed_suffix_length=suffix_length,
                current_reward=previous_reward,
                proposed_reward=proposed_reward,
                log_acceptance=log_acceptance,
                accepted=accepted,
                suffix_schedule=config.suffix_schedule,
                suffix_probability=suffix_probability,
                proposed_token_changes=proposed_token_changes,
                accepted_token_changes=(proposed_token_changes if accepted else 0),
            )
        )
    return RewardMHChainResult(prompt, tuple(mutable_tokens), current_reward,
                               tuple(mutable_base_logs), tuple(mutable_proposal_logs), tuple(trace), chain_id)
