"""Suffix-resampling Metropolis--Hastings over complete outputs.

A state is one complete output: generation stops at EOS (or at a scope
boundary) or at the stage's length limit ``T``. A move draws a cut position
from a full-support distribution over the ``T`` positions of the stage, keeps
the prefix before it, regenerates the suffix from an autoregressive proposal
until it stops or reaches ``T``, and applies the full forward/reverse proposal
correction. The cut distribution does not depend on the state, so it cancels
in the Hastings ratio. A cut at or after the end of a stopped output would
regenerate an empty suffix; that move leaves the state unchanged and is
skipped without a model call. Per-token base and proposal log-probabilities of
the current state are cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from inference_scaling.arllm.algorithms.config import PowerMHConfig, RewardMHConfig
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
class PowerMHStep:
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
class PowerMHChainResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    proposal_token_logprobs: tuple[float, ...]
    trace: tuple[PowerMHStep, ...]
    chain_id: int
    # Cuts past the end of a stopped output: no proposal, state unchanged.
    skipped: int = 0

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
    # Cuts past the end of a stopped output: no proposal, state unchanged.
    skipped: int = 0

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


@dataclass(frozen=True, slots=True)
class _Suffix:
    token_ids: TokenSequence
    proposal_logprobs: tuple[float, ...]
    base_logprobs: tuple[float, ...]
    # Ended at EOS or a scope boundary rather than at the length limit.
    stopped: bool


def _sample_suffix(
    backend: AutoregressiveBackend,
    *,
    prefix: TokenSequence,
    length: int,
    sampling: SamplingConfig,
    seed: int,
    request_id: str,
) -> _Suffix:
    """Generate up to ``length`` tokens with their proposal and base log-probabilities."""

    sample = backend.sample_batch([GenerationRequest(prefix, length, sampling, seed, request_id)])[0]
    stopped = sample.finish_reason != "length"
    if not 0 < len(sample.token_ids) <= length or (len(sample.token_ids) < length and not stopped):
        raise RuntimeError(
            f"backend returned a suffix of {len(sample.token_ids)} tokens for a limit of {length}"
        )
    if sample.policy_id != sampling.policy_id:
        raise RuntimeError("backend did not score tokens under the requested proposal policy")
    if any(not isfinite(value) for value in sample.token_logprobs):
        raise RuntimeError("a sampled proposal token must have finite proposal log-probability")
    base = SamplingConfig(eos_token_id=sampling.eos_token_id)
    if _is_base_proposal(sampling):
        base_logs = sample.token_logprobs
    elif sample.reference_policy_id == base.policy_id and sample.reference_token_logprobs is not None:
        base_logs = sample.reference_token_logprobs
    else:
        base_logs = _score_one(backend, prefix, sample.token_ids, base)
    if any(not isfinite(value) for value in base_logs):
        raise ValueError("proposal generated a sequence outside the base model support")
    return _Suffix(sample.token_ids, sample.token_logprobs, tuple(base_logs), stopped)


def _token_changes(old: TokenSequence, new: TokenSequence) -> int:
    """Positions where two suffixes differ, counting the longer one's extra tokens."""

    return sum(left != right for left, right in zip(old, new)) + abs(len(old) - len(new))


def run_power_mh_chain(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: PowerMHConfig,
    proposal: SamplingConfig,
    seeds: SeedStream,
    *,
    chain_id: int = 0,
) -> PowerMHChainResult:
    """Run the staged power-target MH for one chain.

    Stage ``k`` targets ``p(y)**alpha`` over outputs of at most ``T_k`` tokens;
    an output that stopped earlier is complete and later stages do not extend it.
    """

    _validate_proposal(proposal)
    tokens: TokenSequence = ()
    base_logs: tuple[float, ...] = ()
    proposal_logs: tuple[float, ...] = ()
    stopped = False
    trace: list[PowerMHStep] = []
    skipped = 0
    for stage_index, stage_length in enumerate(config.stages):
        if not stopped and len(tokens) < stage_length:
            extension = _sample_suffix(
                backend, prefix=prompt + tokens, length=stage_length - len(tokens), sampling=proposal,
                seed=seeds.derive("mh", chain_id, stage_index, "extend"),
                request_id=f"mh:{chain_id}:stage:{stage_index}:extend",
            )
            tokens += extension.token_ids
            base_logs += extension.base_logprobs
            proposal_logs += extension.proposal_logprobs
            stopped = extension.stopped
        for step_index in range(config.stage_updates):
            cut, _, suffix_probability = _draw_suffix(
                stage_length=stage_length,
                schedule=config.suffix_schedule,
                rng=seeds.generator("mh", chain_id, stage_index, step_index, "cut"),
            )
            if cut >= len(tokens):
                skipped += 1
                continue
            suffix = _sample_suffix(
                backend, prefix=prompt + tokens[:cut], length=stage_length - cut, sampling=proposal,
                seed=seeds.derive("mh", chain_id, stage_index, step_index, "proposal"),
                request_id=f"mh:{chain_id}:stage:{stage_index}:step:{step_index}",
            )
            decision = decide_metropolis_hastings(
                current_target_log_density=config.alpha * float(sum(base_logs[cut:])),
                proposed_target_log_density=config.alpha * float(sum(suffix.base_logprobs)),
                forward_proposal_log_probability=float(sum(suffix.proposal_logprobs)),
                reverse_proposal_log_probability=float(sum(proposal_logs[cut:])),
                uniform=float(seeds.generator("mh", chain_id, stage_index, step_index, "accept").random()),
            )
            changes = _token_changes(tokens[cut:], suffix.token_ids)
            if decision.accepted:
                tokens = tokens[:cut] + suffix.token_ids
                base_logs = base_logs[:cut] + suffix.base_logprobs
                proposal_logs = proposal_logs[:cut] + suffix.proposal_logprobs
                stopped = suffix.stopped
            trace.append(PowerMHStep(
                stage_length=stage_length, step=step_index, cut=cut,
                proposed_suffix_length=len(suffix.token_ids), log_acceptance=decision.log_acceptance,
                accepted=decision.accepted, suffix_schedule=config.suffix_schedule,
                suffix_probability=suffix_probability, proposed_token_changes=changes,
                accepted_token_changes=changes if decision.accepted else 0,
            ))
    return PowerMHChainResult(prompt, tokens, base_logs, proposal_logs, tuple(trace), chain_id, skipped)


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

    The chain starts from one complete output and every update draws a cut over
    all ``total_length`` positions. For a base-model proposal the likelihood
    terms cancel, leaving only the reward difference; the expanded ratio below
    also remains correct for any full-support temperature proposal.
    """

    _validate_proposal(proposal)
    initial = _sample_suffix(
        backend, prefix=prompt, length=config.total_length, sampling=proposal,
        seed=seeds.derive("reward_mh", chain_id, "initialize"), request_id=f"reward-mh:{chain_id}:initialize",
    )
    tokens, base_logs, proposal_logs = initial.token_ids, initial.base_logprobs, initial.proposal_logprobs
    current_reward = float(reward(prompt, tokens))
    if not isfinite(current_reward):
        raise ValueError("reward must be finite")
    trace: list[RewardMHStep] = []
    skipped = 0
    for step_index in range(config.updates):
        cut, _, suffix_probability = _draw_suffix(
            stage_length=config.total_length,
            schedule=config.suffix_schedule,
            rng=seeds.generator("reward_mh", chain_id, step_index, "cut"),
        )
        if cut >= len(tokens):
            skipped += 1
            continue
        suffix = _sample_suffix(
            backend, prefix=prompt + tokens[:cut], length=config.total_length - cut, sampling=proposal,
            seed=seeds.derive("reward_mh", chain_id, step_index, "proposal"),
            request_id=f"reward-mh:{chain_id}:step:{step_index}",
        )
        proposed_reward = float(reward(prompt, tokens[:cut] + suffix.token_ids))
        if not isfinite(proposed_reward):
            raise ValueError("reward must be finite")
        decision = decide_metropolis_hastings(
            current_target_log_density=float(sum(base_logs[cut:])) + current_reward / config.reward_temperature,
            proposed_target_log_density=(
                float(sum(suffix.base_logprobs)) + proposed_reward / config.reward_temperature
            ),
            forward_proposal_log_probability=float(sum(suffix.proposal_logprobs)),
            reverse_proposal_log_probability=float(sum(proposal_logs[cut:])),
            uniform=float(seeds.generator("reward_mh", chain_id, step_index, "accept").random()),
        )
        changes = _token_changes(tokens[cut:], suffix.token_ids)
        previous_reward = current_reward
        if decision.accepted:
            tokens = tokens[:cut] + suffix.token_ids
            base_logs = base_logs[:cut] + suffix.base_logprobs
            proposal_logs = proposal_logs[:cut] + suffix.proposal_logprobs
            current_reward = proposed_reward
        trace.append(RewardMHStep(
            step=step_index, cut=cut, proposed_suffix_length=len(suffix.token_ids),
            current_reward=previous_reward, proposed_reward=proposed_reward,
            log_acceptance=decision.log_acceptance, accepted=decision.accepted,
            suffix_schedule=config.suffix_schedule, suffix_probability=suffix_probability,
            proposed_token_changes=changes, accepted_token_changes=changes if decision.accepted else 0,
        ))
    return RewardMHChainResult(prompt, tokens, current_reward, base_logs, proposal_logs, tuple(trace), chain_id,
                               skipped)
