"""Suffix-resampling Metropolis--Hastings over complete outputs.

The target is ``p(y)**alpha * exp(r(y) / tau)`` for the base policy ``p``: the
power target without a reward, the reward target with ``alpha = 1``. A state
is one complete output: generation stops at EOS (or at a scope boundary) or at
the stage's length limit ``T``. A move draws a cut position from a
full-support distribution over the ``T`` positions of the stage, keeps the
prefix before it, regenerates the suffix from the proposal until it stops or
reaches ``T``, and applies the full forward/reverse proposal correction. The
cut distribution does not depend on the state, so it cancels in the Hastings
ratio; a cut at or after the end of a stopped output would regenerate an empty
suffix, so that move is skipped without a model call. Per-token base and
proposal log-probabilities of the current state are cached.

``p**alpha`` is defined for every prefix, so a chain without a reward grows its
outputs stage by stage; a reward needs complete outputs, so a rewarded chain
has one full-length stage. The proposal is the base policy at another
temperature, optionally mixed with a frozen history of earlier complete
outputs: after the kept prefix, the history component proposes the rest of a
uniformly chosen history output that continues it, whose base
log-probabilities came with its generation, and the exact mixture
probabilities enter the Hastings ratio. With ``suffix_replay`` the current
suffix goes to the backend as a draft: the proposal is unchanged, and its
tokens that repeat the current ones cost no model call. For the power target
with a proposal temperature between ``1/alpha`` and 1 of the base, each token's
log-weight ``alpha log p - log q`` is at most zero, so with ``early_rejection``
the accept uniform is drawn first and the proposal stops as soon as it can no
longer be accepted.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, log
from sys import float_info

import numpy as np

from inference_scaling.arllm.algorithms.config import MHConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.shared.sampling.mh import decide_metropolis_hastings
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenBatchReward
from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    Draft,
    GenerationRequest,
    LogWeightStop,
    ScoreRequest,
    TokenSequence,
)

# An earlier complete output: its tokens, their base log-probabilities and, when reported, their CDF intervals.
History = tuple[TokenSequence, tuple[float, ...], tuple[tuple[float, float], ...] | None]


@dataclass(frozen=True, slots=True)
class MHStep:
    stage_length: int
    step: int
    cut: int
    # Generated proposal tokens; a proposal stopped by early rejection counts the tokens before the stop,
    # and its log_acceptance is the bound at the stop.
    proposed_suffix_length: int
    # "base" or "history".
    proposal_source: str
    current_reward: float
    proposed_reward: float
    log_acceptance: float
    accepted: bool
    suffix_probability: float
    proposed_token_changes: int
    accepted_token_changes: int
    # Leading proposal tokens replayed from the current suffix.
    replayed_tokens: int
    early_rejected: bool


@dataclass(frozen=True, slots=True)
class MHChainResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    # The final output's reward; zero without a reward.
    reward: float
    base_token_logprobs: tuple[float, ...]
    proposal_token_logprobs: tuple[float, ...]
    trace: tuple[MHStep, ...]
    chain_id: int
    # Cuts past the end of a stopped output: no proposal, state unchanged.
    skipped: int

    @property
    def attempts(self) -> int:
        return len(self.trace)

    @property
    def accepted(self) -> int:
        return sum(step.accepted for step in self.trace)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0

    def _mean(self, name: str) -> float:
        return sum(getattr(step, name) for step in self.trace) / self.attempts if self.attempts else 0.0

    @property
    def mean_proposed_suffix_length(self) -> float:
        return self._mean("proposed_suffix_length")

    @property
    def mean_accepted_token_changes(self) -> float:
        return self._mean("accepted_token_changes")

    @property
    def replayed_tokens(self) -> int:
        return sum(step.replayed_tokens for step in self.trace)

    @property
    def early_rejected(self) -> int:
        return sum(step.early_rejected for step in self.trace)

    @property
    def proposal_sources(self) -> dict[str, int]:
        return dict(Counter(step.proposal_source for step in self.trace))

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
    # Ended at EOS, a scope boundary or a rejection rather than at the length limit.
    stopped: bool
    # The proposal's CDF interval of each token when the backend reports them.
    bounds: tuple[tuple[float, float], ...] | None
    # Stopped by its log-weight stop: an incomplete proposal that cannot be accepted.
    rejected: bool


def _draft(tokens: TokenSequence, proposal_logs: tuple[float, ...], base_logs: tuple[float, ...],
           bounds: tuple[tuple[float, float], ...] | None, cut: int) -> Draft | None:
    """The current suffix after ``cut`` as the draft of a proposal from the same prefix."""

    return None if bounds is None else Draft(tokens[cut:], proposal_logs[cut:], base_logs[cut:], bounds[cut:])


def _replayed(current: TokenSequence, proposed: TokenSequence) -> int:
    """Leading tokens a proposal shares with the current suffix: the ones its draft replay kept."""

    return next((index for index, (left, right) in enumerate(zip(current, proposed)) if left != right),
                min(len(current), len(proposed)))


def _sample_suffix(
    backend: AutoregressiveBackend,
    *,
    prefix: TokenSequence,
    length: int,
    sampling: SamplingConfig,
    base: SamplingConfig,
    seed: int,
    request_id: str,
    draft: Draft | None = None,
    log_weight_stop: LogWeightStop | None = None,
) -> _Suffix:
    """Generate up to ``length`` tokens with their proposal and base log-probabilities."""

    sample = backend.sample_batch([GenerationRequest(prefix, length, sampling, seed, request_id,
                                                     reference_temperature=base.temperature, draft=draft,
                                                     log_weight_stop=log_weight_stop)])[0]
    stopped = sample.finish_reason != "length"
    if not 0 < len(sample.token_ids) <= length or (len(sample.token_ids) < length and not stopped):
        raise RuntimeError(
            f"backend returned a suffix of {len(sample.token_ids)} tokens for a limit of {length}"
        )
    if sample.policy_id != sampling.policy_id:
        raise RuntimeError("backend did not score tokens under the requested proposal policy")
    if any(not isfinite(value) for value in sample.token_logprobs):
        raise RuntimeError("a sampled proposal token must have finite proposal log-probability")
    if sampling == base:
        base_logs = sample.token_logprobs
    elif sample.reference_policy_id == base.policy_id and sample.reference_token_logprobs is not None:
        base_logs = sample.reference_token_logprobs
    else:
        base_logs = _score_one(backend, prefix, sample.token_ids, base)
    if any(not isfinite(value) for value in base_logs):
        raise ValueError("proposal generated a sequence outside the base model support")
    return _Suffix(sample.token_ids, sample.token_logprobs, tuple(base_logs), stopped, sample.token_cdf_bounds,
                   sample.finish_reason == "rejected")


def _token_changes(old: TokenSequence, new: TokenSequence) -> int:
    """Positions where two suffixes differ, counting the longer one's extra tokens."""

    return sum(left != right for left, right in zip(old, new)) + abs(len(old) - len(new))


def run_mh_chain(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: MHConfig,
    seeds: SeedStream,
    *,
    base: SamplingConfig,
    proposal: SamplingConfig,
    reward: TokenBatchReward | None = None,
    history: Sequence[History] = (),
    history_mixture: float = 0.0,
    chain_id: int = 0,
) -> MHChainResult:
    """Run one chain toward ``p(y)**alpha * exp(r(y) / reward_temperature)`` for the ``base`` policy ``p``.

    ``history`` holds complete outputs of the base policy under the chain's
    length limit and stop rule; the proposal mixes them in with weight
    ``history_mixture``, which needs the base policy as the proposal.
    """

    _validate_proposal(base)
    _validate_proposal(proposal)
    if (reward is None) != (config.reward_temperature is None):
        raise ValueError("a reward and its temperature go together")
    if not 0 <= history_mixture < 1:
        raise ValueError("history_mixture must lie in [0, 1)")
    if history_mixture and (reward is None or proposal != base):
        raise ValueError("a frozen history mixes into the base proposal of a rewarded, full-length chain")
    if any(not tokens or len(tokens) != len(logprobs) for tokens, logprobs, _ in history):
        raise ValueError("each history output needs one log-probability per token")
    # With a power target and a proposal temperature between 1/alpha and 1 of the base each token's
    # log-weight is at most zero, so comparing the clamped weights removes only rounding.
    monotone = reward is None and 1 / config.alpha <= proposal.temperature / base.temperature <= 1
    if config.early_rejection and not monotone:
        raise ValueError("early rejection needs a target without a reward and a proposal temperature between 1/alpha "
                         "and 1 of the base")
    weight = LogWeightStop(config.alpha, 0.0, 0.0)

    def score(tokens: TokenSequence) -> float:
        if reward is None:
            return 0.0
        value = float(reward(prompt, [tokens])[0])
        if not isfinite(value):
            raise ValueError("reward must be finite")
        return value

    def target(base_logprobs: Sequence[float], value: float) -> float:
        return config.alpha * float(sum(base_logprobs)) + (value / config.reward_temperature if reward else 0.0)

    def continuations(kept: TokenSequence) -> list[History]:
        """The rest of each history output that continues ``kept``."""

        cut = len(kept)
        return [(tokens[cut:], logprobs[cut:], None if bounds is None else bounds[cut:])
                for tokens, logprobs, bounds in history if len(tokens) > cut and tokens[:cut] == kept]

    def log_q(kept: TokenSequence, suffix: TokenSequence, logprob: float) -> float:
        """The proposal log-probability of ``suffix`` after ``kept``, given its policy log-probability."""

        matches = continuations(kept) if history_mixture else []
        if not matches:
            return logprob
        count = sum(tokens == suffix for tokens, _, _ in matches)
        terms = [log(1 - history_mixture) + logprob] + ([log(history_mixture) + log(count / len(matches))]
                                                         if count else [])
        return float(np.logaddexp.reduce(terms))

    def propose(kept: TokenSequence, limit: int, key: tuple[object, ...], request_id: str,
                draft: Draft | None = None, stop: LogWeightStop | None = None) -> tuple[_Suffix, str]:
        matches = continuations(kept) if history_mixture else []
        if matches and float(seeds.generator(*key, "source").random()) < history_mixture:
            tokens, logprobs, bounds = matches[int(seeds.generator(*key, "history").integers(len(matches)))]
            return _Suffix(tokens, logprobs, logprobs, len(kept) + len(tokens) < limit, bounds, False), "history"
        return _sample_suffix(backend, prefix=prompt + kept, length=limit - len(kept), sampling=proposal, base=base,
                              seed=seeds.derive(*key), request_id=request_id, draft=draft,
                              log_weight_stop=stop), "base"

    tokens: TokenSequence = ()
    base_logs: tuple[float, ...] = ()
    proposal_logs: tuple[float, ...] = ()
    bounds: tuple[tuple[float, float], ...] | None = ()
    stopped, current_reward, skipped = False, 0.0, 0
    trace: list[MHStep] = []
    for stage, (limit, updates) in enumerate(config.stages):
        if not stopped and len(tokens) < limit:
            extension, _ = propose(tokens, limit, ("mh", chain_id, stage, "extend"), f"mh:{chain_id}:stage:{stage}:extend")
            tokens += extension.token_ids
            base_logs += extension.base_logprobs
            proposal_logs += extension.proposal_logprobs
            bounds = None if bounds is None or extension.bounds is None else bounds + extension.bounds
            stopped, current_reward = extension.stopped, score(tokens)
        for step in range(updates):
            cut, _, suffix_probability = _draw_suffix(stage_length=limit, schedule=config.suffix_schedule,
                                                      rng=seeds.generator("mh", chain_id, stage, step, "cut"))
            if cut >= len(tokens):
                skipped += 1
                continue
            kept = tokens[:cut]
            draft = _draft(tokens, proposal_logs, base_logs, bounds, cut) if config.suffix_replay else None
            uniform = float(seeds.generator("mh", chain_id, stage, step, "accept").random())
            current = weight.weight(base_logs[cut:], proposal_logs[cut:])
            suffix, source = propose(
                kept, limit, ("mh", chain_id, stage, step, "proposal"), f"mh:{chain_id}:stage:{stage}:step:{step}",
                draft, LogWeightStop(config.alpha, current, log(max(uniform, float_info.min)))
                if config.early_rejection else None)
            proposed_reward = score(kept + suffix.token_ids)
            if monotone:
                decision = decide_metropolis_hastings(
                    current_target_log_density=current, uniform=uniform,
                    proposed_target_log_density=weight.weight(suffix.base_logprobs, suffix.proposal_logprobs),
                )
            else:
                decision = decide_metropolis_hastings(
                    current_target_log_density=target(base_logs[cut:], current_reward),
                    proposed_target_log_density=target(suffix.base_logprobs, proposed_reward),
                    forward_proposal_log_probability=log_q(kept, suffix.token_ids, float(sum(suffix.proposal_logprobs))),
                    reverse_proposal_log_probability=log_q(kept, tokens[cut:], float(sum(proposal_logs[cut:]))),
                    uniform=uniform,
                )
            if suffix.rejected and decision.accepted:
                raise RuntimeError("an early-rejected proposal passed the acceptance test")
            changes = _token_changes(tokens[cut:], suffix.token_ids)
            replayed = 0 if draft is None or source != "base" else _replayed(tokens[cut:], suffix.token_ids)
            trace.append(MHStep(
                stage_length=limit, step=step, cut=cut, proposed_suffix_length=len(suffix.token_ids),
                proposal_source=source, current_reward=current_reward, proposed_reward=proposed_reward,
                log_acceptance=decision.log_acceptance, accepted=decision.accepted,
                suffix_probability=suffix_probability, proposed_token_changes=changes,
                accepted_token_changes=changes if decision.accepted else 0, replayed_tokens=replayed,
                early_rejected=suffix.rejected,
            ))
            if decision.accepted:
                tokens = kept + suffix.token_ids
                base_logs = base_logs[:cut] + suffix.base_logprobs
                proposal_logs = proposal_logs[:cut] + suffix.proposal_logprobs
                bounds = None if bounds is None or suffix.bounds is None else bounds[:cut] + suffix.bounds
                stopped, current_reward = suffix.stopped, proposed_reward
    return MHChainResult(prompt, tokens, current_reward, base_logs, proposal_logs, tuple(trace), chain_id, skipped)


__all__ = ["History", "MHChainResult", "MHStep", "run_mh_chain", "suffix_length_probabilities"]
