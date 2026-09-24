"""The frozen-history suffix proposal for reward Metropolis--Hastings.

The proposal is a frozen defensive mixture of base-model suffixes and suffixes
of previously observed sequences. Its forward and reverse probabilities are
evaluated exactly, so the Hastings ratio is never clipped.
"""

from __future__ import annotations

import threading
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite, log

import numpy as np

from inference_scaling.arllm.algorithms.config import RewardMHConfig
from inference_scaling.arllm.algorithms.mh import (
    _draw_suffix,
    _is_base_proposal,
    _sample_exact_length,
    _score_one,
    _validate_proposal,
)
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, TokenSequence
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.mh import decide_metropolis_hastings
from inference_scaling.shared.types import TokenReward


@dataclass(frozen=True, slots=True)
class ReplayProposalMHStep:
    step: int
    cut: int
    proposed_suffix_length: int
    current_reward: float
    proposed_reward: float
    old_proposal_logprob: float
    new_proposal_logprob: float
    log_acceptance: float
    accepted: bool
    proposal_source: str
    suffix_schedule: str
    suffix_probability: float
    proposed_token_changes: int
    accepted_token_changes: int


@dataclass(frozen=True, slots=True)
class ReplayProposalMHResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    reward: float
    base_token_logprobs: tuple[float, ...]
    trace: tuple[ReplayProposalMHStep, ...]
    chain_id: int
    proposal_logprob: float

    @property
    def attempts(self) -> int:
        return len(self.trace)

    @property
    def accepted(self) -> int:
        return sum(step.accepted for step in self.trace)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.attempts if self.attempts else 0.0


@dataclass(frozen=True, slots=True)
class ReplayProposalSnapshot:
    stored_suffixes: int
    stored_prefix_lengths: int
    base_draws: int
    history_draws: int
    logprob_queries: int


@dataclass(frozen=True, slots=True)
class ReplayProposalDraw:
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    proposal_logprob: float
    source: str


def _finite_reward(
    reward: TokenReward, prompt: TokenSequence, sequence: TokenSequence
) -> float:
    value = float(reward(prompt, sequence))
    if not isfinite(value):
        raise ValueError("reward must be finite")
    return value


class FrozenReplaySuffixProposal:
    """A frozen defensive mixture of base suffixes and empirical replay suffixes."""

    def __init__(
        self,
        backend: AutoregressiveBackend,
        *,
        history_mixture: float = 0.25,
        sampling: SamplingConfig | None = None,
    ) -> None:
        if not 0 <= history_mixture < 1:
            raise ValueError("history_mixture must lie in [0, 1)")
        self.backend = backend
        self.sampling = sampling or SamplingConfig()
        _validate_proposal(self.sampling)
        if not _is_base_proposal(self.sampling):
            raise ValueError("defensive replay proposal currently requires base sampling")
        self.history_mixture = float(history_mixture)
        self._suffixes: dict[tuple[TokenSequence, int], Counter[TokenSequence]] = defaultdict(
            Counter
        )
        self._frozen = False
        self._base_draws = 0
        self._history_draws = 0
        self._logprob_queries = 0
        self._lock = threading.RLock()

    def observe_suffix(self, prefix: TokenSequence, suffix: TokenSequence) -> None:
        if self._frozen:
            raise RuntimeError("replay proposal is frozen")
        if not suffix:
            raise ValueError("replay proposal suffix cannot be empty")
        self._suffixes[(tuple(prefix), len(suffix))][tuple(suffix)] += 1

    def observe_sequence(self, prompt: TokenSequence, sequence: TokenSequence) -> None:
        values = tuple(sequence)
        for cut in range(len(values)):
            self.observe_suffix(tuple(prompt) + values[:cut], values[cut:])

    def observe_sequences(
        self, prompt: TokenSequence, sequences: Iterable[TokenSequence]
    ) -> None:
        for sequence in sequences:
            self.observe_sequence(prompt, sequence)

    def freeze(self) -> None:
        self._frozen = True

    @staticmethod
    def _history_probability(
        counts: Counter[TokenSequence] | None, suffix: TokenSequence
    ) -> float:
        if not counts:
            return 0.0
        return counts.get(tuple(suffix), 0) / sum(counts.values())

    def _mixture_logprob(
        self,
        base_logprob: float,
        counts: Counter[TokenSequence] | None,
        suffix: TokenSequence,
    ) -> float:
        if not counts or self.history_mixture == 0:
            return float(base_logprob)
        history_probability = self._history_probability(counts, suffix)
        terms = [log(1.0 - self.history_mixture) + float(base_logprob)]
        if history_probability > 0:
            terms.append(log(self.history_mixture) + log(history_probability))
        return float(np.logaddexp.reduce(np.asarray(terms, dtype=np.float64)))

    def draw(
        self,
        prefix: TokenSequence,
        length: int,
        *,
        seed: int,
        request_id: str,
    ) -> ReplayProposalDraw:
        if not self._frozen:
            raise RuntimeError("replay proposal must be frozen before sampling")
        if length <= 0:
            raise ValueError("replay proposal length must be positive")
        counts = self._suffixes.get((tuple(prefix), int(length)))
        component_rng = SeedStream(seed).generator("replay-proposal-component")
        use_history = bool(counts) and float(component_rng.random()) < self.history_mixture
        if use_history:
            assert counts is not None
            support = tuple(sorted(counts))
            masses = np.asarray([counts[value] for value in support], dtype=np.float64)
            masses /= masses.sum()
            index = int(
                SeedStream(seed)
                .generator("replay-proposal-history")
                .choice(len(support), p=masses)
            )
            suffix = support[index]
            base_token_logprobs = _score_one(self.backend, prefix, suffix, None)
            source = "history"
            with self._lock:
                self._history_draws += 1
        else:
            suffix, base_token_logprobs, cached_base = _sample_exact_length(
                self.backend,
                prefix=prefix,
                length=length,
                sampling=self.sampling,
                seed=SeedStream(seed).derive("replay-proposal-base"),
                request_id=request_id,
            )
            if cached_base is not None:
                base_token_logprobs = cached_base
            source = "base"
            with self._lock:
                self._base_draws += 1
        base_total = float(sum(base_token_logprobs))
        return ReplayProposalDraw(
            token_ids=tuple(suffix),
            base_token_logprobs=tuple(base_token_logprobs),
            proposal_logprob=self._mixture_logprob(base_total, counts, tuple(suffix)),
            source=source,
        )

    def logprob(
        self,
        prefix: TokenSequence,
        suffix: TokenSequence,
        *,
        base_logprob: float | None = None,
    ) -> float:
        if not self._frozen:
            raise RuntimeError("replay proposal must be frozen before scoring")
        if not suffix:
            raise ValueError("replay proposal suffix cannot be empty")
        if base_logprob is None:
            base_logprob = float(sum(_score_one(self.backend, prefix, suffix, None)))
        counts = self._suffixes.get((tuple(prefix), len(suffix)))
        with self._lock:
            self._logprob_queries += 1
        return self._mixture_logprob(float(base_logprob), counts, tuple(suffix))

    def snapshot(self) -> ReplayProposalSnapshot:
        with self._lock:
            return ReplayProposalSnapshot(
                stored_suffixes=sum(sum(counts.values()) for counts in self._suffixes.values()),
                stored_prefix_lengths=len(self._suffixes),
                base_draws=self._base_draws,
                history_draws=self._history_draws,
                logprob_queries=self._logprob_queries,
            )


def run_reward_mh_chain_replay_proposal(
    proposal: FrozenReplaySuffixProposal,
    prompt: TokenSequence,
    config: RewardMHConfig,
    reward: TokenReward,
    seeds: SeedStream,
    *,
    chain_id: int = 0,
) -> ReplayProposalMHResult:
    """Run reward MH with exact forward/reverse defensive replay probabilities."""

    proposal.freeze()
    initial = proposal.draw(
        prompt,
        config.total_length,
        seed=seeds.derive("reward_mh", chain_id, "initialize"),
        request_id=f"reward-mh-replay:{chain_id}:initialize",
    )
    tokens = initial.token_ids
    base_logs = initial.base_token_logprobs
    current_reward = _finite_reward(reward, prompt, tokens)
    trace: list[ReplayProposalMHStep] = []

    for step_index in range(config.updates):
        cut, suffix_length, suffix_probability = _draw_suffix(
            stage_length=config.total_length,
            schedule=config.suffix_schedule,
            rng=seeds.generator("reward_mh", chain_id, step_index, "cut"),
        )
        retained = tokens[:cut]
        prefix = prompt + retained
        old_suffix = tokens[cut:]
        old_p = float(sum(base_logs[cut:]))
        old_q = proposal.logprob(prefix, old_suffix, base_logprob=old_p)
        draw = proposal.draw(
            prefix,
            config.total_length - cut,
            seed=seeds.derive("reward_mh", chain_id, step_index, "proposal"),
            request_id=f"reward-mh-replay:{chain_id}:step:{step_index}",
        )
        proposed_sequence = retained + draw.token_ids
        proposed_reward = _finite_reward(reward, prompt, proposed_sequence)
        new_p = float(sum(draw.base_token_logprobs))
        decision = decide_metropolis_hastings(
            current_target_log_density=(
                old_p + current_reward / config.reward_temperature
            ),
            proposed_target_log_density=(
                new_p + proposed_reward / config.reward_temperature
            ),
            forward_proposal_log_probability=draw.proposal_logprob,
            reverse_proposal_log_probability=old_q,
            uniform=float(
                seeds.generator("reward_mh", chain_id, step_index, "accept").random()
            ),
        )
        log_acceptance = decision.log_acceptance
        accepted = decision.accepted
        previous_reward = current_reward
        proposed_token_changes = sum(
            old != new
            for old, new in zip(old_suffix, draw.token_ids, strict=True)
        )
        if accepted:
            tokens = proposed_sequence
            base_logs = base_logs[:cut] + draw.base_token_logprobs
            current_reward = proposed_reward
        trace.append(
            ReplayProposalMHStep(
                step=step_index,
                cut=cut,
                proposed_suffix_length=suffix_length,
                current_reward=previous_reward,
                proposed_reward=proposed_reward,
                old_proposal_logprob=old_q,
                new_proposal_logprob=draw.proposal_logprob,
                log_acceptance=log_acceptance,
                accepted=accepted,
                proposal_source=draw.source,
                suffix_schedule=config.suffix_schedule,
                suffix_probability=suffix_probability,
                proposed_token_changes=proposed_token_changes,
                accepted_token_changes=(proposed_token_changes if accepted else 0),
            )
        )

    final_q = proposal.logprob(
        prompt,
        tokens,
        base_logprob=float(sum(base_logs)),
    )
    return ReplayProposalMHResult(
        prompt=prompt,
        token_ids=tokens,
        reward=current_reward,
        base_token_logprobs=base_logs,
        trace=tuple(trace),
        chain_id=chain_id,
        proposal_logprob=final_q,
    )


__all__ = [
    "FrozenReplaySuffixProposal",
    "ReplayProposalMHResult",
    "ReplayProposalMHStep",
    "run_reward_mh_chain_replay_proposal",
]
