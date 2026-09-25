"""The frozen-history suffix proposal for reward Metropolis--Hastings.

The proposal is a frozen defensive mixture of base-model suffixes and suffixes
of previously generated sequences. Its forward and reverse probabilities are
evaluated exactly, so the Hastings ratio is never clipped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, log

import numpy as np

from inference_scaling.arllm.algorithms.config import RewardMHConfig
from inference_scaling.arllm.algorithms.mh import (
    _draw_suffix,
    _is_base_proposal,
    _sample_suffix,
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
    accepted: bool
    proposal_source: str


@dataclass(frozen=True, slots=True)
class ReplayProposalMHResult:
    token_ids: TokenSequence
    reward: float
    trace: tuple[ReplayProposalMHStep, ...]
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


@dataclass(frozen=True, slots=True)
class ReplayProposalDraw:
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    proposal_logprob: float
    source: str


class FrozenReplaySuffixProposal:
    """A frozen defensive mixture of base suffixes and history suffixes.

    After a kept prefix, the history component proposes the rest of a uniformly
    chosen history sequence that starts with that prefix. A history sequence
    carries the base log-probabilities of its tokens from generation; after a
    shared prefix they are also the log-probabilities of its suffix, so a replay
    needs no scoring pass. History sequences must be complete outputs under the
    chain's length limit and stop rule, as the base draws are.
    """

    def __init__(
        self,
        backend: AutoregressiveBackend,
        prompt: TokenSequence,
        history: Sequence[tuple[TokenSequence, Sequence[float]]],
        *,
        history_mixture: float,
        sampling: SamplingConfig,
    ) -> None:
        if not 0 <= history_mixture < 1:
            raise ValueError("history_mixture must lie in [0, 1)")
        _validate_proposal(sampling)
        if not _is_base_proposal(sampling):
            raise ValueError("defensive replay proposal currently requires base sampling")
        if any(not tokens or len(tokens) != len(logprobs) for tokens, logprobs in history):
            raise ValueError("each history sequence needs one log-probability per token")
        self.backend = backend
        self.prompt = tuple(prompt)
        self.history = [(tuple(tokens), tuple(logprobs)) for tokens, logprobs in history]
        self.history_mixture = float(history_mixture)
        self.sampling = sampling

    def _matches(self, kept: TokenSequence) -> list[tuple[TokenSequence, tuple[float, ...]]]:
        """Suffixes, with their base log-probabilities, of the history sequences that continue ``kept``."""

        cut = len(kept)
        return [(tokens[cut:], logprobs[cut:]) for tokens, logprobs in self.history
                if len(tokens) > cut and tokens[:cut] == kept]

    def _mixture_logprob(
        self, base_logprob: float, matches: Sequence[tuple[TokenSequence, tuple[float, ...]]], suffix: TokenSequence,
    ) -> float:
        if not matches or self.history_mixture == 0:
            return float(base_logprob)
        count = sum(tokens == suffix for tokens, _ in matches)
        terms = [log(1.0 - self.history_mixture) + float(base_logprob)]
        if count:
            terms.append(log(self.history_mixture) + log(count / len(matches)))
        return float(np.logaddexp.reduce(np.asarray(terms, dtype=np.float64)))

    def draw(self, kept: TokenSequence, length: int, *, seed: int, request_id: str) -> ReplayProposalDraw:
        matches = self._matches(kept)
        seeds = SeedStream(seed)
        if matches and float(seeds.generator("replay-proposal-component").random()) < self.history_mixture:
            suffix, logprobs = matches[int(seeds.generator("replay-proposal-history").integers(len(matches)))]
            source = "history"
        else:
            sample = _sample_suffix(
                self.backend, prefix=self.prompt + kept, length=length, sampling=self.sampling,
                seed=seeds.derive("replay-proposal-base"), request_id=request_id,
            )
            suffix, logprobs, source = sample.token_ids, sample.base_logprobs, "base"
        return ReplayProposalDraw(suffix, logprobs, self._mixture_logprob(sum(logprobs), matches, suffix), source)

    def logprob(self, kept: TokenSequence, suffix: TokenSequence, *, base_logprob: float) -> float:
        return self._mixture_logprob(base_logprob, self._matches(kept), tuple(suffix))


def run_reward_mh_chain_replay_proposal(
    proposal: FrozenReplaySuffixProposal,
    config: RewardMHConfig,
    reward: TokenReward,
    seeds: SeedStream,
    *,
    chain_id: int = 0,
) -> ReplayProposalMHResult:
    """Run reward MH with exact forward/reverse defensive replay probabilities."""

    prompt = proposal.prompt

    def score(sequence: TokenSequence) -> float:
        value = float(reward(prompt, sequence))
        if not isfinite(value):
            raise ValueError("reward must be finite")
        return value

    initial = proposal.draw((), config.total_length, seed=seeds.derive("reward_mh", chain_id, "initialize"),
                            request_id=f"reward-mh-replay:{chain_id}:initialize")
    tokens, base_logs = initial.token_ids, initial.base_token_logprobs
    current_reward = score(tokens)
    trace: list[ReplayProposalMHStep] = []
    skipped = 0
    for step_index in range(config.updates):
        cut, _, _ = _draw_suffix(
            stage_length=config.total_length, schedule=config.suffix_schedule,
            rng=seeds.generator("reward_mh", chain_id, step_index, "cut"),
        )
        if cut >= len(tokens):
            skipped += 1
            continue
        kept = tokens[:cut]
        old_p = float(sum(base_logs[cut:]))
        draw = proposal.draw(kept, config.total_length - cut,
                             seed=seeds.derive("reward_mh", chain_id, step_index, "proposal"),
                             request_id=f"reward-mh-replay:{chain_id}:step:{step_index}")
        proposed_reward = score(kept + draw.token_ids)
        decision = decide_metropolis_hastings(
            current_target_log_density=old_p + current_reward / config.reward_temperature,
            proposed_target_log_density=float(sum(draw.base_token_logprobs)) + proposed_reward / config.reward_temperature,
            forward_proposal_log_probability=draw.proposal_logprob,
            reverse_proposal_log_probability=proposal.logprob(kept, tokens[cut:], base_logprob=old_p),
            uniform=float(seeds.generator("reward_mh", chain_id, step_index, "accept").random()),
        )
        if decision.accepted:
            tokens, base_logs = kept + draw.token_ids, base_logs[:cut] + draw.base_token_logprobs
            current_reward = proposed_reward
        trace.append(ReplayProposalMHStep(step_index, cut, decision.accepted, draw.source))
    return ReplayProposalMHResult(tokens, current_reward, tuple(trace), skipped)


__all__ = [
    "FrozenReplaySuffixProposal",
    "ReplayProposalMHResult",
    "ReplayProposalMHStep",
    "run_reward_mh_chain_replay_proposal",
]
