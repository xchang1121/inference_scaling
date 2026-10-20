"""The conditional SIR step shared by the AR and diffusion importance samplers.

A step weighs each candidate block by the mean ``exp(reward / temperature)`` of
its completions, selects a candidate in proportion to that weight and one of its
completions in proportion to the completion's own weight, so it selects a whole
suffix in proportion to its reward weight.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.importance import categorical_index_from_uniform, logmeanexp, normalize_log_weights
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class RolloutEvaluation:
    """One completion of a candidate; its log weight is reward / temperature."""

    token_ids: TokenSequence
    reward: float
    log_weight: float


@dataclass(frozen=True, slots=True)
class ConditionalCandidate:
    token_ids: TokenSequence
    rollouts: tuple[RolloutEvaluation, ...]
    log_weight: float


@dataclass(frozen=True, slots=True)
class RolloutSummary:
    token_count: int
    reward: float
    log_weight: float


@dataclass(frozen=True, slots=True)
class CandidateSummary:
    token_count: int
    rollouts: tuple[RolloutSummary, ...]
    log_weight: float


@dataclass(frozen=True, slots=True)
class ConditionalISStep:
    generated_length_before: int
    candidates: tuple[ConditionalCandidate | CandidateSummary, ...]
    selected_index: int
    # The completion of the selected candidate kept in the sequence; None when the sampler keeps none.
    completion_index: int | None
    # Whether candidate 0 continues the kept sequence.
    retained_candidate: bool
    # Completions generated and scored in this step; a reused one is excluded.
    rollout_evaluations_performed: int
    # Forward token slots this step cost, measured per request; None when the backend cannot separate it.
    # Measurement never enters the step's identity: two execution paths are compared on what they sampled.
    forward_token_slots: int | None = field(default=None, compare=False)
    # Requested (block size, candidates, rollouts), before terminal groups collapse to one leaf.
    sampling_plan: tuple[int, int, int] | None = None
    # Original candidate slots when completion quotas remove entire groups.
    candidate_indices: tuple[int, ...] | None = None

    @property
    def selected(self) -> ConditionalCandidate | CandidateSummary:
        return self.candidates[self.selected_index]

    def compact(self) -> ConditionalISStep:
        """Only lengths and scalar statistics survive a completed step."""
        return replace(self, candidates=tuple(
            CandidateSummary(len(candidate.token_ids), tuple(
                RolloutSummary(len(rollout.token_ids), rollout.reward, rollout.log_weight)
                for rollout in candidate.rollouts), candidate.log_weight)
            if isinstance(candidate, ConditionalCandidate) else candidate for candidate in self.candidates))


@dataclass(frozen=True, slots=True)
class ConditionalISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[ConditionalISStep, ...]
    # Forward token slots of all steps, measured per request; None when any step could not be attributed.
    forward_token_slots: int | None = field(default=None, compare=False)
    stopping_reason: str = "stop"


def weigh(blocks: Sequence[TokenSequence],
          completions: Sequence[Sequence[RolloutEvaluation]]) -> tuple[ConditionalCandidate, ...]:
    """Each candidate block with its completions, weighted by the mean of their weights."""

    return tuple(ConditionalCandidate(block, tuple(group), logmeanexp([item.log_weight for item in group]))
                 for block, group in zip(blocks, completions, strict=True))


def select(candidates: Sequence[ConditionalCandidate], seeds: SeedStream, *key: object) -> tuple[int, int]:
    """A candidate in proportion to its weight and one of its completions in proportion to the completion's weight.

    A completion is drawn for every candidate, so the kept one does not depend on which candidate is selected.
    """

    completions = [categorical_index_from_uniform(
        normalize_log_weights([rollout.log_weight for rollout in candidate.rollouts]),
        float(seeds.generator(*key, "candidate", index, "completion").random()),
    ) for index, candidate in enumerate(candidates)]
    selected = categorical_index_from_uniform(normalize_log_weights([candidate.log_weight for candidate in candidates]),
                                              float(seeds.generator(*key, "select").random()))
    return selected, completions[selected]


__all__ = ["ConditionalCandidate", "ConditionalISResult", "ConditionalISStep", "RolloutEvaluation", "select", "weigh"]
