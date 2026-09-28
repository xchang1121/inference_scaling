"""The conditional SIR step shared by the AR and diffusion importance samplers.

A step weighs each candidate block by the mean ``exp(reward / temperature)`` of
its completions, selects a candidate in proportion to that weight and one of its
completions in proportion to the completion's own weight, so it selects a whole
suffix in proportion to its reward weight.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
class ConditionalISStep:
    generated_length_before: int
    candidates: tuple[ConditionalCandidate, ...]
    selected_index: int
    # The completion of the selected candidate kept in the sequence; None when the sampler keeps none.
    completion_index: int | None
    # Whether candidate 0 continues the kept sequence.
    retained_candidate: bool
    # Completions generated and scored in this step; a reused one is excluded.
    rollout_evaluations_performed: int

    @property
    def selected(self) -> ConditionalCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class ConditionalISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[ConditionalISStep, ...]


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
