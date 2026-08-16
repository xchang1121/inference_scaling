"""Model-family-independent driver for finite stepwise generation.

The driver sees an opaque generation state and opaque candidates.  AR and
diffusion adapters implement state transitions and rollout collection; the
driver only performs the energy-guided candidate selection shared by both.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from inference_scaling.shared.rng import SeedStream

StateT = TypeVar("StateT")
ProposalT = TypeVar("ProposalT")
CandidateT = TypeVar("CandidateT")


@dataclass(frozen=True, slots=True)
class StepwiseCandidate(Generic[CandidateT]):
    """One evaluated candidate and its conditional log-energy."""

    value: CandidateT
    log_energy: float

    def __post_init__(self) -> None:
        if not isfinite(self.log_energy):
            raise ValueError("candidate log-energy must be finite")


@dataclass(frozen=True, slots=True)
class StepwiseSelection(Generic[StateT, CandidateT]):
    """One energy-guided transition selected from a finite candidate set."""

    step_index: int
    state_before: StateT
    candidates: tuple[StepwiseCandidate[CandidateT], ...]
    probabilities: tuple[float, ...]
    selected_index: int

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if not self.candidates:
            raise ValueError("at least one evaluated candidate is required")
        if len(self.probabilities) != len(self.candidates):
            raise ValueError("each candidate must have one selection probability")
        if not 0 <= self.selected_index < len(self.candidates):
            raise ValueError("selected_index lies outside the candidate set")

    @property
    def selected(self) -> StepwiseCandidate[CandidateT]:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class StepwiseGenerationResult(Generic[StateT, CandidateT]):
    final_state: StateT
    steps: tuple[StepwiseSelection[StateT, CandidateT], ...]


@runtime_checkable
class StepwiseGenerationBackend(Protocol[StateT, ProposalT, CandidateT]):
    """Operations needed by the common energy-guided generation loop.

    A state may be an AR prefix, a partially denoised sequence, or another
    finite-step generation state.  Implementations retain all model-specific
    batching, transition, scoring, and completion logic.
    """

    @property
    def initial_state(self) -> StateT: ...

    def is_terminal(self, state: StateT) -> bool: ...

    def propose(
        self,
        state: StateT,
        step_index: int,
        seeds: SeedStream,
    ) -> Sequence[ProposalT]: ...

    def evaluate(
        self,
        state: StateT,
        proposals: Sequence[ProposalT],
        step_index: int,
        seeds: SeedStream,
    ) -> Sequence[StepwiseCandidate[CandidateT]]: ...

    def advance(
        self,
        state: StateT,
        selected: CandidateT,
        step_index: int,
    ) -> StateT: ...


def normalize_log_energies(log_energies: Sequence[float]) -> tuple[float, ...]:
    """Normalize finite log-energies without exposing model-family details."""

    if not log_energies:
        raise ValueError("at least one candidate log-energy is required")
    values = np.asarray(log_energies, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("candidate log-energies must be finite")
    shifted = np.exp(values - float(np.max(values)))
    total = float(shifted.sum())
    if not isfinite(total) or total <= 0:
        raise ValueError("candidate energies cannot be normalized")
    return tuple(float(value) for value in shifted / total)


def select_stepwise_candidate(
    *,
    state: StateT,
    candidates: Sequence[StepwiseCandidate[CandidateT]],
    step_index: int,
    rng: np.random.Generator,
) -> StepwiseSelection[StateT, CandidateT]:
    """Perform the common finite-candidate SIR selection step."""

    evaluated = tuple(candidates)
    probabilities = normalize_log_energies(
        [candidate.log_energy for candidate in evaluated]
    )
    selected_index = int(rng.choice(len(evaluated), p=probabilities))
    return StepwiseSelection(
        step_index=step_index,
        state_before=state,
        candidates=evaluated,
        probabilities=probabilities,
        selected_index=selected_index,
    )


def stepwise_generation_step(
    backend: StepwiseGenerationBackend[StateT, ProposalT, CandidateT],
    state: StateT,
    step_index: int,
    seeds: SeedStream,
    *,
    selection_namespace: Sequence[object] = ("stepwise",),
) -> StepwiseSelection[StateT, CandidateT]:
    """Generate, evaluate, and select one model-independent transition."""

    if backend.is_terminal(state):
        raise ValueError("cannot advance a terminal generation state")
    proposals = tuple(backend.propose(state, step_index, seeds))
    if not proposals:
        raise RuntimeError("stepwise backend returned no candidates")
    candidates = tuple(backend.evaluate(state, proposals, step_index, seeds))
    if len(candidates) != len(proposals):
        raise RuntimeError("stepwise backend must evaluate every proposal exactly once")
    return select_stepwise_candidate(
        state=state,
        candidates=candidates,
        step_index=step_index,
        rng=seeds.generator(*selection_namespace, step_index, "select"),
    )


def run_stepwise_generation(
    backend: StepwiseGenerationBackend[StateT, ProposalT, CandidateT],
    seeds: SeedStream,
    *,
    selection_namespace: Sequence[object] = ("stepwise",),
) -> StepwiseGenerationResult[StateT, CandidateT]:
    """Run energy-guided transitions until the adapter reaches a terminal state."""

    state = backend.initial_state
    steps: list[StepwiseSelection[StateT, CandidateT]] = []
    step_index = 0
    while not backend.is_terminal(state):
        selection = stepwise_generation_step(
            backend,
            state,
            step_index,
            seeds,
            selection_namespace=selection_namespace,
        )
        state = backend.advance(state, selection.selected.value, step_index)
        steps.append(selection)
        step_index += 1
    return StepwiseGenerationResult(state, tuple(steps))


__all__ = [
    "StepwiseCandidate",
    "StepwiseGenerationBackend",
    "StepwiseGenerationResult",
    "StepwiseSelection",
    "normalize_log_energies",
    "run_stepwise_generation",
    "select_stepwise_candidate",
    "stepwise_generation_step",
]
