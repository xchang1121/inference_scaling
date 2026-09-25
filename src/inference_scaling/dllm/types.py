"""Requests, sampled trajectories, and backend contracts for dLLMs."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Protocol, Sequence, runtime_checkable

from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class DiffusionGenerationRequest:
    prefix: TokenSequence
    generation_length: int
    sampling: DiffusionSamplingConfig
    seed: int
    request_id: str
    # Also report the trajectory's log-probability at this base temperature, from the same logits.
    reference_temperature: float | None = None

    def __post_init__(self) -> None:
        self.sampling.validate_generation_length(
            self.generation_length,
            prefix_length=len(self.prefix),
        )
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.reference_temperature is not None and not (
            isfinite(self.reference_temperature) and self.reference_temperature > 0
        ):
            raise ValueError("reference_temperature must be finite and positive")


@dataclass(frozen=True, slots=True)
class DiffusionTraceStep:
    """One committed reverse-diffusion transition.

    Positions are relative to the generated continuation.  ``logprob`` includes
    both the selected token probabilities and, for random remasking, the
    uniform subset probability.
    """

    block_index: int
    step_index: int
    positions: tuple[int, ...]
    token_ids: TokenSequence
    logprob: float | None

    def __post_init__(self) -> None:
        if self.block_index < 0 or self.step_index < 0:
            raise ValueError("trace indices must be non-negative")
        if len(self.positions) != len(self.token_ids):
            raise ValueError("each committed position must have one token")
        if tuple(sorted(self.positions)) != self.positions:
            raise ValueError("committed positions must be strictly sorted")
        if len(set(self.positions)) != len(self.positions):
            raise ValueError("committed positions must be unique")
        if self.logprob is not None and not isfinite(self.logprob):
            raise ValueError("trace log-probability must be finite")


@dataclass(frozen=True, slots=True)
class DiffusionSample:
    prefix: TokenSequence
    token_ids: TokenSequence
    trace: tuple[DiffusionTraceStep, ...]
    trajectory_logprob: float | None
    policy_id: str
    model_id: str
    request_id: str
    finish_reason: str = "length"
    reference_trajectory_logprob: float | None = None

    def __post_init__(self) -> None:
        if self.reference_trajectory_logprob is not None and not isfinite(self.reference_trajectory_logprob):
            raise ValueError("reference_trajectory_logprob must be finite")
        seen: set[int] = set()
        for step in self.trace:
            for position, token_id in zip(step.positions, step.token_ids, strict=True):
                if not 0 <= position < len(self.token_ids):
                    raise ValueError("trace position lies outside the continuation")
                if position in seen:
                    raise ValueError("a continuation position was committed more than once")
                if self.token_ids[position] != token_id:
                    raise ValueError("trace token does not match the final continuation")
                seen.add(position)
        if self.trace and len(seen) != len(self.token_ids):
            raise ValueError("a complete trace must commit every continuation position")
        step_logprobs = [step.logprob for step in self.trace]
        if self.trajectory_logprob is not None:
            if self.token_ids and not self.trace:
                raise ValueError("an exact trajectory requires a complete trace")
            if not isfinite(self.trajectory_logprob):
                raise ValueError("trajectory_logprob must be finite")
            if any(value is None for value in step_logprobs):
                raise ValueError("an exact trajectory requires every step probability")
            total = sum(float(value) for value in step_logprobs)
            if not isclose(total, self.trajectory_logprob, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("trajectory_logprob must equal the sum of step log-probabilities")


@runtime_checkable
class DiffusionBackend(Protocol):
    @property
    def model_id(self) -> str: ...

    def sample_batch(
        self, requests: Sequence[DiffusionGenerationRequest]
    ) -> list[DiffusionSample]: ...


__all__ = [
    "DiffusionBackend",
    "DiffusionGenerationRequest",
    "DiffusionSample",
    "DiffusionTraceStep",
]
