"""Core request and backend contracts.

Algorithms depend on these contracts rather than on Transformers or a particular
inference server.  A backend must return probabilities under the *actual* sampling
policy, not merely unprocessed model logits.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Protocol, Sequence

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prefix: TokenSequence
    max_new_tokens: int
    sampling: SamplingConfig
    seed: int
    request_id: str
    # Generation also ends right after any of these token sequences; a backend may ignore them.
    stop_sequences: tuple[TokenSequence, ...] = ()
    # Temperature of the full-support reference policy whose log-probabilities are also reported.
    reference_temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not all(self.stop_sequences):
            raise ValueError("stop sequences must be nonempty")
        if not (isfinite(self.reference_temperature) and self.reference_temperature > 0):
            raise ValueError("reference_temperature must be finite and positive")

    @property
    def reference_policy(self) -> SamplingConfig:
        return SamplingConfig(temperature=self.reference_temperature, eos_token_id=self.sampling.eos_token_id)


@dataclass(frozen=True, slots=True)
class SequenceSample:
    prefix: TokenSequence
    token_ids: TokenSequence
    token_logprobs: tuple[float, ...]
    policy_id: str
    model_id: str
    request_id: str
    finish_reason: str = "length"
    reference_token_logprobs: tuple[float, ...] | None = None
    reference_policy_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.token_logprobs):
            raise ValueError(
                "each sampled token must have one actual-policy log-probability"
            )
        if any(not isfinite(value) for value in self.token_logprobs):
            raise ValueError("actual-policy token log-probabilities must be finite")
        if (self.reference_token_logprobs is None) != (
            self.reference_policy_id is None
        ):
            raise ValueError(
                "reference token probabilities and their policy id must be provided together"
            )
        if self.reference_token_logprobs is not None and len(self.token_ids) != len(
            self.reference_token_logprobs
        ):
            raise ValueError(
                "each sampled token must have one reference-policy log-probability"
            )

    @property
    def logprob(self) -> float:
        return float(sum(self.token_logprobs))


@dataclass(frozen=True, slots=True)
class ScoreRequest:
    prefix: TokenSequence
    continuations: tuple[TokenSequence, ...]
    sampling: SamplingConfig | None = None


class AutoregressiveBackend(Protocol):
    """Minimal interface required by MH, conditional IS, and replay correction."""

    @property
    def model_id(self) -> str: ...

    def sample_batch(
        self, requests: Sequence[GenerationRequest]
    ) -> list[SequenceSample]: ...

    def score_batch(
        self, requests: Sequence[ScoreRequest]
    ) -> list[tuple[float, ...]]: ...
