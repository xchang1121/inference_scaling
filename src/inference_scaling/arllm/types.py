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
class Draft:
    """Tokens that followed a request's prefix under its policy, with what their sampling recorded.

    A backend that samples by inverse CDF keeps a draft token exactly when the request's own uniform
    falls in the token's CDF interval, which is when plain generation would draw it, so the output is
    unchanged and the kept tokens need no model call.
    """

    token_ids: TokenSequence
    token_logprobs: tuple[float, ...]
    reference_token_logprobs: tuple[float, ...]
    token_cdf_bounds: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        lengths = {len(self.token_ids), len(self.token_logprobs), len(self.reference_token_logprobs),
                   len(self.token_cdf_bounds)}
        if len(lengths) != 1:
            raise ValueError("a draft needs its log-probabilities and CDF interval for every token")

    def after(self, count: int) -> Draft | None:
        """The draft past its first ``count`` tokens; ``None`` when nothing is left."""

        if count >= len(self.token_ids):
            return None
        return Draft(self.token_ids[count:], self.token_logprobs[count:], self.reference_token_logprobs[count:],
                     self.token_cdf_bounds[count:])


@dataclass(frozen=True, slots=True)
class LogWeightStop:
    """End a proposal as soon as it can no longer be accepted (exact early rejection).

    Token ``t`` adds ``min(0, scale * reference_logprob_t - logprob_t)`` to a running log-weight that
    starts at ``start``; generation ends once ``running - current < threshold``. The increments are never
    positive, so the complete proposal would fail the same test.
    """

    scale: float
    current: float
    threshold: float
    start: float = 0.0

    def advance(self, running: float, reference_logprob: float, logprob: float) -> float:
        return running + min(0.0, self.scale * reference_logprob - logprob)

    def rejects(self, running: float) -> bool:
        return running - self.current < self.threshold

    def weight(self, reference_logprobs: Sequence[float], logprobs: Sequence[float]) -> float:
        """The running log-weight after ``logprobs``, summed in the order generation sums it."""

        running = self.start
        for reference, logprob in zip(reference_logprobs, logprobs, strict=True):
            running = self.advance(running, reference, logprob)
        return running


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
    # Position of the first generated token in the seed's random stream: a request that continues
    # another one with the same seed reproduces it where the backend indexes its stream by position.
    uniform_offset: int = 0
    # Tokens to replay before generating; the output is the same with or without them.
    draft: Draft | None = None
    # End with finish_reason "rejected" once the output can no longer be accepted.
    log_weight_stop: LogWeightStop | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.seed < 0 or self.uniform_offset < 0:
            raise ValueError("seed and uniform_offset must be non-negative")
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
    # Per token, the policy's cumulative probabilities (below, through) the token in token-id order:
    # an inverse-CDF sampler draws the token exactly when its uniform u satisfies below < u <= through.
    token_cdf_bounds: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.token_logprobs):
            raise ValueError("each sampled token must have one actual-policy log-probability")
        if any(not isfinite(value) for value in self.token_logprobs):
            raise ValueError("actual-policy token log-probabilities must be finite")
        if (self.reference_token_logprobs is None) != (self.reference_policy_id is None):
            raise ValueError("reference token probabilities and their policy id must be provided together")
        for name in ("reference_token_logprobs", "token_cdf_bounds"):
            values = getattr(self, name)
            if values is not None and len(values) != len(self.token_ids):
                raise ValueError(f"{name} needs one entry per sampled token")

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
