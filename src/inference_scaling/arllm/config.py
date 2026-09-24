"""The autoregressive sampling policy: the model-level configuration of one request.

Algorithm settings (candidate counts, block sizes, horizons) live in
:mod:`inference_scaling.arllm.algorithms.config`.
"""

from __future__ import annotations

from dataclasses import dataclass

from inference_scaling.shared.config import (
    canonical_float,
    require_positive,
    require_probability,
)


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """The complete stochastic policy used for one autoregressive request."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        require_positive("temperature", self.temperature)
        require_probability("top_p", self.top_p, include_zero=False)
        if self.top_k is not None:
            require_positive("top_k", self.top_k)
        if self.eos_token_id is not None and self.eos_token_id < 0:
            raise ValueError("eos_token_id must be non-negative")

    @property
    def policy_id(self) -> str:
        return (
            f"temperature={canonical_float(self.temperature)};"
            f"top_p={canonical_float(self.top_p)};"
            f"top_k={self.top_k};eos={self.eos_token_id}"
        )


__all__ = ["SamplingConfig"]
