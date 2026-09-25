"""The autoregressive sampling policy: the model-level configuration of one request.

Algorithm settings (candidate counts, block sizes, horizons) live in
:mod:`inference_scaling.arllm.algorithms.config`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True, slots=True)
class TokenPenalty:
    """Lower the model's logits of fixed tokens by ``strength`` before any policy reads them.

    The penalized model p(v|x)·exp(−strength·1[v ∈ tokens]), renormalized, is
    then the model itself: sampling, reference and scoring probabilities all
    come from it, so every algorithm keeps an exact full-support base policy.
    """

    token_ids: tuple[int, ...]
    strength: float

    def __post_init__(self) -> None:
        require_positive("strength", self.strength)
        if not self.token_ids or min(self.token_ids) < 0:
            raise ValueError("the penalty needs non-negative token ids")

    @classmethod
    def from_words(cls, tokenizer: Any, words: Sequence[str], strength: float) -> TokenPenalty:
        """Each word after a space, lower-case and capitalized, and capitalized at a line start, where that form
        is one token that decodes back to it; a lower-case form without a space is mostly a piece of a longer word."""

        token_ids = set()
        for word in words:
            forms = (" " + word.lower(), " " + word.capitalize(), word.capitalize())
            encoded = [tokenizer.encode(form, add_special_tokens=False) for form in forms]
            single = {ids[0] for form, ids in zip(forms, encoded) if len(ids) == 1 and tokenizer.decode(ids) == form}
            if not single:
                raise ValueError(f"no form of {word!r} is a single token of this tokenizer")
            token_ids |= single
        return cls(tuple(sorted(token_ids)), float(strength))

    @property
    def penalty_id(self) -> str:
        return f"token_penalty={canonical_float(self.strength)}:{','.join(map(str, self.token_ids))}"


__all__ = ["SamplingConfig", "TokenPenalty"]
