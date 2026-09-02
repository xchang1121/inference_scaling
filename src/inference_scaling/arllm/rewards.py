"""Autoregressive rewards derived from model probabilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Sequence

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, ScoreRequest
from inference_scaling.shared.types import TokenSequence


@dataclass(frozen=True, slots=True)
class SequenceLogProbabilityReward:
    """Return a scaled full-sequence ``log p(completion | prompt)``.

    This is a model-derived reward, not an external verifier.  The backend must
    support exact scoring under ``sampling``.  With scale ``c`` and reward
    temperature ``tau``, reward reweighting targets
    ``p(completion | prompt) ** (1 + c / tau)``.
    """

    backend: AutoregressiveBackend
    sampling: SamplingConfig | None = None
    scale: float = 1.0

    def __post_init__(self) -> None:
        if not isfinite(self.scale):
            raise ValueError("log-probability reward scale must be finite")

    def __call__(self, prompt: TokenSequence, completion: TokenSequence) -> float:
        return self.batch(prompt, (completion,))[0]

    def batch(
        self,
        prompt: TokenSequence,
        completions: Sequence[TokenSequence],
    ) -> tuple[float, ...]:
        if not completions:
            return ()
        scored = self.backend.score_batch(
            [
                ScoreRequest(prompt, (tuple(completion),), self.sampling)
                for completion in completions
            ]
        )
        if len(scored) != len(completions):
            raise RuntimeError("backend returned an invalid log-probability score batch")
        if any(
            len(token_scores) != len(completion)
            for token_scores, completion in zip(scored, completions, strict=True)
        ):
            raise RuntimeError("backend returned an invalid token score shape")
        return tuple(self.scale * float(sum(token_scores)) for token_scores in scored)

    def describe(self) -> dict[str, object]:
        return {
            "source": "model_sequence_log_probability",
            "model_id": self.backend.model_id,
            "policy_id": self.sampling.policy_id if self.sampling is not None else None,
            "scale": self.scale,
        }


__all__ = ["SequenceLogProbabilityReward"]
