"""Autoregressive rewards read from the model's own next-token distributions."""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Any, Literal, Sequence

from inference_scaling.arllm.types import ScoreRequest, TokenStatistic
from inference_scaling.shared.types import TokenSequence
from inference_scaling.shared.rewards.consilience import confidence_windows
from inference_scaling.shared.model.output import OutputParser


@dataclass(frozen=True, slots=True)
class TokenStatisticReward:
    """A reward that reduces a token statistic of the model over a span of the completion.

    A subclass picks the statistic's kind and the reduction. The backend's ``token_statistics`` supplies the
    values, from generation where it recorded them. The thinking scope scores a
    complete, nonempty thinking segment and otherwise the whole completion. The
    reward depends only on the sequence, so it is pointwise; an empty span has
    neutral reward zero.
    """

    # A backend with ``token_statistics``.
    backend: Any
    statistic: TokenStatistic
    thinking_format: OutputParser | None = None
    scope: Literal["thinking", "full"] = "thinking"

    def __post_init__(self) -> None:
        if self.scope not in {"thinking", "full"}:
            raise ValueError("the reward scope must be thinking or full")
        if self.scope == "thinking" and self.thinking_format is None:
            raise ValueError("the thinking scope needs a thinking format")

    def __call__(self, prompt: TokenSequence, completion: TokenSequence) -> float:
        return self.batch(prompt, (completion,))[0]

    def span(self, prompt: TokenSequence, completion: TokenSequence) -> tuple[int, int, str | None]:
        """The scored positions of ``completion``, and why the thinking scope fell back to the whole completion."""

        if self.scope == "full":
            return 0, len(completion), None
        assert self.thinking_format is not None
        eos = getattr(getattr(self.backend, "tokenizer", None), "eos_token_id", None)
        segments = self.thinking_format.split(prompt, completion, eos_token_id=eos)
        if not segments.has_complete_thinking:
            return 0, len(completion), segments.status
        assert segments.thinking_start is not None and segments.thinking_end is not None
        return segments.thinking_start, segments.thinking_end, None

    def fallback(self, prompt: TokenSequence, completion: TokenSequence) -> str | None:
        """Why the thinking scope scores the whole completion; no model forward."""

        return self.span(prompt, completion)[2]

    def reduce(self, values: Sequence[float]) -> float:
        raise NotImplementedError

    def batch(self, prompt: TokenSequence, completions: Sequence[TokenSequence]) -> tuple[float, ...]:
        spans = [self.span(prompt, tuple(completion))[:2] for completion in completions]
        scored = [index for index, (start, end) in enumerate(spans) if start < end]
        # A span's statistics come with the context before it, so the completion is read up to the span's end.
        rows = self.backend.token_statistics([ScoreRequest(tuple(prompt), tuple(
            tuple(completions[index][: spans[index][1]]) for index in scored))], self.statistic) if scored else []
        if len(rows) != len(scored) or any(len(row) != spans[index][1] for index, row in zip(scored, rows)):
            raise RuntimeError("backend returned invalid token statistics")
        rewards = [0.0] * len(completions)
        for index, row in zip(scored, rows):
            rewards[index] = self.reduce(row[spans[index][0] :])
        return tuple(rewards)

    def describe(self) -> dict[str, object]:
        return {
            "model_id": self.backend.model_id, "policy_id": self.statistic.policy.policy_id,
            "top_k": self.statistic.top_k, "scope": self.scope, "fallback": "full_sequence",
            "thinking_format": self.thinking_format.describe() if self.thinking_format is not None else None,
        }


@dataclass(frozen=True, slots=True)
class SelfCertaintyReward(TokenStatisticReward):
    """Self-Certainty (arXiv:2502.18581): the mean KL divergence of the next-token distribution from uniform.

    It rewards a model that was decided at every step over the whole
    vocabulary, whichever token it sampled; the mean keeps it independent of the length.
    """

    def __post_init__(self) -> None:
        TokenStatisticReward.__post_init__(self)
        if self.statistic.top_k is not None:
            raise ValueError("Self-Certainty reads the whole vocabulary, not a top-K confidence")

    def reduce(self, values: Sequence[float]) -> float:
        if not all(isfinite(value) for value in values):
            raise ValueError("Self-Certainty requires finite confidences")
        return fsum(values) / len(values)

    def describe(self) -> dict[str, object]:
        return {"source": "model_self_certainty", **TokenStatisticReward.describe(self)}


@dataclass(frozen=True, slots=True)
class ConsilienceReward(TokenStatisticReward):
    """Verifier-free confidence-trajectory reward of one generated sequence.

    The reward is the final window mean of the top-K confidence minus
    ``initial_penalty`` times the initial window mean, which omits the first
    ``skip_fraction`` of tokens.
    """

    window_fraction: float = 0.2
    window_tokens: int | None = None
    skip_fraction: float = 0.05
    initial_penalty: float = 3.0

    def __post_init__(self) -> None:
        TokenStatisticReward.__post_init__(self)
        if self.statistic.top_k is None:
            raise ValueError("Consilience reads the top-K confidence statistic")
        self.reduce((0.0,))

    def reduce(self, values: Sequence[float]) -> float:
        return confidence_windows(values, window_fraction=self.window_fraction, window_tokens=self.window_tokens,
                                  skip_fraction=self.skip_fraction, initial_penalty=self.initial_penalty).score

    def describe(self) -> dict[str, object]:
        return {
            "source": "model_consilience", **TokenStatisticReward.describe(self),
            "window_fraction": self.window_fraction, "window_tokens": self.window_tokens,
            "skip_fraction": self.skip_fraction, "initial_penalty": self.initial_penalty,
        }


__all__ = ["ConsilienceReward", "SelfCertaintyReward", "TokenStatisticReward"]
