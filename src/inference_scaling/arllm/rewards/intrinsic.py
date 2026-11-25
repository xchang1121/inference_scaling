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
    # Optional additive penalty applied in thinking scope when a completion does not close thinking cleanly.
    incomplete_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.scope not in {"thinking", "full"}:
            raise ValueError("the reward scope must be thinking or full")
        if self.scope == "thinking" and self.thinking_format is None:
            raise ValueError("the thinking scope needs a thinking format")
        if not isfinite(self.incomplete_penalty) or self.incomplete_penalty < 0:
            raise ValueError("incomplete_penalty must be finite and non-negative")

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
        span_rows = [self.span(prompt, tuple(completion)) for completion in completions]
        scored = [index for index, (start, end, _) in enumerate(span_rows) if start < end]
        # A span's statistics come with the context before it, so the completion is read up to the span's end.
        rows = self.backend.token_statistics([ScoreRequest(tuple(prompt), tuple(
            tuple(completions[index][: span_rows[index][1]]) for index in scored))], self.statistic) if scored else []
        if len(rows) != len(scored) or any(len(row) != span_rows[index][1] for index, row in zip(scored, rows)):
            raise RuntimeError("backend returned invalid token statistics")
        rewards = [0.0] * len(completions)
        for index, row in zip(scored, rows):
            value = self.reduce(row[span_rows[index][0] :])
            if self.incomplete_penalty > 0.0 and span_rows[index][2] is not None:
                value -= self.incomplete_penalty
            rewards[index] = value
        if self.incomplete_penalty > 0.0:
            for index, (start, end, reason) in enumerate(span_rows):
                if start >= end and reason is not None:
                    rewards[index] = -self.incomplete_penalty
        return tuple(rewards)

    def describe(self) -> dict[str, object]:
        description: dict[str, object] = {
            "model_id": self.backend.model_id, "policy_id": self.statistic.policy.policy_id,
            "top_k": self.statistic.top_k, "scope": self.scope, "fallback": "full_sequence",
            "thinking_format": self.thinking_format.describe() if self.thinking_format is not None else None,
        }
        if self.incomplete_penalty > 0.0:
            description["incomplete_penalty"] = self.incomplete_penalty
        return description

    def cached_prefix(self, prompt: TokenSequence, prefix: TokenSequence) -> float | None:
        """A running confidence proxy for scheduling, separate from the complete-sequence reward.

        Only already recorded statistics are read. Missing positions disable
        filtering; they never trigger a scoring request. An open thinking span
        is usable here, although the final reward still requires its completion.
        """
        start, end = 0, len(prefix)
        if self.scope == "thinking":
            assert self.thinking_format is not None
            segments = self.thinking_format.split(prompt, prefix)
            if segments.thinking_start is not None and segments.status in {"complete", "incomplete"}:
                start = segments.thinking_start
                end = segments.thinking_end if segments.thinking_end is not None else len(prefix)
        read = getattr(self.backend, "recorded_statistics", None)
        if read is None or start >= end:
            return None
        values = read(tuple(prompt), tuple(prefix[:end]), self.statistic)
        return None if values is None else self.reduce(values[start:])


@dataclass(frozen=True, slots=True)
class SelfCertaintyReward(TokenStatisticReward):
    """Self-Certainty (arXiv:2502.18581): KL divergence of the next-token distribution from uniform.

    Supports three length-invariant reductions over the per-token Self-Certainty
    trajectory:
    - ``"mean"`` (default): arithmetic mean across all scored tokens.
    - ``"tail_mean"``: lower-tail mean of the lowest ``tail_fraction`` tokens
      (after skipping ``skip_fraction`` opening tokens), focusing on reasoning
      uncertainty bottlenecks rather than boilerplate tokens.
    - ``"final_window"``: convex combination ``(1 - final_weight) * body_mean +
      final_weight * final_window_mean``, rewarding both overall clarity and
      late-stage convergence.
    """

    aggregation: Literal["mean", "tail_mean", "final_window"] = "mean"
    tail_fraction: float = 0.25
    window_fraction: float = 0.2
    final_weight: float = 0.5
    skip_fraction: float = 0.05

    def __post_init__(self) -> None:
        TokenStatisticReward.__post_init__(self)
        if self.statistic.top_k is not None:
            raise ValueError("Self-Certainty reads the whole vocabulary, not a top-K confidence")
        if self.aggregation not in {"mean", "tail_mean", "final_window"}:
            raise ValueError("Self-Certainty aggregation must be mean, tail_mean or final_window")
        if not isfinite(self.tail_fraction) or not 0 < self.tail_fraction <= 1:
            raise ValueError("Self-Certainty tail_fraction must be in (0, 1]")
        if not isfinite(self.window_fraction) or not 0 < self.window_fraction <= 1:
            raise ValueError("Self-Certainty window_fraction must be in (0, 1]")
        if not isfinite(self.final_weight) or not 0 <= self.final_weight <= 1:
            raise ValueError("Self-Certainty final_weight must be in [0, 1]")
        if not isfinite(self.skip_fraction) or not 0 <= self.skip_fraction < 1:
            raise ValueError("Self-Certainty skip_fraction must be in [0, 1)")

    def reduce(self, values: Sequence[float]) -> float:
        if not all(isfinite(value) for value in values):
            raise ValueError("Self-Certainty requires finite confidences")
        if self.aggregation == "mean":
            return fsum(values) / len(values)
        skipped = min(len(values) - 1, int(len(values) * self.skip_fraction))
        body = values[skipped:]
        if self.aggregation == "tail_mean":
            count = max(1, int(len(body) * self.tail_fraction))
            return fsum(sorted(body)[:count]) / count
        window = max(1, int(len(body) * self.window_fraction))
        body_mean = fsum(body) / len(body)
        final_mean = fsum(body[-window:]) / window
        return (1.0 - self.final_weight) * body_mean + self.final_weight * final_mean

    def describe(self) -> dict[str, object]:
        description: dict[str, object] = {"source": "model_self_certainty", **TokenStatisticReward.describe(self)}
        if self.aggregation != "mean":
            description["aggregation"] = self.aggregation
            description["skip_fraction"] = self.skip_fraction
            if self.aggregation == "tail_mean":
                description["tail_fraction"] = self.tail_fraction
            else:
                description["window_fraction"] = self.window_fraction
                description["final_weight"] = self.final_weight
        return description


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
