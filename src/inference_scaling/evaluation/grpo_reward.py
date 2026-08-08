"""Training reward that exactly matches the GSM8K inference scorer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from inference_scaling.evaluation.gsm8k import extract_numeric_answer


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence) and completion:
        message = completion[-1]
        if isinstance(message, dict) and "content" in message:
            return str(message["content"])
    raise TypeError(f"unsupported GRPO completion value: {type(completion).__name__}")


@dataclass
class ExactNumericReward:
    """Binary correctness reward with observable rollout accounting."""

    calls: int = 0
    completions: int = 0
    completion_tokens: int = 0
    parseable: int = 0
    correct: int = 0

    def __call__(
        self,
        completions: Sequence[Any],
        gold_answer: Sequence[str],
        completion_ids: Sequence[Sequence[int]] | None = None,
        **_: Any,
    ) -> list[float]:
        if len(completions) != len(gold_answer):
            raise ValueError("GRPO completions and gold answers have different lengths")
        rewards: list[float] = []
        for completion, gold in zip(completions, gold_answer, strict=True):
            predicted = extract_numeric_answer(_completion_text(completion))
            expected = extract_numeric_answer(f"#### {gold}")
            valid = predicted is not None
            is_correct = valid and predicted == expected
            self.parseable += int(valid)
            self.correct += int(is_correct)
            rewards.append(float(is_correct))
        self.calls += 1
        self.completions += len(completions)
        if completion_ids is not None:
            self.completion_tokens += sum(len(tokens) for tokens in completion_ids)
        return rewards

    def snapshot(self, *, num_generations: int) -> dict[str, int | float]:
        return {
            "reward_calls": self.calls,
            "generated_completions": self.completions,
            "generated_prompt_groups": self.completions // num_generations,
            "generated_completion_tokens": self.completion_tokens,
            "parseable_completions": self.parseable,
            "correct_completions": self.correct,
            "observed_rollout_accuracy": (
                self.correct / self.completions if self.completions else 0.0
            ),
        }
