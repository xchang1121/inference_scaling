"""The verifier's vote: agreement with the model's own answers, independent of any dataset.

A dataset supplies only an ``AnswerRule``: the final answer of a text (``None``
when it has none) and whether two answers agree. Unparseable texts cast no vote.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol


class AnswerRule(Protocol):
    def answer(self, text: str) -> Any | None: ...

    def same(self, left: Any, right: Any) -> bool: ...


def pool_agreement_reward(rule: AnswerRule, pool: Sequence[str]) -> Callable[[str], float]:
    """The fraction of a pool of texts whose answer agrees with a text's answer."""

    if not pool:
        raise ValueError("the vote needs a nonempty pool")
    answers = [rule.answer(text) for text in pool]

    def reward(text: str) -> float:
        answer = rule.answer(text)
        if answer is None:
            return 0.0
        return sum(other is not None and rule.same(answer, other) for other in answers) / len(answers)

    return reward


__all__ = ["AnswerRule", "pool_agreement_reward"]
