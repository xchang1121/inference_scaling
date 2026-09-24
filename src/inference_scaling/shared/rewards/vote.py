"""Answer votes, independent of any dataset.

A dataset supplies only an ``AnswerRule``: the final answer of a text (``None``
when it has none) and whether two answers agree. Unparseable texts cast no vote.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any, Protocol


class AnswerRule(Protocol):
    def answer(self, text: str) -> Any | None: ...

    def same(self, left: Any, right: Any) -> bool: ...


def answer_groups(rule: AnswerRule, answers: Sequence[Any | None]) -> list[list[int]]:
    """Indices of mutually agreeing answers, one group per distinct answer."""

    groups: list[list[int]] = []
    for index, answer in enumerate(answers):
        if answer is None:
            continue
        for group in groups:
            if rule.same(answer, answers[group[0]]):
                group.append(index)
                break
        else:
            groups.append([index])
    return groups


def vote_index(rule: AnswerRule, texts: Sequence[str], rng: random.Random) -> int:
    """A text whose answer has the most votes; ties go to a uniformly random top-voted text."""

    if not texts:
        raise ValueError("a vote needs at least one text")
    groups = answer_groups(rule, [rule.answer(text) for text in texts])
    if not groups:
        return rng.randrange(len(texts))
    top = max(len(group) for group in groups)
    return rng.choice([index for group in groups if len(group) == top for index in group])


def pool_agreement_reward(rule: AnswerRule, pool: Sequence[str]) -> Callable[[str], float]:
    """The fraction of a frozen pool of independent samples whose answer agrees with a text's answer."""

    if not pool:
        raise ValueError("the vote reward needs a nonempty pool")
    answers = [rule.answer(text) for text in pool]

    def reward(text: str) -> float:
        answer = rule.answer(text)
        if answer is None:
            return 0.0
        return sum(other is not None and rule.same(answer, other) for other in answers) / len(answers)

    return reward


__all__ = ["AnswerRule", "answer_groups", "pool_agreement_reward", "vote_index"]
