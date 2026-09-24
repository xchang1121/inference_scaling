"""Answer-agreement rewards and selection, independent of any dataset.

A dataset supplies only an ``AnswerRule``: the final answer of a text (``None``
when it has none) and whether two answers agree.  Callers supply ``decode``,
the text of a generated sequence that carries its answer.  Ties between equally
frequent answers go to the smallest answer, so they do not depend on request
order.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from inference_scaling.shared.types import TokenSequence

Decode = Callable[[TokenSequence, TokenSequence], str]


class AnswerRule(Protocol):
    def answer(self, text: str) -> Any | None: ...

    def same(self, left: Any, right: Any) -> bool: ...


def _tally(rule: AnswerRule, answers: Sequence[Any | None], groups: list[list[Any]]) -> list[list[Any]]:
    """Add answers to ``[representative, count]`` groups of agreeing answers."""
    for answer in answers:
        if answer is None:
            continue
        for group in groups:
            if rule.same(answer, group[0]):
                group[1] += 1
                break
        else:
            groups.append([answer, 1])
    return groups


def _mode(groups: list[list[Any]]) -> Any | None:
    if not groups:
        return None
    top = max(count for _, count in groups)
    return min(answer for answer, count in groups if count == top)


def modal_answer(rule: AnswerRule, answers: Sequence[Any | None]) -> Any | None:
    return _mode(_tally(rule, answers, []))


def consensus_index(rule: AnswerRule, texts: Sequence[str], scores: Sequence[float]) -> int:
    """Choose a text agreeing with the modal answer, then the highest score, then the earliest."""

    if not texts or len(texts) != len(scores):
        raise ValueError("texts and scores must be non-empty and equally sized")
    answers = [rule.answer(text) for text in texts]
    mode = modal_answer(rule, answers)
    eligible = [
        index
        for index, answer in enumerate(answers)
        if mode is None or (answer is not None and rule.same(answer, mode))
    ]
    return max(eligible, key=lambda index: (scores[index], -index))


class CumulativeConsensusReward:
    """Self-consistency over every sequence scored so far in one run.

    A sequence scores 1 when its answer agrees with the modal answer of all
    scored sequences, including itself.  The value depends on the other
    sequences, so it is a batch reward rather than a fixed per-sequence one.
    """

    def __init__(self, rule: AnswerRule, decode: Decode) -> None:
        self.rule = rule
        self.decode = decode
        self._groups: list[list[Any]] = []

    def __call__(self, prompt: TokenSequence, generated: Sequence[TokenSequence]) -> tuple[float, ...]:
        answers = [self.rule.answer(self.decode(prompt, tokens)) for tokens in generated]
        mode = _mode(_tally(self.rule, answers, self._groups))
        return tuple(
            float(mode is not None and answer is not None and self.rule.same(answer, mode))
            for answer in answers
        )


def frozen_consensus_reward(
    rule: AnswerRule, decode: Decode, pilots: Sequence[str]
) -> Callable[[TokenSequence, TokenSequence], float]:
    """Score 1 when a sequence agrees with the modal answer of fixed pilot texts."""

    reference = modal_answer(rule, [rule.answer(text) for text in pilots])

    def reward(prompt: TokenSequence, tokens: TokenSequence) -> float:
        answer = rule.answer(decode(prompt, tokens))
        return float(reference is not None and answer is not None and rule.same(answer, reference))

    return reward


def pilot_agreement_reward(
    rule: AnswerRule, decode: Decode, pilots: Sequence[str]
) -> Callable[[TokenSequence, TokenSequence], float]:
    """Score the fraction of fixed pilot texts whose answer agrees with the sequence."""

    if not pilots:
        raise ValueError("pilot agreement requires at least one pilot text")
    answers = [rule.answer(text) for text in pilots]

    def reward(prompt: TokenSequence, tokens: TokenSequence) -> float:
        answer = rule.answer(decode(prompt, tokens))
        agreeing = sum(
            answer is not None and pilot is not None and rule.same(answer, pilot) for pilot in answers
        )
        return agreeing / len(answers)

    return reward


__all__ = [
    "AnswerRule",
    "CumulativeConsensusReward",
    "Decode",
    "consensus_index",
    "frozen_consensus_reward",
    "modal_answer",
    "pilot_agreement_reward",
]
