"""The chosen reward bound to one problem.

A reward is a fixed per-sequence score r(x, y); with temperature tau the
sampling target is p(y | x) exp(r / tau). ``verifier`` and ``vote`` read the
answer text of a completion and are built here for every model family;
``logprob`` and ``consilience`` read the model's own probabilities and are
built by the family that owns the model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from inference_scaling.datasets.base import Dataset, Problem
from inference_scaling.shared.rewards.verifier import VerifierContext, build_verifier
from inference_scaling.shared.rewards.vote import pool_agreement_reward
from inference_scaling.shared.types import TokenSequence

REWARDS = ("verifier", "vote", "logprob", "consilience")
AnswerText = Callable[[TokenSequence, TokenSequence], str]


@dataclass(frozen=True)
class Reward:
    temperature: float
    point: Callable[[TokenSequence, TokenSequence], float]
    batch: Callable[[TokenSequence, Sequence[TokenSequence]], Sequence[float]] | None
    # Base-model forward passes per scored sequence, charged by budgeted IS.
    forward_passes: int
    description: Mapping[str, Any]
    model: Any = field(default=None, compare=False)


def text_reward(
    kind: str,
    settings: Mapping[str, Any],
    *,
    dataset: Dataset,
    problem: Problem,
    prompt_text: str,
    answer_text: AnswerText,
    pool: Sequence[str] = (),
) -> Reward:
    """``verifier``: an external source; ``vote``: agreement with a frozen pool of independent samples."""

    temperature = float(settings["temperature"])
    if kind == "verifier":
        verifier = build_verifier(
            settings,
            context=VerifierContext(prompt_text, problem.answer, {"dataset": dataset.name, "problem_id": problem.id}),
            grade=lambda text: dataset.grade(text, problem),
        )
        return Reward(
            temperature,
            lambda prompt, tokens: verifier.score(prompt_text, answer_text(prompt, tokens)),
            lambda prompt, sequences: verifier.score_batch(prompt_text, [answer_text(prompt, tokens) for tokens in sequences]),
            0,
            verifier.describe(),
        )
    if kind != "vote":
        raise ValueError(f"{kind} is not a text reward")
    agreement = pool_agreement_reward(dataset, pool)
    grades = [dataset.grade(text, problem) for text in pool]
    return Reward(
        temperature,
        lambda prompt, tokens: agreement(answer_text(prompt, tokens)),
        None,
        0,
        {"pool": [{"answer": grade.answer, "correct": grade.correct} for grade in grades]},
    )


__all__ = ["REWARDS", "Reward", "text_reward"]
