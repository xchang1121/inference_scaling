"""The chosen reward bound to one problem.

A reward is a fixed per-sequence score r(x, y); with temperature tau the
sampling target is p(y | x) exp(r / tau). The ``verifier`` reads the answer
text of a completion and is built here for every model family; ``logprob`` and
``consilience`` read the model's own probabilities and are built by the family
that owns the model.
"""

from __future__ import annotations

import hashlib
import random
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from inference_scaling.datasets.base import Dataset, Problem
from inference_scaling.shared.rewards.vote import pool_agreement_reward
from inference_scaling.shared.types import TokenBatchReward, TokenSequence, pointwise

REWARDS = ("verifier", "logprob", "consilience")
AnswerText = Callable[[TokenSequence, TokenSequence], str]


@dataclass(frozen=True)
class Reward:
    temperature: float
    # Rewards of complete sequences of one problem.
    batch: TokenBatchReward
    # Base-model forward passes per scored generated sequence, charged by budgeted IS.
    forward_passes: int
    description: Mapping[str, Any]
    model: Any = field(default=None, compare=False)


def memoized(batch: TokenBatchReward) -> TokenBatchReward:
    """Score each distinct sequence once; the reward belongs to one prompt."""

    values: dict[bytes, float] = {}

    def score(prompt: TokenSequence, sequences: Sequence[TokenSequence]) -> list[float]:
        # Digests keep long sequences out of the memo.
        keys = [hashlib.blake2b(array("q", tokens).tobytes(), digest_size=16).digest() for tokens in sequences]
        missing = {key: tokens for key, tokens in zip(keys, sequences, strict=True) if key not in values}
        if missing:
            values.update(zip(missing, map(float, batch(prompt, list(missing.values()))), strict=True))
        return [values[key] for key in keys]

    return score


def verifier_reward(
    settings: Mapping[str, Any],
    *,
    dataset: Dataset,
    problem: Problem,
    answer_text: AnswerText,
    sample_pool: Callable[[int], Sequence[str]] | None,
) -> Reward:
    """1 when the dataset's grader finds the answer correct (the oracle), else 0; or the vote.

    The vote, chosen by ``source = "vote"`` or by a dataset without reference
    answers, is the fraction of the model's own answers that agree: those of
    ``pool_size`` samples that ``sample_pool`` draws before the search, which
    keeps the reward a fixed function of the sequence, or, without
    ``sample_pool``, those of the scored batch, which makes Best-of-N a
    majority vote.
    """

    temperature = float(settings["temperature"])
    if settings["source"] == "dataset" and dataset.oracle:
        return Reward(temperature, memoized(pointwise(
            lambda prompt, tokens: float(dataset.grade(answer_text(prompt, tokens), problem).correct))), 0,
            {"source": "dataset"})
    if sample_pool is None:
        def majority(prompt: TokenSequence, sequences: Sequence[TokenSequence]) -> list[float]:
            texts = [answer_text(prompt, tokens) for tokens in sequences]
            return list(map(pool_agreement_reward(dataset, texts), texts))

        return Reward(temperature, majority, 0, {"source": "vote", "pool": "scored_batch"})
    pool = list(sample_pool(int(settings["pool_size"])))
    agreement = pool_agreement_reward(dataset, pool)
    return Reward(temperature, memoized(pointwise(lambda prompt, tokens: agreement(answer_text(prompt, tokens)))), 0,
                  {"source": "vote", "pool": [{"answer": grade.answer, "correct": grade.correct}
                                              for grade in (dataset.grade(text, problem) for text in pool)]})


def best_of_n(reward: Reward, dataset: Dataset, problem: Problem, prompt: TokenSequence,
              sequences: Sequence[TokenSequence], answer_text: AnswerText,
              rng: random.Random) -> tuple[TokenSequence, dict[str, Any], float]:
    """The sequence of highest reward, ties to the seeded RNG; the trace grades every candidate."""

    values = [float(value) for value in reward.batch(prompt, sequences)]
    top = max(values)
    chosen = rng.choice([index for index, value in enumerate(values) if value == top])
    grades = [dataset.grade(answer_text(prompt, tokens), problem) for tokens in sequences]
    return sequences[chosen], {"selected_index": chosen, "candidates": [
        {"answer": grade.answer, "correct": grade.correct, "tokens": len(tokens), "reward": value}
        for grade, tokens, value in zip(grades, sequences, values, strict=True)]}, values[chosen]


__all__ = ["REWARDS", "Reward", "best_of_n", "memoized", "verifier_reward"]
