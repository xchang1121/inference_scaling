"""The chosen reward bound to one problem.

A reward is a fixed per-sequence score r(x, y); with temperature tau the
sampling target is p(y | x) exp(r / tau). ``verifier`` and ``vote`` read the
answer text of a completion and are built here for every model family;
``logprob`` and ``consilience`` read the model's own probabilities and are
built by the family that owns the model.
"""

from __future__ import annotations

import hashlib
import random
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from inference_scaling.datasets.base import Dataset, Problem
from inference_scaling.shared.rewards.verifier import VerifierContext, build_verifier
from inference_scaling.shared.rewards.vote import pool_agreement_reward, vote_index
from inference_scaling.shared.types import TokenBatchReward, TokenSequence, pointwise

REWARDS = ("verifier", "vote", "logprob", "consilience")
AnswerText = Callable[[TokenSequence, TokenSequence], str]


@dataclass(frozen=True)
class Reward:
    temperature: float
    # Rewards of complete sequences of one problem.
    batch: TokenBatchReward
    # Base-model forward passes per scored sequence, charged by budgeted IS.
    forward_passes: int
    description: Mapping[str, Any]
    model: Any = field(default=None, compare=False)
    # The reward as a function of the generation policy's token log-probabilities, when it is one.
    from_logprobs: Callable[[Sequence[float]], float] | None = None

    def generated(self, prompt: TokenSequence, sequences: Sequence[TokenSequence],
                  logprobs: Sequence[Sequence[float]]) -> list[float]:
        """Rewards of generated sequences, read from their generation log-probabilities when they suffice."""

        if self.from_logprobs is not None:
            return [self.from_logprobs(values) for values in logprobs]
        return [float(value) for value in self.batch(prompt, sequences)]


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
        return Reward(temperature, memoized(
            lambda prompt, sequences: verifier.score_batch(prompt_text, [answer_text(prompt, tokens) for tokens in sequences])
        ), 0, verifier.describe())
    if kind != "vote":
        raise ValueError(f"{kind} is not a text reward")
    agreement = pool_agreement_reward(dataset, pool)
    grades = [dataset.grade(text, problem) for text in pool]
    return Reward(temperature, memoized(pointwise(lambda prompt, tokens: agreement(answer_text(prompt, tokens)))), 0,
                  {"pool": [{"answer": grade.answer, "correct": grade.correct} for grade in grades]})



def best_index(rule: Any, texts: Sequence[str], values: Sequence[float] | None, rng: random.Random) -> int:
    """Best-of-N: the majority answer without a reward, else the highest reward; ties go to the seeded RNG."""

    if values is None:
        return vote_index(rule, texts, rng)
    top = max(values)
    return rng.choice([index for index, value in enumerate(values) if value == top])


__all__ = ["REWARDS", "Reward", "best_index", "memoized", "text_reward"]
