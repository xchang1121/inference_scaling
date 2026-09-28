from __future__ import annotations

import random
from fractions import Fraction

import pytest

from inference_scaling.app.rewards import best_index, verifier_reward
from inference_scaling.datasets.base import Grade, Problem
from inference_scaling.shared.rewards.vote import pool_agreement_reward


class Numeric:
    """Answers after ``####``, graded against the problem's reference."""

    oracle = True

    @staticmethod
    def answer(text):
        return Fraction(text.split("####")[-1]) if "####" in text else None

    @staticmethod
    def same(left, right):
        return left == right

    def grade(self, text, problem):
        answer = self.answer(text)
        return Grade(None if answer is None else str(answer), answer is not None, answer == Fraction(problem.answer))


# Each one-token sequence decodes to one of these texts.
TEXTS = ("#### 2", "#### 3", "no answer", "#### 2.0")


def _reward(source, dataset=None, pool=None):
    return verifier_reward({"temperature": 0.1, "source": source, "pool_size": 3}, dataset=dataset or Numeric(),
                           problem=Problem("0", "q", "2"), answer_text=lambda _prompt, tokens: TEXTS[tokens[0]],
                           sample_pool=pool)


def test_the_oracle_grades_against_the_reference_answer() -> None:
    reward = _reward("dataset")
    assert reward.batch((), [(0,), (1,), (2,), (3,)]) == [1.0, 0.0, 0.0, 1.0]
    assert reward.description == {"source": "dataset"}


def test_the_vote_agrees_with_a_frozen_pool_or_with_the_scored_batch() -> None:
    drawn: list[int] = []
    reward = _reward("vote", pool=lambda size: drawn.append(size) or ["#### 2", "#### 3", "#### 2"])
    assert drawn == [3] and reward.batch((), [(0,), (1,), (2,)]) == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert [item["correct"] for item in reward.description["pool"]] == [True, False, True]
    # Without a pool the candidates vote among themselves, so Best-of-N picks a text of the majority answer.
    values = _reward("vote").batch((), [(0,), (1,), (2,), (3,)])
    assert values == pytest.approx([0.5, 0.25, 0.0, 0.5])
    assert {best_index(values, random.Random(seed)) for seed in range(20)} == {0, 3}


def test_a_dataset_without_reference_answers_votes() -> None:
    dataset = Numeric()
    dataset.oracle = False
    reward = _reward("dataset", dataset=dataset, pool=lambda size: ["#### 3"] * size)
    assert reward.description["source"] == "vote" and reward.batch((), [(1,), (0,)]) == [1.0, 0.0]


def test_pool_agreement_is_the_fraction_of_agreeing_pool_answers() -> None:
    reward = pool_agreement_reward(Numeric(), ["#### 2", "#### 3", "#### 2", "nothing"])
    assert reward("#### 2") == pytest.approx(0.5) and reward("#### 3") == pytest.approx(0.25)
    assert reward("no answer") == 0.0
    with pytest.raises(ValueError, match="nonempty pool"):
        pool_agreement_reward(Numeric(), [])
