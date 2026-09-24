from __future__ import annotations

import random
from fractions import Fraction
from types import SimpleNamespace

import pytest

from inference_scaling.datasets.base import Grade
from inference_scaling.shared.rewards.verifier import VerifierContext, build_verifier
from inference_scaling.shared.rewards.vote import answer_groups, pool_agreement_reward, vote_index


def _settings(source: str, **overrides) -> dict:
    settings = {
        "source": source,
        "dataset": {"correct": 3.0, "incorrect": -1.0, "unparseable": -2.0},
        "python": {"factory": None, "options": {}, "requires_reference": False},
        "constant": {"value": 0.25},
    }
    for key, value in overrides.items():
        settings[key] = {**settings[key], **value}
    return settings


def test_dataset_verifier_maps_the_grade_to_configured_values() -> None:
    grades = {"right": Grade("5", True, True), "wrong": Grade("4", True, False), "none": Grade(None, False, False)}
    verifier = build_verifier(_settings("dataset"), context=VerifierContext("question", "5"), grade=grades.__getitem__)
    assert verifier.score_batch("question", ["right", "wrong", "none"]) == (3.0, -1.0, -2.0)
    with pytest.raises(ValueError, match="grader"):
        build_verifier(_settings("dataset"), context=VerifierContext("question"), grade=None)


def test_constant_verifier_needs_no_dataset_or_reference() -> None:
    verifier = build_verifier(_settings("constant"), context=VerifierContext("any task"), grade=None)
    assert verifier.score("prompt", "completion") == 0.25
    assert verifier.describe() == {"source": "constant", "value": 0.25}


class LengthVerifier:
    def __init__(self, *, context: VerifierContext, scale: float) -> None:
        self.context, self.scale = context, scale

    def score(self, prompt: str, completion: str) -> float:
        return self.scale * len(completion)

    def score_batch(self, prompt: str, completions) -> list[float]:
        return [self.score(prompt, completion) for completion in completions]


def build_length_verifier(*, context: VerifierContext, scale: float = 1.0) -> LengthVerifier:
    return LengthVerifier(context=context, scale=scale)


def build_reference_callable(*, context: VerifierContext):
    return lambda _prompt, completion: float(completion == context.reference)


def build_nan_verifier(*, context: VerifierContext):
    return lambda _prompt, _completion: float("nan")


def test_python_factory_verifier_sees_the_reference_only_when_it_asks() -> None:
    settings = _settings("python", python={"factory": f"{__name__}:build_length_verifier", "options": {"scale": 2.0}})
    verifier = build_verifier(settings, context=VerifierContext("q", "secret"), grade=None)
    assert verifier.score_batch("q", ["ab", "abc"]) == (4.0, 6.0)
    assert verifier.describe()["options"] == {"scale": 2.0}

    hidden = _settings("python", python={"factory": f"{__name__}:build_reference_callable"})
    assert build_verifier(hidden, context=VerifierContext("q", "7"), grade=None).score("q", "7") == 0.0
    shown = _settings("python", python={"factory": f"{__name__}:build_reference_callable", "requires_reference": True})
    assert build_verifier(shown, context=VerifierContext("q", "7"), grade=None).score("q", "7") == 1.0
    with pytest.raises(ValueError, match="requires a reference"):
        build_verifier(shown, context=VerifierContext("q"), grade=None)


def test_verifier_rejects_bad_factories_and_nonfinite_rewards() -> None:
    with pytest.raises(ValueError, match="module:function"):
        build_verifier(_settings("python", python={"factory": "no_colon"}), context=VerifierContext("q"), grade=None)
    reserved = _settings("python", python={"factory": f"{__name__}:build_length_verifier", "options": {"context": 1}})
    with pytest.raises(ValueError, match="reserved"):
        build_verifier(reserved, context=VerifierContext("q"), grade=None)
    nan = build_verifier(_settings("python", python={"factory": f"{__name__}:build_nan_verifier"}),
                         context=VerifierContext("q"), grade=None)
    with pytest.raises(ValueError, match="non-finite"):
        nan.score("q", "x")
    with pytest.raises(ValueError, match="non-finite"):
        build_verifier(_settings("constant", constant={"value": float("inf")}), context=VerifierContext("q"), grade=None)


NUMERIC = SimpleNamespace(
    answer=lambda text: Fraction(text.split("####")[-1]) if "####" in text else None,
    same=lambda left, right: left == right,
)


def test_vote_picks_the_most_voted_answer_and_breaks_ties_at_random() -> None:
    texts = ["#### 2", "#### 3", "no answer", "#### 2.0"]
    assert answer_groups(NUMERIC, [NUMERIC.answer(text) for text in texts]) == [[0, 3], [1]]
    assert {vote_index(NUMERIC, texts, random.Random(seed)) for seed in range(20)} == {0, 3}
    tied = ["#### 1", "#### 2"]
    assert {vote_index(NUMERIC, tied, random.Random(seed)) for seed in range(50)} == {0, 1}
    assert vote_index(NUMERIC, ["x", "y"], random.Random(3)) in {0, 1}


def test_pool_agreement_is_the_fraction_of_agreeing_pool_answers() -> None:
    reward = pool_agreement_reward(NUMERIC, ["#### 2", "#### 3", "#### 2", "nothing"])
    assert reward("#### 2") == pytest.approx(0.5)
    assert reward("#### 3") == pytest.approx(0.25)
    assert reward("no answer") == 0.0
    with pytest.raises(ValueError, match="nonempty pool"):
        pool_agreement_reward(NUMERIC, [])
