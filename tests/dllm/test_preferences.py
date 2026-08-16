from fractions import Fraction

from inference_scaling.dllm.preferences import select_verified_preference_pair


def test_verified_rollout_is_preferred_when_group_has_both_outcomes():
    pair = select_verified_preference_pair(
        candidate_texts=("work\n#### 12", "wrong\n#### 7"),
        gold_solution="gold\n#### 12",
        gold_answer=Fraction(12),
    )

    assert pair is not None
    assert pair.chosen == "work\n#### 12"
    assert pair.chosen_source == "verified_rollout:0"
    assert pair.rejected_candidate_index == 1


def test_public_training_solution_fills_missing_positive():
    pair = select_verified_preference_pair(
        candidate_texts=("wrong\n#### 7", "also wrong\n#### 8"),
        gold_solution="gold\n#### 12",
        gold_answer=Fraction(12),
    )

    assert pair is not None
    assert pair.chosen == "gold\n#### 12"
    assert pair.chosen_source == "public_training_solution"


def test_all_correct_group_is_omitted():
    pair = select_verified_preference_pair(
        candidate_texts=("first\n#### 12", "second\n#### 12"),
        gold_solution="gold\n#### 12",
        gold_answer=Fraction(12),
    )

    assert pair is None
