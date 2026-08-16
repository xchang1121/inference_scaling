"""Verified preference-pair construction for GSM8K VRPO."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Sequence

from inference_scaling.shared.evaluation import extract_numeric_answer


@dataclass(frozen=True, slots=True)
class VerifiedPreferencePair:
    chosen: str
    rejected: str
    chosen_source: str
    rejected_candidate_index: int


def select_verified_preference_pair(
    *,
    candidate_texts: Sequence[str],
    gold_solution: str,
    gold_answer: Fraction,
) -> VerifiedPreferencePair | None:
    """Select a correct/incorrect pair without reading evaluation data.

    A correct model rollout is preferred as the chosen completion.  If the
    sampled group contains no correct rollout, the public training solution is
    used as the chosen completion.  An all-correct group has no valid rejected
    completion and is omitted.
    """

    if not candidate_texts:
        raise ValueError("candidate_texts must be non-empty")
    predictions = [extract_numeric_answer(text) for text in candidate_texts]
    correct = [index for index, value in enumerate(predictions) if value == gold_answer]
    incorrect = [index for index, value in enumerate(predictions) if value != gold_answer]
    if not incorrect:
        return None
    if correct:
        chosen = candidate_texts[correct[0]]
        chosen_source = f"verified_rollout:{correct[0]}"
    else:
        chosen = gold_solution
        chosen_source = "public_training_solution"
    rejected_index = incorrect[0]
    rejected = candidate_texts[rejected_index]
    if chosen == rejected:
        raise RuntimeError("verified preference construction produced identical texts")
    return VerifiedPreferencePair(
        chosen=chosen,
        rejected=rejected,
        chosen_source=chosen_source,
        rejected_candidate_index=rejected_index,
    )


__all__ = ["VerifiedPreferencePair", "select_verified_preference_pair"]
