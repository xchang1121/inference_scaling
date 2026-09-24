"""Reward definitions shared by every model family.

- ``verifier``: external reward sources (dataset grader, Python factory, constant)
- ``vote``: answer votes and the frozen-pool agreement reward
- ``consilience``: confidence-trajectory arithmetic of the Consilience score

Rewards computed from a model's own probabilities live with each model family
(``inference_scaling.arllm.rewards``).
"""

from inference_scaling.shared.rewards.consilience import ConfidenceWindows, confidence_windows
from inference_scaling.shared.rewards.verifier import VERIFIER_SOURCES, Verifier, VerifierContext, build_verifier
from inference_scaling.shared.rewards.vote import AnswerRule, answer_groups, pool_agreement_reward, vote_index

__all__ = [
    "AnswerRule",
    "ConfidenceWindows",
    "VERIFIER_SOURCES",
    "Verifier",
    "VerifierContext",
    "answer_groups",
    "build_verifier",
    "confidence_windows",
    "pool_agreement_reward",
    "vote_index",
]
