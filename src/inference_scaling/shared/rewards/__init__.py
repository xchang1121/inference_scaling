"""Reward definitions shared by every model family.

- ``verifier``: configurable external verifiers and their token-level reward adapters
- ``consilience``: confidence-trajectory arithmetic of the Consilience score

Rewards computed from a model's own probabilities live with each model family
(``inference_scaling.arllm.rewards``); dataset-specific verifier plugins live in
``inference_scaling.shared.evaluation``.
"""

from inference_scaling.shared.rewards.consilience import ConfidenceWindows, confidence_windows
from inference_scaling.shared.rewards.verifier import (
    BatchTextVerifier,
    ConfiguredTrainingVerifierReward,
    ConfiguredVerifier,
    TextVerifier,
    TokenBatchReward,
    TokenReward,
    TokenVerifierReward,
    VerifierContext,
    VerifierInput,
    VerifierSpec,
    build_token_verifier_reward,
    build_verifier,
    load_verifier_table,
    replace_verifier_from_file,
    verifier_spec_from_config,
)

__all__ = [
    "BatchTextVerifier",
    "ConfidenceWindows",
    "ConfiguredTrainingVerifierReward",
    "ConfiguredVerifier",
    "TextVerifier",
    "TokenBatchReward",
    "TokenReward",
    "TokenVerifierReward",
    "VerifierContext",
    "VerifierInput",
    "VerifierSpec",
    "build_token_verifier_reward",
    "build_verifier",
    "confidence_windows",
    "load_verifier_table",
    "replace_verifier_from_file",
    "verifier_spec_from_config",
]
