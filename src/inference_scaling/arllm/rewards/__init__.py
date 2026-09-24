"""Autoregressive rewards and their configuration.

- ``intrinsic``: rewards computed from the model's own probabilities
  (Consilience confidence trajectory, mean sequence log-probability, confidence statistics)
- ``factory``: every reward source by name; datasets supply only answer rules,
  pilot texts and verifier references
"""

from inference_scaling.arllm.rewards.factory import (
    MODEL_REWARD_SOURCES,
    REWARD_SOURCES,
    SequenceReward,
    build_reward,
    model_reward_from_config,
    reward_temperature_from_config,
)
from inference_scaling.arllm.rewards.intrinsic import (
    ConsilienceReward,
    SequenceLogProbabilityReward,
)

__all__ = [
    "ConsilienceReward",
    "MODEL_REWARD_SOURCES",
    "REWARD_SOURCES",
    "SequenceLogProbabilityReward",
    "SequenceReward",
    "build_reward",
    "model_reward_from_config",
    "reward_temperature_from_config",
]
