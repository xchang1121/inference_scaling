"""Autoregressive rewards and their configuration.

- ``intrinsic``: rewards computed from the model's own probabilities
  (Consilience confidence trajectory, mean sequence log-probability)
- ``factory``: builds those rewards and their temperatures from experiment configs
"""

from inference_scaling.arllm.rewards.factory import (
    MODEL_REWARD_SOURCES,
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
    "SequenceLogProbabilityReward",
    "model_reward_from_config",
    "reward_temperature_from_config",
]
