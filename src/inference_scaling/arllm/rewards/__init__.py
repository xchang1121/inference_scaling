"""Autoregressive rewards computed from the model's own probabilities.

- ``SequenceLogProbabilityReward``: mean token log-probability (the ``logprob`` reward)
- ``ConsilienceReward``: confidence trajectory of the thinking segment (the ``consilience`` reward)
"""

from inference_scaling.arllm.rewards.intrinsic import ConsilienceReward, SequenceLogProbabilityReward

__all__ = ["ConsilienceReward", "SequenceLogProbabilityReward"]
