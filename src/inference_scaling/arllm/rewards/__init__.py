"""Autoregressive rewards computed from the model's own probabilities.

Both reduce one token statistic (``TokenStatisticReward``), which generation records:

- ``SequenceLogProbabilityReward``: mean token log-probability (the ``logprob`` reward)
- ``ConsilienceReward``: confidence trajectory of the thinking segment (the ``consilience`` reward)
"""
