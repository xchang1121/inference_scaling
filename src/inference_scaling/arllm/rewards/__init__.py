"""Autoregressive rewards computed from the model's own probabilities.

Both reduce a token statistic (``TokenStatisticReward``), which generation records where the backend can:

- ``SelfCertaintyReward``: mean KL divergence from uniform over the vocabulary (the ``self_certainty`` reward)
- ``ConsilienceReward``: top-K confidence trajectory of the thinking segment (the ``consilience`` reward)
"""
