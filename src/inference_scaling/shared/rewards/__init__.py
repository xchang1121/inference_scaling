"""Reward definitions shared by every model family.

- ``verifier``: external reward sources (dataset grader, Python factory, constant)
- ``vote``: answer votes and the frozen-pool agreement reward
- ``consilience``: confidence-trajectory arithmetic of the Consilience score

Rewards computed from a model's own probabilities live with each model family
(``inference_scaling.arllm.rewards``).
"""
