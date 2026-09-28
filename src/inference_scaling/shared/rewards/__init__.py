"""Reward definitions shared by every model family.

- ``vote``: the verifier's agreement with the model's own answers
- ``consilience``: confidence-trajectory arithmetic of the Consilience score

Rewards computed from a model's own probabilities live with each model family
(``inference_scaling.arllm.rewards``).
"""
