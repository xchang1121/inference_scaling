"""Training entry: ``python -m training`` runs the stages listed in ``settings/training.json``.

- ``download``: pinned GSM8K splits and model snapshots, verified by hash
- ``grpo``: GRPO LoRA for the autoregressive model
- ``vrpo_preferences`` / ``vrpo``: preference pairs and the VRPO LoRA for LLaDA
"""
