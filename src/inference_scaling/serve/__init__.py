"""The inference service: one AR model behind OpenAI Chat Completions and Anthropic Messages.

Each request reasons with fixed-shape think-only IS weighted by Consilience, then
answers once from the chosen thought. ``python -m inference_scaling.serve``
starts it with ``settings/inference.json`` (shared generation settings and the ``serve`` section).
"""
