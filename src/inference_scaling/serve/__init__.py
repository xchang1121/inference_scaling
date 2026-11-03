"""The inference service: one AR model behind OpenAI Chat Completions and Anthropic Messages.

Each request reasons with think-only IS weighted by the configured default reward, then
answers once from the chosen thought. ``python -m inference_scaling.serve``
starts it with ``settings/inference.json`` (shared generation settings and the ``serve`` section).
"""
