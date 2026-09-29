"""The inference service: one AR model behind OpenAI Chat Completions and Anthropic Messages.

Each request reasons with think-only budgeted IS weighted by Consilience, then
answers once from the chosen thought. ``python -m inference_scaling.serve``
starts it with ``settings/inference.json`` (model, engine, IS grids and reward)
and ``settings/serve.json`` (server and reasoning-effort profiles).
"""
