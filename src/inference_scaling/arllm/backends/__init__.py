"""Autoregressive execution backends.

- ``transformers_backend`` / ``vllm_backend``: model engines with exact scoring and counters
- ``loader``: the configured engine from the ``ar`` settings
- ``batching``: continuous batching across concurrent problems on a synchronous engine
- ``stopping``: the thinking-scope stop; ``statistics``: a reward's per-token statistic
- ``scheduling`` / ``streaming``: the shared admission queue and the streaming session

The exact finite backend the distributional tests enumerate lives in ``tests/tabular.py``.
"""
