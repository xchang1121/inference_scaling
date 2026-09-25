"""Autoregressive execution backends.

- ``transformers_backend`` / ``vllm_backend``: model engines with exact scoring and counters
- ``loader``: the configured engine from the ``ar`` settings
- ``batching``: continuous batching across concurrent problems
- ``cache``, ``reference``: policy-preserving wrappers; ``stopping``: the thinking-scope stop
- ``tabular``: an exact small backend for tests
"""
