"""Autoregressive execution backends.

- ``transformers_backend`` / ``vllm_backend``: model engines with exact scoring and counters
- ``loader``: the configured engine from the ``ar`` settings
- ``batching``: continuous batching across concurrent problems
- ``reference``: the reference-temperature wrapper; ``stopping``: the thinking-scope stop
- ``tabular``: an exact small backend for tests
"""
