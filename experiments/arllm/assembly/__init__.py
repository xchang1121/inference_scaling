"""Assembly shared by the AR experiment entry points.

Entry points in ``experiments.arllm`` own their command lines and records. The
modules here turn configuration into backends, rewards and algorithm calls:

- ``method_runners``: method name -> algorithm config, reward and call
- ``common``: prompts, backend loading and plain sampling for one problem
- ``reasoning_methods``: fixed-budget SIR/MH comparisons of the reasoning benchmark
- ``runtime``: model artifact validation and metadata
- ``scoped_execution``, ``request_reuse``: output-scope accounting and request replay
"""
