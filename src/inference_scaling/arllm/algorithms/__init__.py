"""Autoregressive sampling algorithms.

- ``conditional_is``: conditional IS on a kept complete sequence (the ``is`` kernel)
- ``joint_budget_is``: IS with block size B, candidates M and completions K planned under a forward-token budget
- ``mh``: suffix Metropolis-Hastings toward power and reward targets, with an optional frozen-history proposal
- ``candidates``: base-policy candidate blocks shared by the IS variants
- ``config``: validated algorithm settings
"""
