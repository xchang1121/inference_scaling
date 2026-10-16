"""Autoregressive sampling algorithms.

- ``conditional_is``: conditional IS on a kept complete sequence (the ``is`` kernel)
- ``mh``: suffix Metropolis-Hastings toward power and reward targets, with an optional frozen-history proposal
- ``candidates``: base-policy candidate blocks for conditional IS
- ``config``: validated algorithm settings
"""
