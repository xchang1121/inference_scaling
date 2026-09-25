"""Sampling-algorithm kernels shared by autoregressive and diffusion models.

The kernels contain only target, weight and selection arithmetic. Model
families supply proposals, rollouts and probability scores through adapters.

- ``importance``: log-mean-exp weights, their normalization and categorical selection
- ``mh``: Metropolis--Hastings acceptance kernel
"""
