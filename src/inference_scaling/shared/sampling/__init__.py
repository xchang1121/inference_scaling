"""Sampling-algorithm kernels shared by autoregressive and diffusion models.

The kernels contain only target, weight and selection arithmetic. Model
families supply proposals, rollouts and probability scores through adapters.

- ``importance``: rollout importance weights
- ``mh``: Metropolis--Hastings acceptance kernel
- ``stepwise``: finite-candidate SIR driver used by conditional IS
"""
