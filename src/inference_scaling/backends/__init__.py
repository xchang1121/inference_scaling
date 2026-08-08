"""Inference backends."""

from inference_scaling.backends.batching import BatchingSnapshot, ContinuousBatchingBackend
from inference_scaling.backends.tabular import TabularAutoregressiveBackend

__all__ = ["BatchingSnapshot", "ContinuousBatchingBackend", "TabularAutoregressiveBackend"]
