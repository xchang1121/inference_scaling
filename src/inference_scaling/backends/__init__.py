"""Inference backends."""

from inference_scaling.backends.absorbing import AbsorbingEOSBackend
from inference_scaling.backends.batching import BatchingSnapshot, ContinuousBatchingBackend
from inference_scaling.backends.cache import ScoreCacheSnapshot, ScoreCachingBackend
from inference_scaling.backends.candidate_cache import CachedCandidateBackend
from inference_scaling.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.backends.transformers_backend import (
    SequenceScoreStatistics,
    TransformersBackend,
    TransformersBackendSnapshot,
)

__all__ = [
    "AbsorbingEOSBackend",
    "BatchingSnapshot",
    "CachedCandidateBackend",
    "ContinuousBatchingBackend",
    "ScoreCacheSnapshot",
    "ScoreCachingBackend",
    "SequenceScoreStatistics",
    "TabularAutoregressiveBackend",
    "TransformersBackend",
    "TransformersBackendSnapshot",
]
