"""Inference backends."""

from inference_scaling.acceleration import (
    ActiveBatchSpeculationConfig,
    LowPriorityRunAheadBackend,
    RolloutTokenTree,
    SpeculationTier,
    StreamingRewardEvaluator,
)
from inference_scaling.backends.absorbing import AbsorbingEOSBackend
from inference_scaling.backends.batching import BatchingSnapshot, ContinuousBatchingBackend
from inference_scaling.backends.cache import ScoreCacheSnapshot, ScoreCachingBackend
from inference_scaling.backends.candidate_cache import CachedCandidateBackend
from inference_scaling.backends.loader import (
    BACKEND_CHOICES,
    close_backend,
    configured_backend,
    load_backend_from_config,
    set_backend_override,
)
from inference_scaling.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.backends.transformers_backend import (
    SequenceScoreStatistics,
    TransformersBackend,
    TransformersBackendSnapshot,
)
from inference_scaling.backends.vllm_backend import (
    AsyncVLLMBackend,
    VLLMBackend,
    VLLMBackendSnapshot,
)

__all__ = [
    "AbsorbingEOSBackend",
    "ActiveBatchSpeculationConfig",
    "AsyncVLLMBackend",
    "BACKEND_CHOICES",
    "BatchingSnapshot",
    "CachedCandidateBackend",
    "ContinuousBatchingBackend",
    "LowPriorityRunAheadBackend",
    "RolloutTokenTree",
    "ScoreCacheSnapshot",
    "ScoreCachingBackend",
    "SequenceScoreStatistics",
    "SpeculationTier",
    "StreamingRewardEvaluator",
    "TabularAutoregressiveBackend",
    "TransformersBackend",
    "TransformersBackendSnapshot",
    "VLLMBackend",
    "VLLMBackendSnapshot",
    "close_backend",
    "configured_backend",
    "load_backend_from_config",
    "set_backend_override",
]
