"""Infrastructure shared by autoregressive and diffusion language models."""

from inference_scaling.shared.config import RuntimeConfig
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence

__all__ = ["RuntimeConfig", "SeedStream", "TokenSequence"]
