"""Independent block drafting, exact verification, and adapter distillation.

The implementation uses ordinary PyTorch operators. No upstream decoding or
training package is imported; references and derivations live in ALGORITHM.md.
"""

from .model import Decoder, ModelConfig
from .sampling import SamplingConfig

__all__ = ["Decoder", "ModelConfig", "SamplingConfig"]
