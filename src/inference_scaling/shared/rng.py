"""Order-independent random streams for asynchronous experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SeedStream:
    root_seed: int

    def derive(self, *path: object) -> int:
        if self.root_seed < 0:
            raise ValueError("root_seed must be non-negative")
        encoded_path = "\x1f".join(str(part) for part in path).encode("utf-8")
        digest = hashlib.blake2b(
            encoded_path,
            digest_size=8,
            key=self.root_seed.to_bytes(16, "little", signed=False),
        ).digest()
        return int.from_bytes(digest, "little", signed=False) & ((1 << 63) - 1)

    def generator(self, *path: object) -> np.random.Generator:
        return np.random.default_rng(self.derive(*path))

