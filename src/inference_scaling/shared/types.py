"""Model-independent data types."""

from collections.abc import Callable, Sequence

TokenSequence = tuple[int, ...]
# A fixed per-sequence reward r(prompt, completion) and its batched form.
TokenReward = Callable[[TokenSequence, TokenSequence], float]
TokenBatchReward = Callable[[TokenSequence, Sequence[TokenSequence]], Sequence[float]]

__all__ = ["TokenBatchReward", "TokenReward", "TokenSequence"]
