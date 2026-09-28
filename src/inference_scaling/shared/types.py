"""Model-independent data types."""

from collections.abc import Callable, Sequence

TokenSequence = tuple[int, ...]
# Fixed per-sequence rewards r(prompt, completion) of a batch of complete sequences.
TokenBatchReward = Callable[[TokenSequence, Sequence[TokenSequence]], Sequence[float]]


def pointwise(reward: Callable[[TokenSequence, TokenSequence], float]) -> TokenBatchReward:
    """A batch reward that scores each sequence on its own."""

    return lambda prompt, sequences: [float(reward(prompt, sequence)) for sequence in sequences]


__all__ = ["TokenBatchReward", "TokenSequence", "pointwise"]
