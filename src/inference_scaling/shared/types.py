"""Model-independent data types."""

from collections.abc import Callable, Sequence

TokenSequence = tuple[int, ...]
# Fixed per-sequence rewards r(prompt, completion) of a batch of complete sequences.
TokenBatchReward = Callable[[TokenSequence, Sequence[TokenSequence]], Sequence[float]]
# The same for generated sequences, given each one's token log-probabilities under the generation policy.
GeneratedBatchReward = Callable[[TokenSequence, Sequence[TokenSequence], Sequence[Sequence[float]]], Sequence[float]]


def pointwise(reward: Callable[[TokenSequence, TokenSequence], float]) -> Callable[..., list[float]]:
    """A batch reward that scores each sequence on its own; further arguments are ignored."""

    return lambda prompt, sequences, *_: [float(reward(prompt, sequence)) for sequence in sequences]


__all__ = ["GeneratedBatchReward", "TokenBatchReward", "TokenSequence", "pointwise"]
