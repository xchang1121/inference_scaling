"""Forward-token cost model for sequential block generation.

A rollout or a completion runs until EOS, so its cost depends on how long the
remaining output actually is. The planners price it with ``expected_remaining``,
the expected number of tokens from the current prefix to EOS estimated from
observed completions. The output limit ``total_length`` only caps generation:
it enters a cost when the expected completion would run into it, and a block
that reaches it is terminal. Planned costs are therefore independent of the
output limit whenever that limit does not bind.
"""

from __future__ import annotations


def _completion_length(generated_length: int, total_length: int, expected_remaining: int) -> int:
    """Expected tokens from the prefix to EOS, never beyond the output limit."""
    return min(max(1, expected_remaining), total_length - generated_length)


def block_costs(
    *,
    prompt_length: int,
    generated_length: int,
    total_length: int,
    block_size: int,
    reward_forward_passes: int,
    expected_remaining: int,
) -> tuple[int, int]:
    """Return ``(candidate_cost, rollout_cost)`` for one block choice.

    A candidate costs its cold prefix plus decoded block. A rollout continues it
    until EOS and is scored ``reward_forward_passes`` times. A block reaching the
    output limit is terminal: candidates run to EOS and are scored directly.
    All samples use the same model; auxiliary models need a separate cost model.
    """
    prefix = prompt_length + generated_length
    remaining = _completion_length(generated_length, total_length, expected_remaining)
    if generated_length + block_size == total_length:
        full = max(1, prefix + remaining)
        return full * (1 + reward_forward_passes), 0
    # Early-EOS candidates are scored once rather than K times; K >= 1 covers them.
    full = max(1, prefix + max(remaining, block_size))
    return max(1, prefix + block_size), full * (1 + reward_forward_passes)


def completion_reserve(
    *,
    prompt_length: int,
    generated_length: int,
    total_length: int,
    expected_remaining: int,
    candidates: int,
    reward_forward_passes: int,
) -> int:
    """Expected cost of finishing from the prefix with ``candidates`` scored completions."""
    remaining = _completion_length(generated_length, total_length, expected_remaining)
    return candidates * max(1, prompt_length + generated_length + remaining) * (1 + reward_forward_passes)


__all__ = ["block_costs", "completion_reserve"]
