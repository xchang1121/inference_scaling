"""Reserved forward-token cost model for sequential block generation.

Costs are conservative request-level reservations: cold prefix + decode for a
candidate block, and a full-length rollout plus reward scoring for each
completion. They are not measured FLOPs; early EOS does not refund them.
"""

from __future__ import annotations


def block_costs(
    *,
    prompt_length: int,
    generated_length: int,
    total_length: int,
    block_size: int,
    reward_forward_passes: int,
) -> tuple[int, int]:
    """Return ``(candidate_cost, rollout_cost)`` for one block choice.

    All samples use the same model. Reward callbacks must fit the declared number
    of full-sequence scoring passes; auxiliary models require a separate cost model.
    A block that reaches ``total_length`` is terminal: its candidates are scored
    directly and it has no rollout cost.
    """
    full = max(1, prompt_length + total_length)
    candidate = max(1, prompt_length + generated_length + block_size)
    scoring = reward_forward_passes * full
    if generated_length + block_size == total_length:
        return candidate + scoring, 0
    # Early-EOS candidates are scored once rather than K times; K >= 1 covers them.
    return candidate, full + scoring


def completion_reserve(
    *,
    prompt_length: int,
    total_length: int,
    candidates: int,
    reward_forward_passes: int,
) -> int:
    """Cost of finishing from any prefix with ``candidates`` full-length samples."""
    return candidates * max(1, prompt_length + total_length) * (1 + reward_forward_passes)


__all__ = ["block_costs", "completion_reserve"]
