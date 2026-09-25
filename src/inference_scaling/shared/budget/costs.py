"""Forward-token cost model for sequential block generation.

Requests that share a prefix in one batch prefill it once, as the backends do:
Transformers prefills each repeated prefix of a batch once and vLLM caches
prefixes. A step at prefix length ``P`` draws its fresh candidates as complete
outputs, so a candidate's block and its first completion come from one request:
the step prefills ``P`` once and decodes ``B + d`` tokens per candidate, where
``d`` is the expected length after the block. A candidate with ``K > 1``
completions prefills its own prefix ``P + B`` once more and decodes ``K - 1``
further completions. Every completion is scored ``reward_forward_passes`` times
at its full length.

Completions run until EOS, so ``d`` comes from ``expected_remaining``, the
expected number of tokens from the current prefix to EOS estimated from
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
) -> tuple[int, int, int, int]:
    """Return ``(shared_cost, candidate_cost, rollout_cost, branch_cost)`` for one block choice.

    ``M`` candidates with ``K`` completions each cost
    ``shared + M * (candidate + K * rollout) + (M * branch if K > 1 else 0)``.
    A block reaching the output limit is terminal: its candidates are complete
    outputs scored directly, with ``K = 0``.
    """
    prefix = prompt_length + generated_length
    remaining = _completion_length(generated_length, total_length, expected_remaining)
    if generated_length + block_size == total_length:
        return prefix, remaining + reward_forward_passes * (prefix + remaining), 0, 0
    # Early-EOS candidates are scored once rather than K times; K >= 1 covers them.
    after = max(1, remaining - block_size)
    return prefix, block_size, after + reward_forward_passes * (prefix + block_size + after), prefix + block_size


def completion_reserve(
    *,
    prompt_length: int,
    generated_length: int,
    total_length: int,
    expected_remaining: int,
    candidates: int,
    reward_forward_passes: int,
) -> int:
    """Expected cost of finishing from the prefix with ``candidates`` scored complete outputs."""
    prefix = prompt_length + generated_length
    remaining = _completion_length(generated_length, total_length, expected_remaining)
    return prefix + candidates * (remaining + reward_forward_passes * (prefix + remaining))


__all__ = ["block_costs", "completion_reserve"]
