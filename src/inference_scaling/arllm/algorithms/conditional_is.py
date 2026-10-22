"""Conditional SIR on a kept complete sequence, with token-free step history.

Candidate zero reuses the current block, completion and reward. Optional sibling
replay keeps a bounded subset of the chosen block's other completions for the
next step. Batch and streaming execution share the same tree and selection.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from math import ceil, exp, fsum, isfinite

from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, TokenSequence
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import (
    ConditionalISResult, ConditionalISStep, RolloutEvaluation, select, select_metropolized,
)
from inference_scaling.shared.types import TokenBatchReward


@dataclass(frozen=True, slots=True)
class RetainedSequence:
    """Selected sequence and optional same-prefix siblings; no older candidate pools."""

    token_ids: TokenSequence = ()
    reward: float = 0.0
    fixed: int = 0
    # Suffixes relative to token_ids[:fixed], in uniformly random order.
    siblings: tuple[RolloutEvaluation, ...] = ()


@dataclass(frozen=True, kw_only=True)
class ConditionalISAdapter:
    backend: AutoregressiveBackend
    prompt: TokenSequence
    config: ConditionalISConfig
    sampling: SamplingConfig
    reward: TokenBatchReward
    # A cached prefix proxy; None means no usable observation, never a model rescore.
    prefix_reward: Callable[[TokenSequence, TokenSequence], float | None] | None = None

    @property
    def initial_state(self) -> RetainedSequence:
        return RetainedSequence()

    def is_terminal(self, state: RetainedSequence) -> bool:
        return bool(state.token_ids) and state.fixed >= len(state.token_ids)

    def step(self, state: RetainedSequence, step_index: int,
             seeds: SeedStream) -> tuple[ConditionalISStep, RetainedSequence]:
        """Return the current tree for immediate inspection, and the selected state."""
        from inference_scaling.arllm.algorithms.candidates import evaluate_candidates

        if self.is_terminal(state):
            raise ValueError("cannot advance a terminal generation state")
        candidates, forward_slots, sources, evaluations = evaluate_candidates(self, state, step_index, seeds)
        carried = bool(state.token_ids)
        choose = (select_metropolized if self.config.exploration_schedule == "single_pool"
                  and not self.config.weighted_completion_quota else select)
        selected, kept = choose(candidates, seeds, "conditional_is", step_index)
        candidate = candidates[selected]
        completion = candidate.rollouts[kept]
        record = ConditionalISStep(
            state.fixed, candidates, selected, kept, carried,
            evaluations, forward_slots,
            (self.config.block_size, self.config.candidate_count, self.config.rollout_count),
            sources,
        )
        fixed = state.token_ids[:state.fixed] + candidate.token_ids
        siblings: tuple[RolloutEvaluation, ...] = ()
        if self.config.sibling_replay and completion.token_ids:
            available = [item for index, item in enumerate(candidate.rollouts) if index != kept]
            count = min(len(available), self.config.candidate_count - 1)
            if count:
                order = seeds.generator("conditional_is", step_index, "siblings").permutation(len(available))
                siblings = tuple(available[int(index)] for index in order[:count])
        updated = RetainedSequence(fixed + completion.token_ids, completion.reward, len(fixed), siblings)
        if self.config.exploration_schedule == "single_pool":
            # Every candidate was completed before selection; commit the selected
            # completion directly instead of executing a reuse-only final step.
            updated = replace(updated, fixed=len(updated.token_ids))
        retain = getattr(self.backend, "retain_statistics", None)
        if retain is not None:
            retain(self.prompt, updated.token_ids, *(fixed + sibling.token_ids for sibling in siblings))
        return record, updated


def _consolidated_candidate_count(record: ConditionalISStep, fixed: int, config: ConditionalISConfig) -> int:
    """Fresh suffixes affordable under the unselected groups' fixed-block work forecast.

    Groups have equal mass, including terminal groups with a single leaf. The
    forecast uses no information from the selected group or the next pool.
    It predicts generated tokens, not a hard token/FLOPs or concurrency limit.
    """
    length_sum = work_sum = 0
    for index, candidate in enumerate(record.candidates):
        if index == record.selected_index:
            continue
        multiplicity = config.rollout_count // len(candidate.rollouts)
        for leaf in candidate.rollouts:
            length = max(0, record.generated_length_before + len(candidate.token_ids) + len(leaf.token_ids) - fixed)
            rounds = (length + config.block_size - 1) // config.block_size
            suffix_sum = rounds * length - config.block_size * rounds * (rounds - 1) // 2
            work = ((config.candidate_count - 1) * suffix_sum
                    + config.candidate_count * (config.rollout_count - 1) * (suffix_sum - length))
            length_sum += multiplicity * length
            work_sum += multiplicity * work
    return work_sum // length_sum if length_sum else 0


def _final_candidate_count(record: ConditionalISStep, fixed: int, config: ConditionalISConfig,
                           seeds: SeedStream) -> int:
    """Allocate a complete final pool or reuse the retained sequence without new samples.

    The decision uses only unused first-tree groups and an independent random draw.
    It changes finite-budget exploration, not the reward or completion-time policy.
    """
    full = _consolidated_candidate_count(record, fixed, config)
    if config.final_pool_limit is not None:
        full = min(full, config.final_pool_limit)
    unused = [group for index, group in enumerate(record.candidates) if index != record.selected_index]
    if not full or len(unused) < 2:
        # A single auxiliary group cannot estimate variation between prefixes.
        return full
    peak = max(leaf.log_weight for group in unused for leaf in group.rollouts)
    if not isfinite(peak):
        return full
    means = [fsum(exp(leaf.log_weight - peak) for leaf in group.rollouts) / len(group.rollouts)
             for group in unused]
    seconds = [fsum(exp(2 * (leaf.log_weight - peak)) for leaf in group.rollouts) / len(group.rollouts)
               for group in unused]
    mean, second = fsum(means) / len(unused), fsum(seconds) / len(unused)
    dispersion = max(0., min(1., 1 - mean * mean / second))
    minimum = min(full, config.candidate_count - 1)
    expected = minimum + ceil((full - minimum) * dispersion)
    return full if float(seeds.generator("conditional_is", "final_pool_gate").random()) < expected / full else 0


def run_conditional_is(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: ConditionalISConfig,
    reward: TokenBatchReward,
    seeds: SeedStream,
    *,
    sampling: SamplingConfig | None = None,
    deadline: float | None = None,
    prefix_reward: Callable[[TokenSequence, TokenSequence], float | None] | None = None,
) -> ConditionalISResult:
    """CLI/service loop. A deadline is checked between steps, never within a candidate tree."""
    adapter = ConditionalISAdapter(backend=backend, prompt=prompt, config=config,
                                   sampling=sampling or SamplingConfig(), reward=reward, prefix_reward=prefix_reward)
    state = adapter.initial_state
    steps: list[ConditionalISStep] = []
    while not adapter.is_terminal(state):
        if steps and deadline is not None and time.monotonic() >= deadline:
            break
        record, state = adapter.step(state, len(steps), seeds)
        if (not steps and config.exploration_schedule == "two_stage"
                and config.candidate_count > 1 and not adapter.is_terminal(state)):
            count = _final_candidate_count(record, state.fixed, config, seeds)
            adapter = replace(adapter, config=replace(config, candidate_count=1 + count,
                              rollout_count=1, block_size=config.total_length - state.fixed))
        steps.append(record.compact())
        del record
    forward_slots = None if any(step.forward_token_slots is None for step in steps) else sum(
        step.forward_token_slots or 0 for step in steps)
    reason = ("deadline" if not adapter.is_terminal(state)
              else "eos" if state.token_ids[-1] == adapter.sampling.eos_token_id
              else "length" if len(state.token_ids) >= config.total_length else "stop")
    return ConditionalISResult(prompt, state.token_ids, tuple(steps), forward_slots, reason)
