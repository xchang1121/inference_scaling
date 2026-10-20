"""Conditional SIR on a kept complete sequence, with token-free step history.

Candidate zero reuses the current block, completion and reward. Optional sibling
replay keeps a bounded subset of the chosen block's other completions for the
next step. Batch and streaming execution share the same tree and selection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, TokenSequence
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import ConditionalISResult, ConditionalISStep, RolloutEvaluation, select
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
        candidates, forward_slots, sources = evaluate_candidates(self, state, step_index, seeds)
        carried = bool(state.token_ids)
        selected, kept = select(candidates, seeds, "conditional_is", step_index)
        candidate = candidates[selected]
        completion = candidate.rollouts[kept]
        reused = (min(len(state.siblings), self.config.candidate_count - int(carried))
                  if self.config.sibling_replay else 0)
        record = ConditionalISStep(
            state.fixed, candidates, selected, kept, carried,
            sum(len(item.rollouts) for item in candidates) - int(carried) - reused, forward_slots,
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
        retain = getattr(self.backend, "retain_statistics", None)
        if retain is not None:
            retain(self.prompt, updated.token_ids, *(fixed + sibling.token_ids for sibling in siblings))
        return record, updated


def run_conditional_is(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: ConditionalISConfig,
    reward: TokenBatchReward,
    seeds: SeedStream,
    *,
    sampling: SamplingConfig | None = None,
    deadline: float | None = None,
) -> ConditionalISResult:
    """CLI/service loop. A deadline is checked between steps, never within a candidate tree."""
    adapter = ConditionalISAdapter(backend=backend, prompt=prompt, config=config,
                                   sampling=sampling or SamplingConfig(), reward=reward)
    state = adapter.initial_state
    steps: list[ConditionalISStep] = []
    while not adapter.is_terminal(state):
        if steps and deadline is not None and time.monotonic() >= deadline:
            break
        record, state = adapter.step(state, len(steps), seeds)
        steps.append(record.compact())
        del record
    forward_slots = None if any(step.forward_token_slots is None for step in steps) else sum(
        step.forward_token_slots or 0 for step in steps)
    reason = ("deadline" if not adapter.is_terminal(state)
              else "eos" if state.token_ids[-1] == adapter.sampling.eos_token_id
              else "length" if len(state.token_ids) >= config.total_length else "stop")
    return ConditionalISResult(prompt, state.token_ids, tuple(steps), forward_slots, reason)
