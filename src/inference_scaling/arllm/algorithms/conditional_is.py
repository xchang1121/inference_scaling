"""Conditional SIR on a kept complete sequence, with token-free step history.

Candidate zero reuses the current block, completion and reward. Batch and
streaming execution share the same tree and complete-weight selection.
Supports ``fixed``, ``two_stage``, and ``adaptive`` exploration schedules
parameterized externally by ``(M, K, B)``.
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
    CandidateSummary, ConditionalISResult, ConditionalISStep, RolloutSummary,
    resample_particles, select, select_metropolized,
)
from inference_scaling.shared.types import TokenBatchReward


@dataclass(frozen=True, slots=True)
class RetainedSequence:
    """Selected complete sequence; no older candidate pools."""

    token_ids: TokenSequence = ()
    reward: float = 0.0
    fixed: int = 0
    # An early first pool forecasts from observed prefixes, then discards them.
    next_candidate_count: int | None = None


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
        candidates, forward_slots, sources, evaluations, selection, forecast = evaluate_candidates(
            self, state, step_index, seeds)
        carried = bool(state.token_ids)
        if selection is not None:
            selected, kept = selection
        elif self.config.selection != "sir":
            selected, kept = select_metropolized(candidates, seeds, "conditional_is", step_index,
                                                group_blocks=self.config.selection == "block_metropolized")
        else:
            selected, kept = select(candidates, seeds, "conditional_is", step_index)
        candidate = candidates[selected]
        completion = candidate.rollouts[kept]
        width = (max(self.config.block_size, *(len(item.token_ids) for item in candidates))
                 if self.config.selection_ready else self.config.block_size)
        record = ConditionalISStep(
            state.fixed, candidates, selected, kept, carried,
            evaluations, forward_slots,
            (width, self.config.candidate_count, self.config.rollout_count),
            sources,
        )
        fixed = state.token_ids[:state.fixed] + candidate.token_ids
        updated = RetainedSequence(fixed + completion.token_ids, completion.reward, len(fixed), forecast)
        retain = getattr(self.backend, "retain_statistics", None)
        if retain is not None:
            retain(self.prompt, updated.token_ids)
        return record, updated


def _final_candidate_count(record: ConditionalISStep, fixed: int, config: ConditionalISConfig,
                           seeds: SeedStream) -> int:
    """Allocate a complete final pool or reuse the retained sequence without new samples.

    The decision uses only unused first-tree groups and an independent random draw.
    It changes finite-budget exploration, not the reward or completion-time policy.
    """
    observations = tuple(
        (index, len(candidate.rollouts),
         record.generated_length_before
         + (candidate.token_count if isinstance(candidate, CandidateSummary) else len(candidate.token_ids))
         + (leaf.token_count if isinstance(leaf, RolloutSummary) else len(leaf.token_ids)), leaf.log_weight)
        for index, candidate in enumerate(record.candidates) if index != record.selected_index
        for leaf in candidate.rollouts)
    return forecast_candidate_count(observations, fixed, config, seeds)


def forecast_candidate_count(observations: tuple[tuple[int, int, int, float], ...], fixed: int,
                             config: ConditionalISConfig, seeds: SeedStream) -> int:
    """Forecast from unused (group, planned K, observed length, log weight) rows.

    Complete and early first pools share the same allocation. Each planned
    leaf has one row; unfinished leaves supply only their observed prefix.
    Terminal one-leaf groups retain equal group mass. The result predicts
    generated work, not a hard token/FLOPs budget or an error guarantee.
    """
    length_sum = work_sum = 0
    groups: dict[int, list[float]] = {}
    for index, planned, observed_length, log_weight in observations:
        groups.setdefault(index, []).append(log_weight)
        multiplicity = max(1, config.rollout_count // max(1, planned))
        length = max(0, observed_length - fixed)
        rounds = (length + config.block_size - 1) // config.block_size
        suffix_sum = rounds * length - config.block_size * rounds * (rounds - 1) // 2
        work = ((config.candidate_count - 1) * suffix_sum
                + config.candidate_count * (config.rollout_count - 1) * (suffix_sum - length))
        length_sum += multiplicity * length
        work_sum += multiplicity * work
    full = work_sum // length_sum if length_sum else 0
    if config.final_pool_limit is not None:
        full = min(full, config.final_pool_limit)
    if config.exploration_schedule == "adaptive":
        # Bound Stage-2 fanout to at most M-1 fresh suffixes (total pool <= M)
        # so long proofs stay within a predictable ~2M-scale compute envelope.
        full = min(full, max(1, config.candidate_count - 1))
    unused = list(groups.values())
    if not full or len(unused) < 2:
        # A single auxiliary group cannot estimate variation between prefixes.
        return full
    peak = max(value for group in unused for value in group)
    if not isfinite(peak):
        return full
    means = [fsum(exp(value - peak) for value in group) / len(group) for group in unused]
    seconds = [fsum(exp(2 * (value - peak)) for value in group) / len(group) for group in unused]
    mean, second = fsum(means) / len(unused), fsum(seconds) / len(unused)
    dispersion = max(0., min(1., 1 - mean * mean / second))
    if config.exploration_schedule == "adaptive":
        # Skip Stage 2 when unused Stage-1 candidates are already tightly clustered;
        # otherwise scale Stage-2 fresh suffixes smoothly in [1, M-1] by dispersion.
        if dispersion < 0.15:
            return 0
        return max(1, ceil(full * dispersion))
    minimum = min(full, config.candidate_count - 1)
    expected = minimum + ceil((full - minimum) * dispersion)
    return full if float(seeds.generator("conditional_is", "final_pool_gate").random()) < expected / full else 0


def _unused_dispersion(record: ConditionalISStep, *, exclude_indices: set[int] | None = None) -> float | None:
    """Normalized variance dispersion in [0, 1] across unused candidate groups, or None if <2 groups."""
    excluded = {record.selected_index} if exclude_indices is None else exclude_indices
    unused = [
        [leaf.log_weight for leaf in candidate.rollouts]
        for index, candidate in enumerate(record.candidates)
        if index not in excluded and candidate.rollouts
    ]
    if len(unused) < 2:
        return None
    peak = max(value for group in unused for value in group)
    if not isfinite(peak):
        return None
    means = [fsum(exp(value - peak) for value in group) / len(group) for group in unused]
    seconds = [fsum(exp(2 * (value - peak)) for value in group) / len(group) for group in unused]
    mean, second = fsum(means) / len(unused), fsum(seconds) / len(unused)
    if second <= 0.0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - mean * mean / second))


def _common_prefix(sequences: list[TokenSequence]) -> TokenSequence:
    if not sequences:
        return ()
    limit = min(len(seq) for seq in sequences)
    end = 0
    first = sequences[0]
    while end < limit and all(seq[end] == first[end] for seq in sequences[1:]):
        end += 1
    return first[:end]


def _progressive_next_plan(
    base_config: ConditionalISConfig,
    *,
    next_step: int,
    fixed_len: int,
    retained_len: int,
    dispersion: float | None,
) -> tuple[int, int, int, bool] | None:
    """Return ``(block_size, fresh_count, rollout_count, is_final)`` or ``None`` to early-stop and reuse."""
    rem_total = base_config.total_length - fixed_len
    if rem_total <= 0 or retained_len <= fixed_len:
        return None
    if base_config.max_stages is not None and next_step >= base_config.max_stages:
        return None
    is_final = base_config.max_stages is not None and next_step >= base_config.max_stages - 1
    if is_final:
        next_block = rem_total
    else:
        geom = max(1, int(round(base_config.block_size * (base_config.block_growth ** next_step))))
        next_block = min(geom, rem_total)
        if next_block >= rem_total:
            is_final = True

    rem_retained = retained_len - fixed_len
    disp = 0.5 if dispersion is None else dispersion
    # Early-stop refinement when dispersion is already low and no long remaining tail warrants probing.
    if disp < 0.12 and (is_final or rem_retained <= next_block):
        return None

    max_fresh = max(1, base_config.candidate_count - 1)
    if base_config.final_pool_limit is not None:
        max_fresh = min(max_fresh, base_config.final_pool_limit)
    fresh = 1 if disp < 0.12 else min(max_fresh, max(1, ceil(max_fresh * disp)))
    next_k = 1 if is_final else max(1, ceil(base_config.rollout_count / (2 ** next_step)))
    return next_block, fresh, next_k, is_final


def _run_particle_progressive(
    adapter: ConditionalISAdapter,
    base_config: ConditionalISConfig,
    seeds: SeedStream,
    *,
    deadline: float | None = None,
    on_commit: Callable[[TokenSequence], None] | None = None,
) -> ConditionalISResult:
    """Progressive multi-stage Particle Conditional IS (Block-SMC) for ``beam_particles > 1``."""
    from inference_scaling.arllm.algorithms.candidates import evaluate_candidates, evaluate_particle_pool

    p_count = base_config.beam_particles
    steps: list[ConditionalISStep] = []
    published_len = 0

    # Stage 0 uses evaluate_candidates directly so pilot_gate still exits easy problems at 1x cost.
    candidates, forward_slots, sources, evaluations, selection, _ = evaluate_candidates(
        adapter, adapter.initial_state, 0, seeds,
    )
    if selection is not None:
        # Pilot gate early exit (single clean short trajectory).
        sel_idx, sel_k = selection
        chosen_cand = candidates[sel_idx]
        chosen_roll = chosen_cand.rollouts[sel_k]
        tokens = chosen_cand.token_ids + chosen_roll.token_ids
        step0 = ConditionalISStep(
            0, candidates, sel_idx, sel_k, False, evaluations, forward_slots,
            (base_config.block_size, len(candidates), base_config.rollout_count), sources,
        )
        steps.append(step0.compact())
        if on_commit is not None and tokens:
            on_commit(tokens)
        reason = ("eos" if tokens[-1] == adapter.sampling.eos_token_id
                  else "length" if len(tokens) >= base_config.total_length else "stop")
        return ConditionalISResult(adapter.prompt, tokens, tuple(steps), forward_slots, reason)

    is_stage0_final = base_config.max_stages == 1 or base_config.block_size >= base_config.total_length
    if is_stage0_final:
        if base_config.selection != "sir":
            sel_idx, sel_k = select_metropolized(
                candidates, seeds, "conditional_is", 0,
                group_blocks=base_config.selection == "block_metropolized",
            )
        else:
            sel_idx, sel_k = select(candidates, seeds, "conditional_is", 0)
        chosen_cand = candidates[sel_idx]
        chosen_roll = chosen_cand.rollouts[sel_k]
        tokens = chosen_cand.token_ids + chosen_roll.token_ids
        step0 = ConditionalISStep(
            0, candidates, sel_idx, sel_k, False, evaluations, forward_slots,
            (base_config.block_size, base_config.candidate_count, base_config.rollout_count), sources,
        )
        steps.append(step0.compact())
        if on_commit is not None and tokens:
            on_commit(tokens)
        reason = ("eos" if tokens[-1] == adapter.sampling.eos_token_id
                  else "length" if len(tokens) >= base_config.total_length else "stop")
        return ConditionalISResult(adapter.prompt, tokens, tuple(steps), forward_slots, reason)

    # Resample P surviving particles from Stage 0's candidate tree.
    picked = resample_particles(candidates, p_count, seeds, "conditional_is", 0)
    particles: list[RetainedSequence] = []
    for c_idx, r_idx in picked:
        cand = candidates[c_idx]
        roll = cand.rollouts[r_idx]
        seq = cand.token_ids + roll.token_ids
        particles.append(RetainedSequence(seq, roll.reward, len(cand.token_ids)))

    # Pick representative index for step summary (highest-weight picked particle).
    rep_pos = max(range(len(picked)), key=lambda i: candidates[picked[i][0]].rollouts[picked[i][1]].log_weight)
    rep_c, rep_r = picked[rep_pos]
    step0 = ConditionalISStep(
        0, candidates, rep_c, rep_r, False, evaluations, forward_slots,
        (base_config.block_size, base_config.candidate_count, base_config.rollout_count), sources,
    )
    dispersion = _unused_dispersion(step0, exclude_indices={c_idx for c_idx, _ in picked})
    steps.append(step0.compact())

    retain = getattr(adapter.backend, "retain_statistics", None)
    if retain is not None:
        retain(adapter.prompt, particles[0].token_ids, *(p.token_ids for p in particles[1:]))
    if on_commit is not None:
        common = _common_prefix([p.token_ids[:p.fixed] for p in particles])
        if len(common) > published_len:
            published_len = len(common)
            on_commit(common)

    stopped_by_deadline = False
    while any(p.fixed < len(p.token_ids) for p in particles):
        if deadline is not None and time.monotonic() >= deadline:
            stopped_by_deadline = True
            break
        next_step = len(steps)
        min_fixed = min(p.fixed for p in particles)
        max_len = max(len(p.token_ids) for p in particles)
        plan = _progressive_next_plan(
            base_config,
            next_step=next_step,
            fixed_len=min_fixed,
            retained_len=max_len,
            dispersion=dispersion,
        )
        if plan is None:
            break
        next_block, fresh, next_k, is_final = plan
        total_cands = max(p_count, p_count + fresh)
        joint_cands, origins, stage_slots, stage_evals = evaluate_particle_pool(
            adapter,
            particles,
            next_step,
            seeds,
            block_width=next_block,
            total_candidates=total_cands,
            rollout_count=next_k,
            is_final=is_final,
            base_block_size=base_config.block_size,
        )

        # Check whether all joint candidates are already terminal after advancing by next_block.
        all_after_terminal = is_final or all(
            len(origins[j][1]) + len(cand.token_ids) >= len(origins[j][1]) + len(cand.token_ids) + len(roll.token_ids)
            for j, cand in enumerate(joint_cands)
            for roll in cand.rollouts
        )
        if is_final or all_after_terminal:
            if base_config.selection != "sir":
                win_c, win_r = select_metropolized(
                    joint_cands, seeds, "conditional_is", next_step,
                    group_blocks=base_config.selection == "block_metropolized",
                )
            else:
                win_c, win_r = select(joint_cands, seeds, "conditional_is", next_step)
            win_cand = joint_cands[win_c]
            win_roll = win_cand.rollouts[win_r]
            win_tokens = origins[win_c][1] + win_cand.token_ids + win_roll.token_ids
            particles = [RetainedSequence(win_tokens, win_roll.reward, len(win_tokens))]
            stage_record = ConditionalISStep(
                min_fixed, joint_cands, win_c, win_r, True, stage_evals, stage_slots,
                (next_block, len(joint_cands), next_k), None,
            )
            steps.append(stage_record.compact())
            if retain is not None:
                retain(adapter.prompt, win_tokens)
            break

        picked = resample_particles(joint_cands, p_count, seeds, "conditional_is", next_step)
        next_particles: list[RetainedSequence] = []
        for c_idx, r_idx in picked:
            cand = joint_cands[c_idx]
            roll = cand.rollouts[r_idx]
            fixed_seq = origins[c_idx][1] + cand.token_ids
            next_particles.append(RetainedSequence(fixed_seq + roll.token_ids, roll.reward, len(fixed_seq)))
        particles = next_particles

        rep_pos = max(range(len(picked)), key=lambda i: joint_cands[picked[i][0]].rollouts[picked[i][1]].log_weight)
        rep_c, rep_r = picked[rep_pos]
        stage_record = ConditionalISStep(
            min_fixed, joint_cands, rep_c, rep_r, True, stage_evals, stage_slots,
            (next_block, len(joint_cands), next_k), None,
        )
        dispersion = _unused_dispersion(stage_record, exclude_indices={c_idx for c_idx, _ in picked})
        steps.append(stage_record.compact())

        if retain is not None:
            retain(adapter.prompt, particles[0].token_ids, *(p.token_ids for p in particles[1:]))
        if on_commit is not None:
            common = _common_prefix([p.token_ids[:p.fixed] for p in particles])
            if len(common) > published_len:
                published_len = len(common)
                on_commit(common)

    # Final selection among surviving particles if loop exited early (all terminal, low dispersion, or deadline).
    if len(particles) == 1:
        winner = particles[0]
    else:
        from inference_scaling.shared.sampling.conditional_is import RolloutEvaluation, weigh as _weigh

        final_cands = _weigh(
            [p.token_ids for p in particles],
            [[RolloutEvaluation((), p.reward, p.reward / base_config.reward_temperature)] for p in particles],
        )
        win_idx, _ = select(final_cands, seeds, "conditional_is", len(steps), "particle_final")
        winner = particles[win_idx]
        if retain is not None:
            retain(adapter.prompt, winner.token_ids)

    if on_commit is not None and len(winner.token_ids) > published_len:
        on_commit(winner.token_ids)
    total_slots = None if any(s.forward_token_slots is None for s in steps) else sum(
        s.forward_token_slots or 0 for s in steps
    )
    reason = ("deadline" if stopped_by_deadline
              else "eos" if winner.token_ids[-1] == adapter.sampling.eos_token_id
              else "length" if len(winner.token_ids) >= base_config.total_length else "stop")
    return ConditionalISResult(adapter.prompt, winner.token_ids, tuple(steps), total_slots, reason)


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
    on_commit: Callable[[TokenSequence], None] | None = None,
) -> ConditionalISResult:
    """Report only irrevocable prefixes; observation cannot select a candidate.

    A deadline is checked between steps, never within a candidate tree.
    The callback receives cumulative committed tokens, without retaining history.
    """
    adapter = ConditionalISAdapter(backend=backend, prompt=prompt, config=config,
                                   sampling=sampling or SamplingConfig(), reward=reward, prefix_reward=prefix_reward)
    if config.exploration_schedule == "progressive" and config.beam_particles > 1:
        return _run_particle_progressive(adapter, config, seeds, deadline=deadline, on_commit=on_commit)

    base_config = config
    state = adapter.initial_state
    steps: list[ConditionalISStep] = []
    stopped_by_deadline = False
    while not adapter.is_terminal(state):
        if steps and deadline is not None and time.monotonic() >= deadline:
            stopped_by_deadline = True
            break
        record, state = adapter.step(state, len(steps), seeds)
        if on_commit is not None:
            on_commit(state.token_ids[:state.fixed])
        if (not steps and config.exploration_schedule in {"two_stage", "adaptive"}
                and config.candidate_count > 1 and not adapter.is_terminal(state)):
            count = (state.next_candidate_count if state.next_candidate_count is not None
                     else _final_candidate_count(record, state.fixed, config, seeds))
            adapter = replace(adapter, config=replace(config, candidate_count=1 + count,
                              rollout_count=1, block_size=config.total_length - state.fixed))
        elif base_config.exploration_schedule == "progressive" and not adapter.is_terminal(state):
            next_step = len(steps) + 1
            plan = _progressive_next_plan(
                base_config,
                next_step=next_step,
                fixed_len=state.fixed,
                retained_len=len(state.token_ids),
                dispersion=_unused_dispersion(record),
            )
            if plan is None:
                steps.append(record.compact())
                del record
                # Low dispersion early-stop: commit the complete retained trajectory directly.
                state = replace(state, fixed=len(state.token_ids))
                if on_commit is not None:
                    on_commit(state.token_ids)
                break
            next_block, fresh, next_k, _ = plan
            adapter = replace(
                adapter,
                config=replace(
                    base_config,
                    candidate_count=1 + fresh,
                    rollout_count=next_k,
                    block_size=next_block,
                ),
            )
        steps.append(record.compact())
        del record
    if on_commit is not None and state.fixed < len(state.token_ids):
        # A deadline returns the selected complete rollout, including its suffix.
        on_commit(state.token_ids)
    forward_slots = None if any(step.forward_token_slots is None for step in steps) else sum(
        step.forward_token_slots or 0 for step in steps)
    reason = ("deadline" if stopped_by_deadline or not adapter.is_terminal(state)
              else "eos" if state.token_ids[-1] == adapter.sampling.eos_token_id
              else "length" if len(state.token_ids) >= base_config.total_length else "stop")
    return ConditionalISResult(adapter.prompt, state.token_ids, tuple(steps), forward_slots, reason)
