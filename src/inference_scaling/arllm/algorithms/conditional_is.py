"""Converged multi-stage Block-SMC Conditional IS parameterized by ``(M, K, S)``.

Candidate 0 reuses the retained complete sequence and reward without rescoring.
External inputs fix only candidate width ``M``, Stage-0 lookahead ``K``, and
maximum stage count ``S``; internal scheduling handles:
- Stage-0 hardware-page-aligned probing with ``B_0 // 2`` pilot escalation;
- observed-length-adaptive geometric checkpoints snapped to hardware KV pages;
- decaying lookahead ``K_s = max(1, ceil(K / 2^s))`` and dispersion-gated fanout;
- automatic 2-particle Block-SMC (systematic resampling + exact parent-mass
  conservation) when ``S >= 3`` and ``M >= 4``, degrading seamlessly to single-chain
  search when ``S <= 2`` or ``M < 4``;
- block-Metropolized final selection with zero early-selection prefix bias.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from math import ceil, exp, fsum, isfinite
from statistics import median

from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, TokenSequence
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import (
    CandidateSummary,
    ConditionalISResult,
    ConditionalISStep,
    RolloutEvaluation,
    RolloutSummary,
    resample_particles,
    select_metropolized,
    weigh,
)
from inference_scaling.shared.types import TokenBatchReward


@dataclass(frozen=True, slots=True)
class RetainedSequence:
    """Selected complete sequence; no older candidate pools."""

    token_ids: TokenSequence = ()
    reward: float = 0.0
    fixed: int = 0


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

    def step(
        self,
        state: RetainedSequence,
        step_index: int,
        seeds: SeedStream,
        *,
        block_width: int | None = None,
        candidate_count: int | None = None,
        rollout_count: int | None = None,
        is_final: bool = False,
    ) -> tuple[ConditionalISStep, RetainedSequence]:
        """Run one single-chain IS stage and return ``(step_record, next_state)``."""
        from inference_scaling.arllm.algorithms.candidates import evaluate_candidates

        if self.is_terminal(state):
            raise ValueError("cannot advance a terminal generation state")
        m = self.config.candidate_count if candidate_count is None else candidate_count
        k_plan = self.config.rollout_count if rollout_count is None else rollout_count
        rem = self.config.total_length - state.fixed
        width = rem if is_final else min(
            self.config.stage0_block_size if block_width is None else block_width,
            rem,
        )
        candidates, forward_slots, sources, evaluations, pilot_exit = evaluate_candidates(
            self,
            state,
            step_index,
            seeds,
            block_width=width,
            candidate_count=m,
            rollout_count=k_plan,
            is_final=is_final,
        )
        carried = bool(state.token_ids)
        if pilot_exit:
            selected, kept = 0, 0
        else:
            selected, kept = select_metropolized(candidates, seeds, "conditional_is", step_index, group_blocks=True)
        candidate = candidates[selected]
        completion = candidate.rollouts[kept]
        plan_m = len(candidates) if pilot_exit else m
        record = ConditionalISStep(
            state.fixed,
            candidates,
            selected,
            kept,
            carried,
            evaluations,
            forward_slots,
            (width, plan_m, k_plan),
            sources,
        )
        fixed = state.token_ids[:state.fixed] + candidate.token_ids
        updated = RetainedSequence(fixed + completion.token_ids, completion.reward, len(fixed))
        retain = getattr(self.backend, "retain_statistics", None)
        if retain is not None:
            retain(self.prompt, updated.token_ids)
        return record, updated


def _unused_observations(
    record: ConditionalISStep,
    fixed_len: int,
    *,
    exclude_indices: set[int] | None = None,
) -> tuple[float | None, int | None]:
    """Return ``(dispersion, median_unused_remaining_len)`` from unused candidate groups.

    Reading only unused groups keeps stage-width adaptation and fanout strictly
    conditionally independent of the chosen trajectory's remaining suffix.
    """
    excluded = {record.selected_index} if exclude_indices is None else exclude_indices
    unused_weights: list[list[float]] = []
    unused_rems: list[int] = []
    for index, candidate in enumerate(record.candidates):
        if index in excluded or not candidate.rollouts:
            continue
        unused_weights.append([leaf.log_weight for leaf in candidate.rollouts])
        cand_len = candidate.token_count if isinstance(candidate, CandidateSummary) else len(candidate.token_ids)
        for leaf in candidate.rollouts:
            leaf_len = leaf.token_count if isinstance(leaf, RolloutSummary) else len(leaf.token_ids)
            obs_rem = record.generated_length_before + cand_len + leaf_len - fixed_len
            if obs_rem > 0:
                unused_rems.append(obs_rem)

    median_rem = int(round(median(unused_rems))) if unused_rems else None
    if len(unused_weights) < 2:
        return None, median_rem
    peak = max(value for group in unused_weights for value in group)
    if not isfinite(peak):
        return None, median_rem
    means = [fsum(exp(value - peak) for value in group) / len(group) for group in unused_weights]
    seconds = [fsum(exp(2.0 * (value - peak)) for value in group) / len(group) for group in unused_weights]
    mean = fsum(means) / len(unused_weights)
    second = fsum(seconds) / len(unused_weights)
    if second <= 0.0:
        return 0.0, median_rem
    return max(0.0, min(1.0, 1.0 - mean * mean / second)), median_rem


def _snap_block_width(raw_block: int, target_rem: int, quantum: int, rem_total: int) -> tuple[int, bool]:
    """Snap an intermediate geometric block width to the hardware KV-page granularity.

    Returns ``(block_width, should_seal)``. When the remaining chain is already
    within one hardware page quantum (for real horizons ``quantum >= 64``), seals
    directly to avoid wasteful micro-stages.
    """
    if quantum < 64:
        clamped = max(1, min(raw_block, rem_total))
        return clamped, clamped >= rem_total
    if target_rem <= quantum:
        return rem_total, True
    half_q = max(1, quantum // 2)
    if raw_block >= quantum:
        snapped = max(quantum, int(round(raw_block / quantum)) * quantum)
    else:
        snapped = half_q
    if snapped >= rem_total:
        return rem_total, True
    return snapped, False


def _next_stage_plan(
    config: ConditionalISConfig,
    *,
    next_step: int,
    fixed_len: int,
    retained_len: int,
    dispersion: float | None,
    observed_rem: int | None,
) -> tuple[int, int, int, bool] | None:
    """Return ``(block_width, fresh_count, rollout_count, is_final)`` or ``None`` to early-stop."""
    rem_total = config.total_length - fixed_len
    rem_stages = config.stages - next_step
    if rem_stages <= 0 or rem_total <= 0 or retained_len <= fixed_len:
        return None

    rem_retained = retained_len - fixed_len
    if rem_stages == 1:
        next_block, is_final = rem_total, True
    else:
        target_rem = observed_rem if (observed_rem is not None and observed_rem > 0) else rem_retained
        geom_denom = (2 ** rem_stages) - 1
        raw_block = max(1, int(round(target_rem / geom_denom)))
        next_block, is_final = _snap_block_width(raw_block, target_rem, config.stage0_block_size, rem_total)

    disp = 0.5 if dispersion is None else dispersion
    # Early-stop refinement when unused candidates are tightly clustered and no long tail remains.
    if dispersion is not None and dispersion < 0.12 and (is_final or rem_retained <= next_block):
        return None

    max_fresh = max(1, config.candidate_count - 1)
    if config.final_pool_limit is not None:
        max_fresh = min(max_fresh, config.final_pool_limit)
    fresh = 1 if (dispersion is not None and dispersion < 0.12) else min(max_fresh, max(1, ceil(max_fresh * disp)))
    next_k = 1 if is_final else max(1, ceil(config.rollout_count / (2 ** next_step)))
    return next_block, fresh, next_k, is_final


def _common_prefix(sequences: list[TokenSequence]) -> TokenSequence:
    if not sequences:
        return ()
    limit = min(len(seq) for seq in sequences)
    end = 0
    first = sequences[0]
    while end < limit and all(seq[end] == first[end] for seq in sequences[1:]):
        end += 1
    return first[:end]


def _run_particle_smc(
    adapter: ConditionalISAdapter,
    config: ConditionalISConfig,
    seeds: SeedStream,
    *,
    deadline: float | None = None,
    on_commit: Callable[[TokenSequence], None] | None = None,
) -> ConditionalISResult:
    """Multi-particle Block-SMC execution for ``config.beam_particles > 1``."""
    from inference_scaling.arllm.algorithms.candidates import evaluate_candidates, evaluate_particle_pool

    p_count = config.beam_particles
    stage0_width = config.stage0_block_size
    steps: list[ConditionalISStep] = []
    published_len = 0

    candidates, forward_slots, sources, evaluations, pilot_exit = evaluate_candidates(
        adapter,
        adapter.initial_state,
        0,
        seeds,
        block_width=stage0_width,
        candidate_count=config.candidate_count,
        rollout_count=config.rollout_count,
        is_final=config.stages == 1 or stage0_width >= config.total_length,
    )
    if pilot_exit or config.stages == 1 or stage0_width >= config.total_length:
        sel_idx, sel_k = (0, 0) if pilot_exit else select_metropolized(
            candidates, seeds, "conditional_is", 0, group_blocks=True,
        )
        chosen_cand = candidates[sel_idx]
        chosen_roll = chosen_cand.rollouts[sel_k]
        tokens = chosen_cand.token_ids + chosen_roll.token_ids
        step0 = ConditionalISStep(
            0,
            candidates,
            sel_idx,
            sel_k,
            False,
            evaluations,
            forward_slots,
            (stage0_width, len(candidates) if pilot_exit else config.candidate_count, config.rollout_count),
            sources,
        )
        steps.append(step0.compact())
        if on_commit is not None and tokens:
            on_commit(tokens)
        reason = (
            "eos" if tokens[-1] == adapter.sampling.eos_token_id
            else "length" if len(tokens) >= config.total_length else "stop"
        )
        return ConditionalISResult(adapter.prompt, tokens, tuple(steps), forward_slots, reason)

    picked = resample_particles(candidates, p_count, seeds, "conditional_is", 0)
    particles: list[RetainedSequence] = []
    for c_idx, r_idx in picked:
        cand = candidates[c_idx]
        roll = cand.rollouts[r_idx]
        seq = cand.token_ids + roll.token_ids
        particles.append(RetainedSequence(seq, roll.reward, len(cand.token_ids)))

    rep_pos = max(range(len(picked)), key=lambda i: candidates[picked[i][0]].rollouts[picked[i][1]].log_weight)
    rep_c, rep_r = picked[rep_pos]
    step0 = ConditionalISStep(
        0,
        candidates,
        rep_c,
        rep_r,
        False,
        evaluations,
        forward_slots,
        (stage0_width, config.candidate_count, config.rollout_count),
        sources,
    )
    min_fixed0 = min(p.fixed for p in particles)
    dispersion, observed_rem = _unused_observations(
        step0, min_fixed0, exclude_indices={c_idx for c_idx, _ in picked},
    )
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
        plan = _next_stage_plan(
            config,
            next_step=next_step,
            fixed_len=min_fixed,
            retained_len=max_len,
            dispersion=dispersion,
            observed_rem=observed_rem,
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
        )

        all_after_terminal = is_final or all(
            not roll.token_ids for cand in joint_cands for roll in cand.rollouts
        )
        if is_final or all_after_terminal:
            win_c, win_r = select_metropolized(
                joint_cands, seeds, "conditional_is", next_step, group_blocks=True,
            )
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
        next_min_fixed = min(p.fixed for p in particles)
        dispersion, observed_rem = _unused_observations(
            stage_record, next_min_fixed, exclude_indices={c_idx for c_idx, _ in picked},
        )
        steps.append(stage_record.compact())

        if retain is not None:
            retain(adapter.prompt, particles[0].token_ids, *(p.token_ids for p in particles[1:]))
        if on_commit is not None:
            common = _common_prefix([p.token_ids[:p.fixed] for p in particles])
            if len(common) > published_len:
                published_len = len(common)
                on_commit(common)

    if len(particles) == 1:
        winner = particles[0]
    else:
        final_cands = weigh(
            [p.token_ids for p in particles],
            [[RolloutEvaluation((), p.reward, p.reward / config.reward_temperature)] for p in particles],
        )
        win_idx, _ = select_metropolized(
            final_cands, seeds, "conditional_is", len(steps), "particle_final", group_blocks=True,
        )
        winner = particles[win_idx]
        if retain is not None:
            retain(adapter.prompt, winner.token_ids)

    if on_commit is not None and len(winner.token_ids) > published_len:
        on_commit(winner.token_ids)
    total_slots = None if any(s.forward_token_slots is None for s in steps) else sum(
        s.forward_token_slots or 0 for s in steps
    )
    reason = (
        "deadline" if stopped_by_deadline
        else "eos" if winner.token_ids[-1] == adapter.sampling.eos_token_id
        else "length" if len(winner.token_ids) >= config.total_length else "stop"
    )
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
    """Run converged multi-stage Block-SMC Conditional IS and return the chosen sequence.

    Reports only irrevocable committed prefixes to ``on_commit``. Deadlines are
    checked strictly between stages, never mid-tree.
    """
    adapter = ConditionalISAdapter(
        backend=backend,
        prompt=prompt,
        config=config,
        sampling=sampling or SamplingConfig(),
        reward=reward,
        prefix_reward=prefix_reward,
    )
    if config.beam_particles > 1:
        return _run_particle_smc(adapter, config, seeds, deadline=deadline, on_commit=on_commit)

    state = adapter.initial_state
    steps: list[ConditionalISStep] = []
    stopped_by_deadline = False

    stage0_final = config.stages == 1 or config.stage0_block_size >= config.total_length
    record, state = adapter.step(
        state, 0, seeds,
        block_width=config.stage0_block_size,
        candidate_count=config.candidate_count,
        rollout_count=config.rollout_count,
        is_final=stage0_final,
    )
    if on_commit is not None:
        on_commit(state.token_ids[:state.fixed])
    dispersion, observed_rem = _unused_observations(record, state.fixed)
    steps.append(record.compact())
    del record

    while not adapter.is_terminal(state):
        if deadline is not None and time.monotonic() >= deadline:
            stopped_by_deadline = True
            break
        next_step = len(steps)
        plan = _next_stage_plan(
            config,
            next_step=next_step,
            fixed_len=state.fixed,
            retained_len=len(state.token_ids),
            dispersion=dispersion,
            observed_rem=observed_rem,
        )
        if plan is None:
            state = replace(state, fixed=len(state.token_ids))
            if on_commit is not None:
                on_commit(state.token_ids)
            break
        next_block, fresh, next_k, is_final = plan
        record, state = adapter.step(
            state,
            next_step,
            seeds,
            block_width=next_block,
            candidate_count=1 + fresh,
            rollout_count=next_k,
            is_final=is_final,
        )
        if on_commit is not None:
            on_commit(state.token_ids[:state.fixed])
        dispersion, observed_rem = _unused_observations(record, state.fixed)
        steps.append(record.compact())
        del record

    if on_commit is not None and state.fixed < len(state.token_ids):
        on_commit(state.token_ids)
    forward_slots = None if any(step.forward_token_slots is None for step in steps) else sum(
        step.forward_token_slots or 0 for step in steps
    )
    reason = (
        "deadline" if stopped_by_deadline or not adapter.is_terminal(state)
        else "eos" if state.token_ids[-1] == adapter.sampling.eos_token_id
        else "length" if len(state.token_ids) >= config.total_length else "stop"
    )
    return ConditionalISResult(adapter.prompt, state.token_ids, tuple(steps), forward_slots, reason)


__all__ = ["ConditionalISAdapter", "RetainedSequence", "run_conditional_is"]
