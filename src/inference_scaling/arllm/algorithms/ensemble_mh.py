"""Complete candidate pools with suffix MH and final importance resampling.

The pool target is product_i p(y_i) * sum_i exp(r(y_i)/tau) / (M Z).
Updating one slot uses the ratio of pool weight sums, together with the
reverse/forward cut probabilities. Selecting a slot by its reward weight
therefore has marginal p(y) exp(r(y)/tau) / Z at stationarity. Initializing
from p**M gives ordinary finite-pool SIR before any updates; finite rounds
remain approximate. The proposal and target use the same base policy.

Optional persistent strata replace each slot's base law with one fixed
conditional base law whose average is p. Local cuts are restricted to prefixes
where that condition has resolved; whole-trajectory refreshes stay stratified.
The same pool-weight acceptance formula then preserves the output target.

Batch execution generates each round together, then prunes its statistics.
Pipeline execution overlaps iid base updates whose source states are known,
while preserving the same logical decisions. Only scalar step summaries survive
completed rounds; live requests retain their needed prefix statistics.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from math import isfinite, log
from typing import Any

import numpy as np

from inference_scaling.arllm.algorithms.config import MHConfig
from inference_scaling.arllm.algorithms.mh import suffix_length_probabilities
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, GenerationRequest
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import RolloutEvaluation
from inference_scaling.shared.sampling.importance import categorical_index_from_uniform, logmeanexp, normalize_log_weights
from inference_scaling.shared.sampling.mh import decide_metropolis_hastings
from inference_scaling.shared.sampling.nonself import nonself_cut_law, plan_nonself
from inference_scaling.shared.types import TokenBatchReward, TokenSequence


@dataclass(frozen=True, slots=True)
class EnsembleResult:
    token_ids: TokenSequence
    reward: float
    trace: dict[str, Any]


def run_ensemble_mh(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: MHConfig,
    seeds: SeedStream,
    *,
    candidate_count: int,
    sampling: SamplingConfig,
    reward: TokenBatchReward,
    initialization: str = "iid",
    stratified_updates: bool = False,
    nonself_updates: bool = False,
    execution: str = "batch",
) -> EnsembleResult:
    """M complete base trajectories followed by ``iterations`` full-pool rounds.

    Cut positions range over the actual current length, not the maximum
    length. A length change consequently requires a Hastings correction even
    for uniform cuts. Proposals may be generated in parallel because each
    slot's source sequence stays unchanged until that slot's ordered update.
    """
    if type(candidate_count) is not int or candidate_count < 1:
        raise ValueError("candidate_count must be a positive integer")
    if (config.alpha != 1 or config.reward_temperature is None or config.iterations is None
            or config.suffix_replay or config.early_rejection):
        raise ValueError("ensemble MH requires an explicit iteration count, alpha=1, a reward, and base proposals")
    if not sampling.full_support:
        raise ValueError("ensemble MH requires the full-support base policy")
    if initialization not in {"iid", "stratified"}:
        raise ValueError("ensemble initialization must be iid or stratified")
    if stratified_updates and (initialization != "stratified" or candidate_count <= 1):
        raise ValueError("stratified updates require stratified initialization and candidate_count > 1")
    if nonself_updates and stratified_updates:
        raise ValueError("nonself updates require the base suffix law, not persistent strata")
    if execution not in {"batch", "pipeline"}:
        raise ValueError("ensemble execution must be batch or pipeline")
    if execution == "pipeline":
        if initialization != "iid" or stratified_updates or nonself_updates:
            raise ValueError("pipeline execution requires iid initialization and ordinary base suffix proposals")
        if getattr(backend, 'supports_streaming_generation', False):
            from inference_scaling.arllm.algorithms.ensemble_schedule import run_pipeline

            return EnsembleResult(*run_pipeline(backend, prompt, config, seeds, candidate_count=candidate_count,
                                                sampling=sampling, reward=reward))
    temperature = config.reward_temperature
    context: Any = getattr(backend, "stratified_initialization", None)
    if initialization == "stratified" and context is None:
        raise ValueError("the backend does not support stratified initialization")
    exclude: Any = getattr(backend, "first_token_exclusion", None)
    if nonself_updates and exclude is None:
        raise ValueError("nonself updates require first-token exclusion with base probabilities")
    strata = seeds.generator("initial-pool", "permutation").permutation(candidate_count)

    def cut_probabilities(length: int, resolved_after: int | None):
        chances = np.array(suffix_length_probabilities(length, config.suffix_schedule)[::-1])
        if stratified_updates:
            # Before resolution the native suffix law need not equal the
            # stratum-conditional law. Refresh from the start or keep a resolved prefix.
            chances[1:length if resolved_after is None else resolved_after] = 0.
            chances /= chances.sum()
        return chances

    interval_statistics = []

    def generate(prefixes: list[TokenSequence], stage: int, positions: list[int | None],
                 active: list[int], exclusions: dict[int, int]):
        requests = [GenerationRequest(prompt + prefix, config.total_length - len(prefix), sampling,
                    seeds.derive('ensemble', 'generate', stage, slot) if stage
                    else seeds.derive('conditional_is', 0, 'candidate', slot), f'ensemble:{stage}:{slot}')
                    for slot in active for prefix in [prefixes[slot]]]
        by_slot = dict(zip(active, requests, strict=True))
        global_slots = [slot for slot in active if not prefixes[slot]
                        and (stratified_updates or stage == 0 and initialization == "stratified")]
        entries = [(by_slot[slot].seed, int(strata[slot]), candidate_count) for slot in global_slots]
        excluded = [(by_slot[slot].seed, token) for slot, token in exclusions.items()]
        constraints = context(entries) if entries else exclude(excluded) if excluded else nullcontext({})
        with constraints as statistics:
            samples = backend.sample_batch(requests)
            if len(samples) != len(active):
                raise RuntimeError("ensemble generation returned an invalid number of samples")
            sequences = []
            for slot, sample in zip(active, samples, strict=True):
                prefix = prefixes[slot]
                if (sample.prefix != prompt + prefix or not sample.token_ids
                        or len(sample.token_ids) > config.total_length - len(prefix)):
                    raise RuntimeError("ensemble generation returned an invalid complete suffix")
                if slot in exclusions and sample.token_ids[0] == exclusions[slot]:
                    raise RuntimeError("the backend did not exclude the old token at the first divergence")
                if nonself_updates and len(sample.token_logprobs) != len(sample.token_ids):
                    raise RuntimeError("nonself updates require one base log-probability per generated token")
                sequences.append(prefix + sample.token_ids)
            values = reward(prompt, sequences)
            if len(values) != len(active) or any(not isfinite(float(value) / temperature) for value in values):
                raise ValueError("ensemble MH requires one finite fixed reward per sequence")
        positions = positions.copy()
        statistics = dict(statistics)
        resolutions = statistics.pop("resolved_after", {})
        if stratified_updates:
            for slot in global_slots:
                if by_slot[slot].seed not in resolutions:
                    raise RuntimeError("stratified updates require a resolution status for every refreshed trajectory")
                position = resolutions[by_slot[slot].seed]
                # Generation can run past a subsequently trimmed thinking boundary.
                # Resolution after the visible sequence permits only a whole refresh.
                positions[slot] = position if position is not None and position <= len(sequences[slot]) else None
            interval_statistics.append({'round': stage, 'global_slots': global_slots,
                                        'resolved_at': positions.copy(), **statistics})
        evaluated = [RolloutEvaluation(tokens, float(value), float(value) / temperature)
                     for tokens, value in zip(sequences, values, strict=True)]
        return evaluated, positions, statistics, [sample.token_logprobs for sample in samples]

    def retain(pool: list[RolloutEvaluation]) -> None:
        keep = getattr(backend, 'retain_statistics', None)
        if keep is not None:
            keep(prompt, *(item.token_ids for item in pool))

    pool, resolved, initialization_statistics, pool_logprobs = generate(
        [()] * candidate_count, 0, [None] * candidate_count, list(range(candidate_count)), {})
    if not nonself_updates:
        pool_logprobs = []
    initial_unique = len({item.token_ids for item in pool})
    retain(pool)
    steps: list[dict[str, Any]] = []
    for stage in range(1, config.iterations + 1):
        probabilities = [cut_probabilities(len(item.token_ids), position)
                         for item, position in zip(pool, resolved, strict=True)]
        active = list(range(candidate_count))
        plans = []
        if nonself_updates:
            plans = [plan_nonself(values, law, seeds.generator('nonself', 'plan', stage, slot))
                     for slot, (values, law) in enumerate(zip(pool_logprobs, probabilities, strict=True))]
            active = [slot for slot, plan in enumerate(plans) if plan is not None]
            cuts = [plan.cut if plan is not None else 0 for plan in plans]
            starts = [plan.divergence if plan is not None else 0 for plan in plans]
            steps.extend({'round': stage, 'slot': slot, 'skipped': True, 'accepted': False}
                         for slot in range(candidate_count) if slot not in active)
            if not active:
                continue
        else:
            cuts = [int(seeds.generator('ensemble', 'cut', stage, slot).choice(len(chances), p=chances))
                    for slot, chances in enumerate(probabilities)]
            starts = cuts
        proposed, proposed_positions, _, new_logprobs = generate(
            [item.token_ids[:start] for item, start in zip(pool, starts, strict=True)], stage, resolved, active,
            {slot: pool[slot].token_ids[starts[slot]] for slot in active} if nonself_updates else {})
        candidates = dict(zip(active, proposed, strict=True))
        proposed_logprobs = {slot: pool_logprobs[slot][:starts[slot]] + values
                            for slot, values in zip(active, new_logprobs, strict=True)} if nonself_updates else {}
        logs = [item.log_weight for item in pool]
        order = seeds.generator('ensemble', 'order', stage).permutation(candidate_count) if stratified_updates else active
        for slot in order:
            slot = int(slot)
            candidate, cut = candidates[slot], cuts[slot]
            previous = pool[slot]
            previous_position = resolved[slot]
            forward = probabilities[slot][cut]
            reverse_probabilities = cut_probabilities(len(candidate.token_ids), proposed_positions[slot])
            reverse = reverse_probabilities[cut]
            forward_mass = reverse_mass = 1.
            if nonself_updates:
                plan = plans[slot]
                assert plan is not None
                forward_mass = plan.change_mass
                reverse_mass, _ = nonself_cut_law(proposed_logprobs[slot], reverse_probabilities)
                if reverse_mass <= 0:
                    raise ArithmeticError("a changed suffix must have a positive reverse change probability")
            current_sum = logmeanexp(logs)
            logs[slot] = candidate.log_weight
            decision = decide_metropolis_hastings(
                current_target_log_density=current_sum, proposed_target_log_density=logmeanexp(logs),
                forward_proposal_log_probability=log(forward) - log(forward_mass),
                reverse_proposal_log_probability=log(reverse) - log(reverse_mass),
                uniform=float(seeds.generator('ensemble', 'accept', stage, slot).random()))
            if decision.accepted:
                pool[slot] = candidate
                resolved[slot] = proposed_positions[slot]
                if nonself_updates:
                    pool_logprobs[slot] = proposed_logprobs[slot]
            else:
                logs[slot] = previous.log_weight
            steps.append({'round': stage, 'slot': slot, 'cut': cut,
                          'previous_length': len(previous.token_ids), 'proposed_length': len(candidate.token_ids),
                          'generated_tokens': len(candidate.token_ids) - starts[slot],
                          'previous_reward': previous.reward, 'proposed_reward': candidate.reward,
                          'forward_cut_probability': forward, 'reverse_cut_probability': reverse,
                          'log_acceptance': decision.log_acceptance, 'accepted': decision.accepted})
            if stratified_updates:
                steps[-1].update(stratum=int(strata[slot]), previous_resolved_at=previous_position,
                                 proposed_resolved_at=proposed_positions[slot])
            if nonself_updates:
                steps[-1].update(divergence=starts[slot], reused_prefix_tokens=starts[slot] - cut,
                                 forward_change_mass=forward_mass, reverse_change_mass=reverse_mass)
        retain(pool)
    weights = normalize_log_weights([item.log_weight for item in pool])
    selected = categorical_index_from_uniform(weights, float(seeds.generator('ensemble', 'select').random()))
    result = pool[selected]
    retain([result])
    return EnsembleResult(result.token_ids, result.reward, {
        'method': 'ensemble_mh', 'ensemble_size': candidate_count, 'rounds': config.iterations, 'execution': 'batch',
        'initialization': initialization, 'initialization_statistics': initialization_statistics,
        'stratified_updates': stratified_updates, 'interval_statistics': interval_statistics,
        'nonself_updates': nonself_updates,
        'updates': sum(not step.get('skipped', False) for step in steps),
        'skipped_updates': sum(step.get('skipped', False) for step in steps),
        'accepted': sum(step['accepted'] for step in steps),
        'suffix_schedule': config.suffix_schedule, 'selected_index': selected,
        'initial_unique_sequences': initial_unique, 'final_unique_sequences': len({item.token_ids for item in pool}),
        'final_weight_ess': 1.0 / sum(weight * weight for weight in weights), 'steps': steps})


__all__ = ['EnsembleResult', 'run_ensemble_mh']
