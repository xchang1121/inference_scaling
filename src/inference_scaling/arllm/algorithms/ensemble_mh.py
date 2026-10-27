"""Complete candidate pools with suffix MH and final importance resampling.

The pool target is product_i p(y_i) * sum_i exp(r(y_i)/tau) / (M Z).
Updating one slot uses the ratio of pool weight sums, together with the
reverse/forward cut probabilities. Selecting a slot by its reward weight
therefore has marginal p(y) exp(r(y)/tau) / Z at stationarity. Initializing
from p**M gives ordinary finite-pool SIR before any updates; finite rounds
remain approximate. The proposal and target use the same base policy.

Every round generates its M proposals together and commits them in slot
order. A round ends before the next starts, allowing statistics to be pruned
to the current pool. Only scalar step summaries survive completed rounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, log
from typing import Any

from inference_scaling.arllm.algorithms.config import MHConfig
from inference_scaling.arllm.algorithms.mh import suffix_length_probabilities
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, GenerationRequest
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import RolloutEvaluation
from inference_scaling.shared.sampling.importance import categorical_index_from_uniform, logmeanexp, normalize_log_weights
from inference_scaling.shared.sampling.mh import decide_metropolis_hastings
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
    temperature = config.reward_temperature

    def generate(prefixes: list[TokenSequence], stage: int) -> list[RolloutEvaluation]:
        samples = backend.sample_batch([
            GenerationRequest(prompt + prefix, config.total_length - len(prefix), sampling,
                seeds.derive('ensemble', 'generate', stage, slot) if stage
                else seeds.derive('conditional_is', 0, 'candidate', slot),
                f'ensemble:{stage}:{slot}')
            for slot, prefix in enumerate(prefixes)])
        if len(samples) != candidate_count:
            raise RuntimeError("ensemble generation returned an invalid number of samples")
        sequences = []
        for prefix, sample in zip(prefixes, samples, strict=True):
            if (sample.prefix != prompt + prefix or not sample.token_ids
                    or len(sample.token_ids) > config.total_length - len(prefix)):
                raise RuntimeError("ensemble generation returned an invalid complete suffix")
            sequences.append(prefix + sample.token_ids)
        values = reward(prompt, sequences)
        if len(values) != candidate_count or any(not isfinite(float(value) / temperature) for value in values):
            raise ValueError("ensemble MH requires one finite fixed reward per sequence")
        return [RolloutEvaluation(tokens, float(value), float(value) / temperature)
                for tokens, value in zip(sequences, values, strict=True)]

    def retain(pool: list[RolloutEvaluation]) -> None:
        keep = getattr(backend, 'retain_statistics', None)
        if keep is not None:
            keep(prompt, *(item.token_ids for item in pool))

    pool = generate([()] * candidate_count, 0)
    initial_unique = len({item.token_ids for item in pool})
    retain(pool)
    steps: list[dict[str, Any]] = []
    for stage in range(1, config.iterations + 1):
        probabilities = [suffix_length_probabilities(len(item.token_ids), config.suffix_schedule)[::-1]
                         for item in pool]
        cuts = [int(seeds.generator('ensemble', 'cut', stage, slot).choice(len(chances), p=chances))
                for slot, chances in enumerate(probabilities)]
        proposed = generate([item.token_ids[:cut] for item, cut in zip(pool, cuts, strict=True)], stage)
        logs = [item.log_weight for item in pool]
        for slot, (candidate, cut) in enumerate(zip(proposed, cuts, strict=True)):
            previous = pool[slot]
            forward = probabilities[slot][cut]
            reverse = suffix_length_probabilities(len(candidate.token_ids), config.suffix_schedule)[::-1][cut]
            current_sum = logmeanexp(logs)
            logs[slot] = candidate.log_weight
            decision = decide_metropolis_hastings(
                current_target_log_density=current_sum, proposed_target_log_density=logmeanexp(logs),
                forward_proposal_log_probability=log(forward), reverse_proposal_log_probability=log(reverse),
                uniform=float(seeds.generator('ensemble', 'accept', stage, slot).random()))
            if decision.accepted:
                pool[slot] = candidate
            else:
                logs[slot] = previous.log_weight
            steps.append({'round': stage, 'slot': slot, 'cut': cut,
                          'previous_length': len(previous.token_ids), 'proposed_length': len(candidate.token_ids),
                          'generated_tokens': len(candidate.token_ids) - cut,
                          'previous_reward': previous.reward, 'proposed_reward': candidate.reward,
                          'forward_cut_probability': forward, 'reverse_cut_probability': reverse,
                          'log_acceptance': decision.log_acceptance, 'accepted': decision.accepted})
        retain(pool)
    weights = normalize_log_weights([item.log_weight for item in pool])
    selected = categorical_index_from_uniform(weights, float(seeds.generator('ensemble', 'select').random()))
    result = pool[selected]
    retain([result])
    return EnsembleResult(result.token_ids, result.reward, {
        'method': 'ensemble_mh', 'ensemble_size': candidate_count, 'rounds': config.iterations,
        'updates': len(steps), 'accepted': sum(step['accepted'] for step in steps),
        'suffix_schedule': config.suffix_schedule, 'selected_index': selected,
        'initial_unique_sequences': initial_unique, 'final_unique_sequences': len({item.token_ids for item in pool}),
        'final_weight_ess': 1.0 / sum(weight * weight for weight in weights), 'steps': steps})


__all__ = ['EnsembleResult', 'run_ensemble_mh']
