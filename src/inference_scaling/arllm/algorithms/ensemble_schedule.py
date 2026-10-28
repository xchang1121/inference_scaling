"""Execute the existing ordered pool kernel across overlapping generation rounds.

An unfinished slot contributes a nonnegative weight interval. An acceptance draw
can resolve early only when every value in that interval gives the same result.
The next suffix then uses a known source, without extra or speculative proposals.
This changes execution dependencies, not the prescribed logical update order.
"""
from __future__ import annotations

from math import exp, fsum, inf, isfinite, log
from sys import float_info

import numpy as np

from inference_scaling.arllm.algorithms.mh import suffix_length_probabilities
from inference_scaling.arllm.backends.scheduling import SampleBuffer
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.shared.sampling.conditional_is import RolloutEvaluation
from inference_scaling.shared.sampling.importance import categorical_index_from_uniform, logmeanexp, normalize_log_weights
from inference_scaling.shared.sampling.mh import decide_metropolis_hastings


def _ratio_bounds(old, new, correction, intervals):
    """Bounds for log[h (A + w_new)/(A + w_old)], where A sums other weights."""
    def total(values):
        if not values:
            return -inf
        peak = max(values)
        return peak if not isfinite(peak) else peak + log(fsum(exp(value - peak) for value in values))

    lower, upper = total([lo for lo, _ in intervals]), total([hi for _, hi in intervals])
    def ratio(other):
        return correction if other == inf else correction + float(np.logaddexp(other, new) - np.logaddexp(other, old))

    endpoints = ratio(lower), ratio(upper)
    # Widen the comparison for rounding; near-boundary draws wait for exact inputs.
    scale = 1 + abs(old) + abs(new) + abs(correction)
    scale += sum(abs(value) for value in (lower, upper) if isfinite(value))
    margin = 64 * np.finfo(float).eps * scale
    return min(endpoints) - margin, max(endpoints) + margin


def run_pipeline(backend, prompt, config, seeds, *, candidate_count, sampling, reward):
    """Return tokens, reward and the production trace for iid base proposals."""
    m, rounds, temperature = candidate_count, config.iterations, config.reward_temperature
    current: dict[int, RolloutEvaluation] = {}
    known: dict[tuple[int, int], float] = {}
    prepared: dict[tuple[int, int], tuple[RolloutEvaluation, RolloutEvaluation, int, float, float]] = {}
    decisions = {}
    requests, buffers = {}, {}
    initial_sequences = set()
    initial_unique = submitted = finished = 0
    session = backend.open_generation_session()

    def probabilities(length):
        return suffix_length_probabilities(length, config.suffix_schedule)[::-1]

    def submit(stage, slot):
        nonlocal submitted
        old = current[slot] if stage else None
        if stage:
            assert old is not None
            chances = probabilities(len(old.token_ids))
            cut = int(seeds.generator('ensemble', 'cut', stage, slot).choice(len(chances), p=chances))
            prefix, forward = old.token_ids[:cut], float(chances[cut])
        else:
            prefix, cut, forward = (), 0, 1.
        key = f'ensemble:{stage}:{slot}'
        seed = (seeds.derive('ensemble', 'generate', stage, slot) if stage
                else seeds.derive('conditional_is', 0, 'candidate', slot))
        request = GenerationRequest(prompt + prefix, config.total_length - cut, sampling, seed, key,
                                    intermediate_events=False)
        requests[key], buffers[key] = (stage, slot, old, cut, forward, request), SampleBuffer()
        session.submit([request])
        submitted += 1
        if submitted == m * (rounds + 1):
            session.seal()

    def required(stage, slot):
        return [(stage if other < slot else stage - 1, other) for other in range(m)]

    def interval(key):
        if key in known:
            return known[key], known[key]
        if key in prepared:
            new, old, *_ = prepared[key]
            return min(new.log_weight, old.log_weight), max(new.log_weight, old.log_weight)
        return -inf, inf

    def exact(stage, slot, new_weight, forward, reverse):
        logs = [known[key] for key in required(stage, slot)]
        before = logmeanexp(logs)
        logs[slot] = new_weight
        return decide_metropolis_hastings(current_target_log_density=before,
            proposed_target_log_density=logmeanexp(logs), forward_proposal_log_probability=log(forward),
            reverse_proposal_log_probability=log(reverse),
            uniform=float(seeds.generator('ensemble', 'accept', stage, slot).random()))

    def resolve_ready():
        changed = True
        while changed:
            changed = False
            for key in sorted(prepared):
                stage, slot = key
                new, old, cut, forward, reverse = prepared[key]
                keys = required(stage, slot)
                early = not all(item in known for item in keys)
                if not early:
                    accepted = exact(stage, slot, new.log_weight, forward, reverse).accepted
                else:
                    lo, hi = _ratio_bounds(old.log_weight, new.log_weight, log(reverse) - log(forward),
                                           [interval(item) for item in keys if item[1] != slot])
                    uniform = float(seeds.generator('ensemble', 'accept', stage, slot).random())
                    draw = log(max(uniform, float_info.min))
                    if draw <= min(0., lo):
                        accepted = True
                    elif draw > min(0., hi):
                        accepted = False
                    else:
                        continue
                current[slot] = new if accepted else old
                known[key] = current[slot].log_weight
                decisions[key] = {'round': stage, 'slot': slot, 'cut': cut,
                    'previous_length': len(old.token_ids), 'proposed_length': len(new.token_ids),
                    'generated_tokens': len(new.token_ids) - cut,
                    'previous_reward': old.reward, 'proposed_reward': new.reward,
                    'forward_cut_probability': forward, 'reverse_cut_probability': reverse,
                    'accepted': accepted, 'early_resolved': early}
                del prepared[key]
                changed = True
                if stage < rounds:
                    submit(stage + 1, slot)

    try:
        for slot in range(m):
            submit(0, slot)
        while finished < m * (rounds + 1):
            event = session.next_event()
            key = event.delta.request_id
            if key not in buffers:
                raise RuntimeError('unexpected pool generation event')
            buffers[key].append(event.delta)
            if not event.finished:
                continue
            sample = buffers.pop(key).sample()
            stage, slot, old, cut, forward, request = requests.pop(key)
            if (sample.prefix != request.prefix or not sample.token_ids
                    or len(sample.token_ids) > request.max_new_tokens):
                raise RuntimeError('ensemble generation returned an invalid complete suffix')
            tokens = request.prefix[len(prompt):] + sample.token_ids
            values = reward(prompt, [tokens])
            if len(values) != 1 or not isfinite(float(values[0]) / temperature):
                raise ValueError('ensemble MH requires one finite fixed reward per sequence')
            new = RolloutEvaluation(tokens, float(values[0]), float(values[0]) / temperature)
            finished += 1
            if stage == 0:
                initial_sequences.add(tokens)
                current[slot], known[0, slot] = new, new.log_weight
                if all((0, other) in known for other in range(m)):
                    initial_unique = len(initial_sequences)
                    initial_sequences.clear()
                submit(1, slot)
            else:
                reverse = float(probabilities(len(tokens))[cut])
                prepared[stage, slot] = new, old, cut, forward, reverse
            resolve_ready()
            compact = getattr(backend, 'compact_generation_statistics', None)
            if compact is not None:
                compact(prompt, *(item.token_ids for item in current.values()),
                        *(value[0].token_ids for value in prepared.values()))
        if prepared or requests or buffers or len(known) != m * (rounds + 1) or submitted != finished:
            raise RuntimeError('incomplete pool updates after generation')
        steps = []
        for (stage, slot), step in sorted(decisions.items()):
            decision = exact(stage, slot, step['proposed_reward'] / temperature,
                             step['forward_cut_probability'], step['reverse_cut_probability'])
            if decision.accepted != step['accepted']:
                raise ArithmeticError('an early pool decision differs from its complete logical context')
            step['log_acceptance'] = decision.log_acceptance
            steps.append(step)
    finally:
        session.close()
    weights = normalize_log_weights([current[slot].log_weight for slot in range(m)])
    selected = categorical_index_from_uniform(weights, float(seeds.generator('ensemble', 'select').random()))
    result = current[selected]
    retain = getattr(backend, 'retain_statistics', None)
    if retain is not None:
        retain(prompt, result.token_ids)
    return result.token_ids, result.reward, {
        'method': 'ensemble_mh', 'ensemble_size': m, 'rounds': rounds, 'execution': 'pipeline',
        'initialization': 'iid', 'initialization_statistics': {}, 'stratified_updates': False,
        'interval_statistics': [], 'nonself_updates': False, 'updates': len(steps), 'skipped_updates': 0,
        'accepted': sum(step['accepted'] for step in steps),
        'early_resolved': sum(step['early_resolved'] for step in steps),
        'suffix_schedule': config.suffix_schedule, 'selected_index': selected,
        'initial_unique_sequences': initial_unique, 'final_unique_sequences': len({item.token_ids for item in current.values()}),
        'final_weight_ess': 1.0 / sum(weight * weight for weight in weights), 'steps': steps}
