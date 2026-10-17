"""Reproduce the CPU distribution and virtual-latency checks.

Run from the repository root with its Python environment. No model, GPU,
network, wall-clock sleep, or optional model libraries are needed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from collections import Counter
from math import exp, log
from pathlib import Path

import numpy as np

# The deterministic tabular backend is a test double, so it lives with the test suite; this script is
# the CPU harness that reproduces the scheduler checks against it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference_scaling.arllm.algorithms.conditional_is import ConditionalISAdapter, RetainedSequence, run_conditional_is
from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.backends.scheduling import SchedulerConfig
from tests.simulation import SimulatedBackend
from tests.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import ScoreRequest, SequenceSample
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.importance import normalize_log_weights
from inference_scaling.shared.types import pointwise


def service_time(request, index, sample):
    # Correlate service time with sampled output, deliberately making a
    # fastest-completion rule biased. The admission predictor cannot see this.
    return .001 * (1 + 5 * int(sample.token_ids[-1] == 1) + request.seed % 3)


def coupled_runs(trials):
    model = TabularAutoregressiveBackend({(): [.25, .25, .5], (0,): [.7, .25, .05]}, fallback=[.48, .48, .04])
    reward = pointwise(lambda prompt, tokens: .4 * sum(tokens) - .1 * len(tokens))
    sampling = SamplingConfig(eos_token_id=2)
    cases = []
    for block_first, capacity, order in itertools.product((False, True), (1, 2, 8), ("fifo", "fair_lpt")):
        config = ConditionalISConfig(4, 3, 2, 7, .7, block_first)
        baseline_counts, scheduled_counts = Counter(), Counter()
        mismatches, steps, maximum = 0, 0, 0
        for seed in range(trials):
            baseline = run_conditional_is(model, (), config, reward, SeedStream(seed), sampling=sampling)
            sim = SimulatedBackend(model, SchedulerConfig(max_in_flight=capacity, policy=order), token_seconds=service_time)
            scheduled = run_conditional_is(sim, (), config, reward, SeedStream(seed), sampling=sampling)
            mismatches += int(baseline != scheduled)
            steps += len(scheduled.steps)
            maximum = max(maximum, sim.queue.maximum_active)
            baseline_counts[baseline.token_ids] += 1
            scheduled_counts[scheduled.token_ids] += 1
        keys = baseline_counts.keys() | scheduled_counts.keys()
        tv = .5 * sum(abs(baseline_counts[key] - scheduled_counts[key]) for key in keys) / trials
        cases.append({"block_first": block_first, "capacity": capacity, "policy": order, "paired_runs": trials,
                      "steps_compared": steps, "full_result_mismatches": mismatches,
                      "histogram_total_variation": tv, "maximum_active": maximum})
        if mismatches or tv or maximum > capacity:
            raise AssertionError(cases[-1])
    return cases


def exact_transition_check():
    """Enumerate all auxiliary draws and integrate the actual implementation's weights.

    For length two, M=K=2 and a retained (a,b), the independent draws are:
    retained branch e, fresh candidate c and its own d, and fresh branch f.
    There are 16 combinations per retained state. Selection is integrated
    analytically from ConditionalISStep, rather than estimated with seed counts.
    """
    first = (.6, .4)
    second = ((.8, .2), (.3, .7))
    states = list(itertools.product((0, 1), repeat=2))
    reward = lambda tokens: 1.3 * sum(tokens)
    weights = [first[a] * second[a][b] * exp(reward((a, b))) for a, b in states]
    target = np.asarray(weights) / sum(weights)
    config = ConditionalISConfig(2, 2, 1, 2, 1.0, False)

    class Draws:
        model_id = "enumerated"

        def __init__(self, c, d, e, f):
            self.c, self.d, self.e, self.f = c, d, e, f

        def sample_batch(self, requests):
            samples = []
            for request in requests:
                if request.request_id.endswith("candidate:1"):
                    tokens = (self.c, self.d)
                    logs = (log(first[self.c]), log(second[self.c][self.d]))
                else:
                    token = self.e if ":candidate:0:" in request.request_id else self.f
                    tokens, logs = (token,), (log(second[request.prefix[-1]][token]),)
                samples.append(SequenceSample(request.prefix, tokens, logs, request.sampling.policy_id,
                                              self.model_id, request.request_id))
            return samples

    def transition(scheduler):
        matrix = np.zeros((4, 4))
        for row, (a, b) in enumerate(states):
            for c, d, e, f in itertools.product((0, 1), repeat=4):
                model = Draws(c, d, e, f)
                backend = model if scheduler is None else SimulatedBackend(model, scheduler, token_seconds=service_time)
                adapter = ConditionalISAdapter(backend=backend, prompt=(), config=config, sampling=SamplingConfig(),
                                                reward=pointwise(lambda prompt, tokens: reward(tokens)))
                record, _ = adapter.step(RetainedSequence((a, b), reward((a, b))), 0, SeedStream(100))
                proposal = second[a][e] * first[c] * second[c][d] * second[c][f]
                candidate_probabilities = normalize_log_weights([candidate.log_weight for candidate in record.candidates])
                for candidate, probability in zip(record.candidates, candidate_probabilities, strict=True):
                    leaf_probabilities = normalize_log_weights([leaf.log_weight for leaf in candidate.rollouts])
                    for leaf, conditional in zip(candidate.rollouts, leaf_probabilities, strict=True):
                        output = candidate.token_ids + leaf.token_ids
                        matrix[row, states.index(output)] += proposal * probability * conditional
        return matrix

    baseline = transition(None)
    comparisons = []
    for capacity, policy in itertools.product((1, 2, 4), ("fifo", "fair_lpt")):
        scheduled = transition(SchedulerConfig(max_in_flight=capacity, policy=policy))
        difference = float(np.max(np.abs(scheduled - baseline)))
        invariant_error = float(np.max(np.abs(target @ scheduled - target)))
        row_error = float(np.max(np.abs(scheduled.sum(axis=1) - 1)))
        comparisons.append({"capacity": capacity, "policy": policy, "maximum_matrix_difference": difference,
                            "target_invariance_max_error": invariant_error, "row_sum_max_error": row_error})
        if max(difference, invariant_error, row_error) > 1e-12:
            raise AssertionError(comparisons[-1])
    return {"states": states, "auxiliary_cases_per_state": 16, "target": target.tolist(),
            "transition_matrix": baseline.tolist(), "schedules": comparisons}


def stationary_runs(trials):
    model = TabularAutoregressiveBackend({(): [.6, .4], (0,): [.8, .2], (1,): [.3, .7]}, fallback=[.5, .5])
    states = list(itertools.product((0, 1), repeat=3))
    reward = lambda tokens: 2.0 * sum(tokens)
    logs = model.score_batch([ScoreRequest((), tuple(states), SamplingConfig())])
    masses = [exp(sum(values) + reward(tokens)) for tokens, values in zip(states, logs, strict=True)]
    target = np.asarray(masses) / sum(masses)
    starts = np.random.default_rng(20261016).choice(len(states), size=trials, p=target)
    counts, mismatches = Counter(), 0
    for trial, index in enumerate(starts):
        initial = RetainedSequence(states[index], reward(states[index]))
        results = []
        for streaming in (False, True):
            sim = SimulatedBackend(model, SchedulerConfig(max_in_flight=2), streaming=streaming, token_seconds=service_time)
            adapter = ConditionalISAdapter(backend=sim, prompt=(), config=ConditionalISConfig(2, 2, 1, 3, 1.0, False),
                                            sampling=SamplingConfig(), reward=pointwise(lambda prompt, tokens: reward(tokens)))
            state, records = initial, []
            while not adapter.is_terminal(state):
                record, state = adapter.step(state, len(records), SeedStream(100000 + trial))
                records.append(record)
            results.append((state, records))
        mismatches += int(results[0] != results[1])
        counts[results[1][0].token_ids] += 1
    empirical = np.asarray([counts[tokens] / trials for tokens in states])
    tv = float(.5 * np.abs(empirical - target).sum())
    if mismatches:
        raise AssertionError("stationary coupled runs differ")
    return {"paired_runs": trials, "full_result_mismatches": mismatches,
            "empirical_target_total_variation": tv,
            "states": [{"tokens": tokens, "target": float(probability), "count": counts[tokens]}
                       for tokens, probability in zip(states, target, strict=True)]}


class LongTailModel(TabularAutoregressiveBackend):
    def __init__(self):
        super().__init__({}, fallback=[.45, .45, .1])

    def _base_probabilities(self, prefix):
        if not prefix:
            return np.array([.15, .85, 0.0])
        eos = .008 if prefix[0] == 0 else .08
        return np.array([(1 - eos) / 2, (1 - eos) / 2, eos])


def latency_runs(trials):
    rows = []
    for scenario, capacity in itertools.product(("fixed_length", "long_tail"), (1, 2, 4, 8, 16)):
        times = {"barrier_fifo": [], "eager_fair_lpt": []}
        mismatches, maximum = 0, 0
        for trial in range(trials):
            results = []
            for label, streaming in (("barrier_fifo", False), ("eager_fair_lpt", True)):
                model = LongTailModel() if scenario == "long_tail" else TabularAutoregressiveBackend({}, fallback=[.5, .5])
                scheduler = SchedulerConfig(max_in_flight=capacity, policy="fair_lpt" if streaming else "fifo")
                sim = SimulatedBackend(model, scheduler, streaming=streaming, prefill_seconds=.00002)
                config = ConditionalISConfig(6, 4, 8, 192 if scenario == "long_tail" else 64, 1.0, False)
                adapter = ConditionalISAdapter(backend=sim, prompt=(), config=config,
                    sampling=SamplingConfig(eos_token_id=2 if scenario == "long_tail" else None),
                    reward=pointwise(lambda prompt, tokens: float(sum(tokens)) / len(tokens)))
                results.append(adapter.step(adapter.initial_state, 0, SeedStream(200000 + trial)))
                times[label].append(sim.now)
                maximum = max(maximum, sim.queue.maximum_active)
            mismatches += int(results[0] != results[1])
        before, after = np.asarray(times["barrier_fifo"]), np.asarray(times["eager_fair_lpt"])
        row = {"scenario": scenario, "capacity": capacity, "trees": trials,
               "baseline_mean_seconds": float(before.mean()), "scheduled_mean_seconds": float(after.mean()),
               "baseline_p95_seconds": float(np.quantile(before, .95)), "scheduled_p95_seconds": float(np.quantile(after, .95)),
               "mean_speedup": float(before.mean() / after.mean()),
               "slower_trees": int((after > before + 1e-10).sum()), "full_result_mismatches": mismatches,
               "maximum_active": maximum}
        rows.append(row)
        if mismatches or maximum > capacity:
            raise AssertionError(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--stationary-trials", type=int, default=6000)
    parser.add_argument("--latency-trials", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("docs/reports/ROLLOUT_SCHEDULING_CPU.json"))
    args = parser.parse_args()
    if min(args.trials, args.stationary_trials, args.latency_trials) <= 0:
        parser.error("trial counts must be positive")
    start = time.monotonic()
    report = {"python": platform.python_version(), "numpy": np.__version__,
              "description": "CPU exact finite backend with virtual-time latency simulation"}
    report["exact_transition"] = exact_transition_check()
    print("Exact transition matrices verified", flush=True)
    report["coupled_runs"] = coupled_runs(args.trials)
    print("Paired complete generation results verified", flush=True)
    report["stationarity"] = stationary_runs(args.stationary_trials)
    print("Stationary target samples checked", flush=True)
    report["latency"] = latency_runs(args.latency_trials)
    report["wall_seconds"] = round(time.monotonic() - start, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "wall_seconds": report["wall_seconds"],
                      "stationary_tv": report["stationarity"]["empirical_target_total_variation"]}, indent=2))


if __name__ == "__main__":
    main()
