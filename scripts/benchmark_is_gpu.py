"""Paired Base / IS quality and latency on one CUDA or Ascend engine.

Both arms share model, prompt, sampling and sequence budgets. Timing excludes
loading, warm-up, cache reset, grading and file writes. Incomplete pairs do not
produce a latency ratio. Output is append-only JSONL.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from inference_scaling.app import ar as ar_app
from inference_scaling.app.ar import ARFamily
from inference_scaling.app.records import environment, git_state, json_sha256, source_sha256
from inference_scaling.app.settings import load_settings, validate_settings
from inference_scaling.datasets import load_dataset
from inference_scaling.shared.model.thinking import resolve_thinking
from inference_scaling.shared.rng import SeedStream


POOL_MODES = ("sir-k1", "met-k1", "sir-k2", "met-k2", "block-met-k2")
SC_SUFFIX = "-self-certainty"
SELECTION_PAIRS = (("sir-k1", "met-k1"), ("sir-k2", "met-k2"),
                   ("sir-k2", "block-met-k2"), ("met-k2", "block-met-k2"))

STUDIES = {
    "latency": ("base", "is"),
    # K=1 with distinct blocks has identical leaf/block Metropolized kernels.
    "selection": ("base", "is", *POOL_MODES),
    "rewards": ("base", "is", "bon-consilience", "is-self-certainty"),
    "selection-self-certainty": ("base", "is-self-certainty", *(mode + SC_SUFFIX for mode in POOL_MODES)),
}


def single_request_seconds(requests):
    """Time with exactly one unfinished request in this tree, including one-job rounds.

    Timestamps are application submission/delivery times. This is not a GPU
    utilization measure; batch delivery cannot provide this timeline.
    """
    changes = Counter()
    for request in requests.values():
        start, end = request["submitted"], request["finished"]
        if end < start:
            raise ValueError("request completion precedes submission")
        changes[start] += 1
        changes[end] -= 1
    active, previous, seconds = 0, 0.0, 0.0
    for timestamp, change in sorted(changes.items()):
        if active == 1:
            seconds += timestamp - previous
        active += change
        previous = timestamp
    return seconds


def reset_cache(raw):
    async def reset():
        return await raw._engine.reset_prefix_cache()
    result = raw._runner.run(reset())
    if result is False:
        raise RuntimeError("engine could not reset its prefix cache between questions")


def engine_settings(raw):
    cfg = raw._engine.vllm_config
    return {
        "flat_logprobs": getattr(raw, "_flat_logprobs", False),
        "scalar_self_certainty": getattr(raw, "_scalar_self_certainty", False),
        "stream_interval": cfg.scheduler_config.stream_interval,
        "async_scheduling": cfg.scheduler_config.async_scheduling,
        "enforce_eager": cfg.model_config.enforce_eager,
        "quantization": cfg.model_config.quantization,
        "cudagraph_mode": str(cfg.compilation_config.cudagraph_mode),
        "cudagraph_capture_sizes": cfg.compilation_config.cudagraph_capture_sizes,
        "compile_sizes": cfg.compilation_config.compile_sizes,
        "compile_ranges_endpoints": cfg.compilation_config.compile_ranges_endpoints,
        "model_runner_v2": getattr(cfg, "use_v2_model_runner", None),
        "attention_backend": str(cfg.attention_config.backend),
        "kv_block_size": cfg.cache_config.block_size,
        "qk_norm_rope_fusion": cfg.compilation_config.pass_config.enable_qk_norm_rope_fusion,
        "cudagraph_metrics": cfg.observability_config.cudagraph_metrics,
    }


class GraphMeasurements:
    """Read native graph statistics only when explicitly enabled for a benchmark."""

    def __init__(self, raw):
        self.raw, self.counts = raw, Counter()
        self.manager = raw._engine.logger_manager
        self.original = None
        if raw._engine.vllm_config.observability_config.cudagraph_metrics:
            if self.manager is None:
                raise RuntimeError("graph measurements require engine statistics")
            self.original = self.manager.record
            self.manager.record = self.record

    def record(self, scheduler_stats, *args, **kwargs):
        if scheduler_stats is not None and scheduler_stats.cudagraph_stats is not None:
            stat = scheduler_stats.cudagraph_stats
            self.counts[(stat.runtime_mode, stat.num_unpadded_tokens, stat.num_padded_tokens)] += 1
        return self.original(scheduler_stats, *args, **kwargs)

    def snapshot(self):
        async def snapshot():
            return self.counts.copy()
        return self.raw._runner.run(snapshot()) if self.original is not None else Counter()

    def difference(self, before):
        if self.original is None:
            return {}
        counts = self.snapshot() - before
        forward = sum(unpadded * count for (_, unpadded, _), count in counts.items())
        padded = sum(padded * count for (_, _, padded), count in counts.items())
        return {"graph_execution": {
            "forward_token_slots": forward,
            "estimated_dense_forward_flops": 2 * self.raw.parameter_count * forward,
            "estimated_dense_padded_flops": 2 * self.raw.parameter_count * padded,
            "padded_token_slots": padded,
            "padding_token_slots": sum((padded - unpadded) * count
                                      for (_, unpadded, padded), count in counts.items()),
            "shapes": [{"mode": mode, "tokens": unpadded, "padded": padded, "steps": count}
                       for (mode, unpadded, padded), count in sorted(counts.items())],
        }}

    def close(self):
        if self.original is not None:
            self.manager.record = self.original


class AdmissionMeasurements:
    """Actual admitted intervals, including requests canceled after selection."""

    def __init__(self, raw):
        self.raw, self.original = raw, raw.configure_scheduler
        raw.configure_scheduler = self.configure
        self.configure(raw.scheduler_config)

    def configure(self, config):
        self.original(config)
        self.intervals, self.search, self.commits = {}, None, []
        pool = self.raw._admission
        acquire, finish = pool.acquire, pool.finish
        self.pool = pool, acquire, finish

        async def measured_acquire(item):
            await acquire(item)
            self.intervals[item.key] = {"submitted": time.perf_counter()}

        def measured_finish(key, generated=None):
            if key in self.intervals and "finished" not in self.intervals[key]:
                self.intervals[key]["finished"] = time.perf_counter()
            return finish(key, generated)

        pool.acquire, pool.finish = measured_acquire, measured_finish

    def measure_search(self, run, *args, **kwargs):
        started = time.perf_counter()
        observer = kwargs.pop("on_commit", None)

        def committed(tokens):
            if tokens:
                self.commits.append({"seconds": time.perf_counter() - started, "tokens": len(tokens)})
            if observer is not None:
                observer(tokens)

        try:
            return run(*args, **kwargs, on_commit=committed)
        finally:
            self.search = started, time.perf_counter()

    def summary(self):
        timing = {"is_search_seconds": self.search[1] - self.search[0] if self.search is not None else None,
                  "is_first_commit_seconds": self.commits[0]["seconds"] if self.commits else None,
                  "is_commits": self.commits.copy()}
        if not self.raw.scheduler_config.enabled:
            return {"admitted_requests": None, "single_admitted_seconds": None,
                    "is_single_admitted_seconds": None, **timing}
        async def collect():
            if any("finished" not in interval for interval in self.intervals.values()):
                raise RuntimeError("admitted work outlived a measured solve")
            scoped = {}
            if self.search is not None:
                start, end = self.search
                scoped = {key: {"submitted": max(start, interval["submitted"]),
                                "finished": min(end, interval["finished"])}
                          for key, interval in self.intervals.items()
                          if interval["submitted"] < end and interval["finished"] > start}
            return {"admitted_requests": len(self.intervals),
                    "single_admitted_seconds": single_request_seconds(self.intervals),
                    "is_single_admitted_seconds": single_request_seconds(scoped), **timing}
        return self.raw._runner.run(collect())

    def close(self):
        self.raw.configure_scheduler = self.original
        pool, acquire, finish = self.pool
        pool.acquire, pool.finish = acquire, finish


def measured_solve(family, problem, seed, graph_measurements, admission_measurements=None):
    """One cold-prefix question with identical timing/accounting for Base and IS."""
    reset_cache(family.raw)
    family.raw.configure_scheduler(family.raw.scheduler_config)
    family.synchronize()
    before = asdict(family.raw.snapshot())
    graph_before = graph_measurements.snapshot()
    started = time.perf_counter()
    result = family.solve(problem, SeedStream(seed))
    family.synchronize()
    elapsed = time.perf_counter() - started
    # A core RPC drains preceding abort/work messages before reading native metrics.
    # This cache reset is outside latency timing, like the reset before the question.
    if graph_measurements.original is not None:
        reset_cache(family.raw)
    after = asdict(family.raw.snapshot())
    counters = {key: value - before[key] for key, value in after.items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                and key != "maximum_in_flight_requests"}
    execution = graph_measurements.difference(graph_before)
    if admission_measurements is not None:
        execution.update(admission_measurements.summary())
    if ("graph_execution" in execution
            and execution["graph_execution"]["forward_token_slots"] < counters["generation_forward_token_slots"]):
        raise RuntimeError("native execution statistics omitted observed work")
    return result, elapsed, counters, execution


def pair_summary(rows, expected_pairs):
    """Complete paired latency, with failures and missing work kept explicit."""
    indexed, pairs = {}, {}
    for row in rows:
        key = row["problem_id"], row["repeat"], row["mode"]
        if key in indexed or row["mode"] not in {"base", "is"}:
            raise ValueError(f"duplicate or unknown measurement: {key}")
        if row["status"] == "complete" and (not np.isfinite(row["seconds"]) or row["seconds"] <= 0):
            raise ValueError("completed measurements need positive finite latency")
        indexed[key] = row
        pairs.setdefault(key[:2], {})[row["mode"]] = row
    if len(pairs) > expected_pairs:
        raise ValueError("more measured pairs than requested")
    complete = [pair for pair in pairs.values()
                if set(pair) == {"base", "is"} and all(row["status"] == "complete" for row in pair.values())]
    all_complete = bool(expected_pairs) and len(complete) == expected_pairs
    result = {"kind": "summary", "expected_pairs": expected_pairs, "complete_pairs": len(complete),
              "independent_questions": len({problem for problem, _ in pairs}),
              "missing_measurements": 2 * expected_pairs - len(indexed),
              "latency_ratio_is_over_base": None, "paired_ratio_p50": None, "paired_ratio_p95": None}
    for mode in ("base", "is"):
        arm = [row for row in rows if row["mode"] == mode]
        valid = [row for row in arm if row["status"] == "complete"]
        ungraded = sum("correct" not in row for row in valid)
        generated = [row["counters"].get("generated_tokens") for row in valid]
        measured = [row.get("graph_execution", {}).get("estimated_dense_padded_flops") for row in valid]
        unknown_work = any(row["counters"].get("cancelled_requests", 0)
                           or row["counters"].get("preemptions", 0) for row in valid)
        result[mode] = {
            "completed": len(valid), "failed": sum(row["status"] != "complete" for row in arm),
            "ungraded": ungraded, "correct": sum(row.get("correct", False) for row in valid),
            "accuracy": (sum(row.get("correct", False) for row in valid) / expected_pairs
                         if expected_pairs and len(arm) == expected_pairs and not ungraded else None),
            "wall_seconds": sum(row["seconds"] for row in valid),
            "latency_p50_seconds": statistics.median(row["seconds"] for row in valid) if valid else None,
            "latency_p95_seconds": float(np.percentile([row["seconds"] for row in valid], 95)) if valid else None,
            "output_tokens": sum(row["output_tokens"] for row in valid),
            "generated_tokens": (sum(generated) if all_complete and not unknown_work
                                 and all(value is not None for value in generated) else None),
            "observed_generated_tokens": sum(value for value in generated if value is not None),
            "estimated_dense_padded_flops": (sum(measured) if all_complete
                                            and all(value is not None for value in measured) else None),
            "length_exhausted": sum(row["length_exhausted"] for row in valid),
            "incomplete_thinking": sum(row["thinking_status"] not in {"complete", "disabled"} for row in valid),
            "reward_fallbacks": sum(bool(row["fallbacks"]) for row in valid),
        }
    if all_complete:
        ratios = [pair["is"]["seconds"] / pair["base"]["seconds"] for pair in complete]
        result.update(latency_ratio_is_over_base=result["is"]["wall_seconds"] / result["base"]["wall_seconds"],
                      paired_ratio_p50=statistics.median(ratios), paired_ratio_p95=float(np.percentile(ratios, 95)))
    quality_known = all_complete and all(result[mode]["accuracy"] is not None for mode in ("base", "is"))
    result["sample_quality_gate"] = (result["is"]["correct"] >= result["base"]["correct"]
                                     and result["is"]["correct"] > 0 if quality_known else None)
    return result


def comparison_summary(rows, expected_pairs, modes):
    """Reuse paired accounting; each alternative is compared on identical questions."""
    if tuple(modes) == STUDIES["latency"]:
        return pair_summary(rows, expected_pairs)
    unknown = {row["mode"] for row in rows} - set(modes)
    if unknown:
        raise ValueError(f"unknown comparison modes: {unknown}")

    def compare(reference, alternative):
        paired = [{**row, "mode": "base" if row["mode"] == reference else "is"}
                  for row in rows if row["mode"] in {reference, alternative}]
        summary = pair_summary(paired, expected_pairs)
        indexed = {(row["problem_id"], row["repeat"], row["mode"]): row for row in paired}
        wins = losses = ties = 0
        for (problem, repeat, mode), row in indexed.items():
            other = indexed.get((problem, repeat, "is"))
            if (mode != "base" or other is None or row["status"] != "complete" or other["status"] != "complete"
                    or "correct" not in row or "correct" not in other):
                continue
            wins += bool(other["correct"] and not row["correct"])
            losses += bool(row["correct"] and not other["correct"])
            ties += row["correct"] == other["correct"]
        return {"reference": reference, "alternative": alternative, "wins": wins, "losses": losses,
                "ties": ties, **summary}

    return {"kind": "summary", "expected_pairs": expected_pairs, "modes": list(modes),
            "versus_base": {mode: compare("base", mode) for mode in modes if mode != "base" and "base" in modes},
            "reward_pairs": [compare(reference, alternative) for reference, alternative in
                             (("is", "bon-consilience"), ("is", "is-self-certainty"),
                              ("bon-consilience", "is-self-certainty"),
                              *((mode, mode + SC_SUFFIX) for mode in POOL_MODES))
                             if reference in modes and alternative in modes],
            "selection_pairs": [compare(reference + suffix, alternative + suffix)
                                for suffix in ("", SC_SUFFIX) for reference, alternative in SELECTION_PAIRS
                                if reference + suffix in modes and alternative + suffix in modes]}


def select_mode(family, mode, settings):
    if mode not in {mode for modes in STUDIES.values() for mode in modes}:
        raise ValueError(f"unknown benchmark mode: {mode}")
    algorithm = "sample" if mode == "base" else "best_of_n" if mode == "bon-consilience" else "is"
    reward = None if mode == "base" else "self_certainty" if mode.endswith(SC_SUFFIX) else "consilience"
    family.choices = SimpleNamespace(algorithm=algorithm, reward=reward)
    family.config = dict(family.ar["algorithms"][algorithm])
    pool_mode = mode.removesuffix(SC_SUFFIX)
    if pool_mode in POOL_MODES:
        family.config.update(rollout_count=int(pool_mode[-1]), selection_ready=False,
                             selection=("sir" if pool_mode.startswith("sir-") else
                                        "block_metropolized" if pool_mode.startswith("block-") else "metropolized"))
    if mode == "bon-consilience":
        family.config["samples"] = 4
    family.reward_settings = None if reward is None else dict(settings["rewards"][reward])
    if reward == "self_certainty":
        family.reward_settings["scope"] = "thinking"


def run_pairs(family, dataset, settings, repeats, graph, admission, write, modes=STUDIES["latency"]):
    """Pair questions and seeds, rotate method order, stop after an engine failure."""
    rows = []
    expected = len(dataset.problems) * repeats
    try:
        for repeat in range(repeats):
            for index, problem in enumerate(dataset.problems):
                seed = SeedStream(settings["run"]["seed"]).derive("paired-benchmark", problem.id, repeat)
                offset = (index + repeat) % len(modes)
                order = modes[offset:] + modes[:offset]
                for mode in order:
                    select_mode(family, mode, settings)
                    row = {"kind": "measurement", "problem_id": problem.id, "repeat": repeat,
                           "mode": mode, "seed": seed, "algorithm_config": family.config,
                           "reward_config": family.reward_settings}
                    print(json.dumps({"kind": "start", **{key: row[key] for key in
                                                         ("problem_id", "repeat", "mode", "seed")}}), flush=True)
                    try:
                        solved, seconds, counters, execution = measured_solve(family, problem, seed, graph, admission)
                    except BaseException as error:
                        row.update(status="failed", error=f"{type(error).__name__}: {error}")
                        rows.append(row)
                        write(row)
                        raise
                    row.update(status="complete", seconds=seconds, counters=counters, **execution,
                               output_tokens=solved["output"]["tokens"], answer_text=solved["answer_text"],
                               output_text=solved["output"]["text"], trace=solved["trace"], cost=solved["cost"],
                               thinking_status=solved["output"]["thinking_status"],
                               length_exhausted=solved["output"]["length_exhausted"], fallbacks=solved["fallbacks"])
                    try:
                        row.update(asdict(dataset.grade(solved["answer_text"], problem)))
                    except Exception as error:
                        row["grade_error"] = f"{type(error).__name__}: {error}"
                    rows.append(row)
                    write(row)
                    print(json.dumps({key: row.get(key) for key in
                                      ("problem_id", "repeat", "mode", "seconds", "correct", "output_tokens")}), flush=True)
    finally:
        summary = comparison_summary(rows, expected, modes)
        write(summary)
        print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=Path("settings/inference.json"))
    parser.add_argument("--dataset", choices=("gsm8k", "math500"), default="math500")
    parser.add_argument("--questions", type=int)
    parser.add_argument("--selection-skip", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--max-tokens", type=int, help="explicit common output cap; otherwise use settings")
    effort = parser.add_mutually_exclusive_group()
    effort.add_argument("--effort", help="explicit common thinking effort; otherwise use settings")
    effort.add_argument("--thinking-budget", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--study", choices=tuple(STUDIES), default="latency",
                        help="selection compares Consilience pools; selection-self-certainty uses full-vocabulary KL; "
                             "rewards compares Best-of-4 and fast IS rewards")
    parser.add_argument("--modes", nargs="+", help="run only these arms from the selected study (independent supplement)")
    parser.add_argument("--summary", type=Path, help="summarize paired JSONL on CPU without loading a model")
    args = parser.parse_args()
    if args.summary:
        rows = [json.loads(line) for line in args.summary.read_text(encoding="utf-8").splitlines()]
        meta = next(row for row in rows if row["kind"] == "metadata")
        print(json.dumps(comparison_summary([row for row in rows if row["kind"] == "measurement"], meta["expected_pairs"],
                                           tuple(meta.get("modes", STUDIES["latency"]))),
                         ensure_ascii=False, indent=2))
        return
    if not args.output or not args.label:
        parser.error("--output and --label are required")
    modes = tuple(args.modes) if args.modes else STUDIES[args.study]
    if len(set(modes)) != len(modes) or not set(modes) <= set(STUDIES[args.study]):
        parser.error("--modes must be distinct arms from the selected --study")
    if args.output.exists():
        parser.error("output exists; use --summary to read it or choose a new output")
    settings = load_settings(args.settings)
    ar, selection = settings["ar"], settings["datasets"][args.dataset]["selection"]
    if args.questions is not None:
        if args.questions < 1:
            parser.error("--questions must be positive")
        selection["count"] = args.questions
    if args.selection_skip is not None:
        if args.dataset != "math500" or args.selection_skip < 0:
            parser.error("--selection-skip requires MATH-500 and a nonnegative offset")
        selection["skip"] = args.selection_skip
    if args.max_tokens is not None:
        ar["max_new_tokens"] = args.max_tokens
    if args.effort is not None or args.thinking_budget is not None:
        if ar["thinking"] is None:
            parser.error("explicit effort/budget requires an ar.thinking profile")
        ar["thinking"].update(reasoning_effort=args.effort, budget_tokens=args.thinking_budget)
    repeats = settings["run"]["draws"] if args.repeats is None else args.repeats
    if repeats < 1:
        parser.error("--repeats must be positive")
    engine = ar["engine"]
    if engine["backend"] != "vllm" or not engine["vllm"]["asynchronous"] or engine["continuous_batching"]["workers"] != 1:
        parser.error("paired single-question measurements require asynchronous vLLM and workers=1")
    if ar["algorithms"]["is"]["exploration_schedule"] != "two_stage":
        parser.error("configure the fast two_stage IS path; fixed-block performance comparisons are retired")
    if args.study.startswith("selection") and (ar["algorithms"]["is"]["candidate_count"] != 4
                                             or ar["algorithms"]["is"]["rollout_count"] != 1
                                             or not ar["algorithms"]["is"]["selection_ready"]):
        parser.error("selection study requires the current M=4, K=1, selection_ready=true baseline")
    if "bon-consilience" in modes:
        ar["algorithms"]["best_of_n"]["samples"] = 4
    uses_self_certainty = any(mode.endswith(SC_SUFFIX) for mode in modes)
    if uses_self_certainty:
        settings["rewards"]["self_certainty"]["scope"] = "thinking"
    resolved, _ = resolve_thinking(ar)
    if (resolved["output"]["thinking_mode"] != "enabled" or ar["output"]["sampling_scope"] != "thinking"
            or settings["rewards"]["consilience"]["scope"] != "thinking"):
        parser.error("this comparison requires thinking generation and thinking-only IS")
    # Count padded and canceled work through the engine's native execution statistics.
    engine["vllm"]["engine_kwargs"]["cudagraph_metrics"] = True
    validate_settings(settings)
    dataset = load_dataset(args.dataset, settings["datasets"][args.dataset])
    family, graph, admission = None, None, None
    original_run = ar_app.run_conditional_is
    try:
        if not dataset.problems:
            parser.error("the selected dataset is empty")
        initial_reward = "self_certainty" if uses_self_certainty else "consilience"
        family = ARFamily(settings, SimpleNamespace(algorithm="is", reward=initial_reward), dataset)
        family.load()
        if not family.raw.supports_streaming_generation:
            raise RuntimeError("the fast IS path requires native streaming")
        graph, admission = GraphMeasurements(family.raw), AdmissionMeasurements(family.raw)
        ar_app.run_conditional_is = partial(admission.measure_search, original_run)
        maximum = family.max_new_tokens
        try:
            family.max_new_tokens = 64
            for mode in modes:
                select_mode(family, mode, settings)
                family.solve(dataset.problems[0], SeedStream(SeedStream(settings["run"]["seed"]).derive("paired-warmup", mode)))
        finally:
            family.max_new_tokens = maximum
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as sink:
            def write(row):
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
            write({"kind": "metadata", "label": args.label, "expected_pairs": len(dataset.problems) * repeats,
                   "modes": list(modes),
                   "command_line": sys.argv[1:], "settings": settings, "dataset": dataset.describe(),
                   "engine": engine_settings(family.raw), "parameter_count": family.raw.parameter_count,
                   "git": git_state(), "source_sha256": json_sha256(source_sha256()), "environment": environment(),
                   "timing": "cold-prefix generation, including final content; excluding loading, warm-up, grading and writes"})
            run_pairs(family, dataset, settings, repeats, graph, admission, write, modes)
    finally:
        ar_app.run_conditional_is = original_run
        if graph is not None:
            graph.close()
        if admission is not None:
            admission.close()
        if family is not None:
            family.close()
        dataset.close()


if __name__ == "__main__":
    main()
