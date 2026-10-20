"""Single-question CUDA/NPU latency of the production conditional IS path.

Run in separate processes with PYTHONPATH pointing at each revision's src.
Model loading, warm-up, prefix-cache reset and disk writes are outside timing.
Only token digests and lengths are saved, never full rollout token histories.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import struct
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from inference_scaling.app.ar import ARFamily
from inference_scaling.arllm.algorithms import candidates
from inference_scaling.arllm.algorithms.conditional_is import ConditionalISAdapter
from inference_scaling.datasets.math500 import MATH500
from inference_scaling.datasets.gsm8k import GSM8K
from inference_scaling.shared.rng import SeedStream


class TimedSession:
    def __init__(self, inner, row):
        self.inner, self.row = inner, row
        self.requests = {}

    def submit(self, requests, *, phase="rollout", checkpoint=None):
        for job in requests:
            self.requests[job.request_id] = {
                "seed": job.seed, "limit": job.max_new_tokens, "prefix_tokens": len(job.prefix),
                "prefix_sha256": hashlib.sha256(struct.pack(f"<{len(job.prefix)}I", *job.prefix)).hexdigest(),
                "submitted": time.perf_counter() - self.row["started"], "tokens": 0,
                "digest": hashlib.sha256(), "events": 0,
            }
        return self.inner.submit(requests, phase=phase, checkpoint=checkpoint)

    def next_event(self):
        event = self.inner.next_event()
        job = self.requests[event.delta.request_id]
        tokens = event.delta.token_ids
        job["tokens"] += len(tokens)
        job["digest"].update(struct.pack(f"<{len(tokens)}I", *tokens))
        job["events"] += 1
        if event.finished:
            job["finished"] = time.perf_counter() - self.row["started"]
            job["reason"] = event.delta.finish_reason
            job["forward_token_slots"] = event.delta.forward_token_slots
        return event

    def seal(self):
        return self.inner.seal()

    def stop(self, request_id):
        return self.inner.stop(request_id)

    def close(self):
        try:
            self.inner.close()
        finally:
            for job in self.requests.values():
                if "finished" not in job:
                    job.update(finished=time.perf_counter() - self.row["started"],
                               reason="discarded", forward_token_slots=None)
            self.row["requests"] = {
                key: {**job, "digest": job["digest"].hexdigest()} for key, job in self.requests.items()
            }


class TimedBackend:
    def __init__(self, inner, row):
        self.inner, self.row = inner, row

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def open_generation_session(self):
        return TimedSession(self.inner.open_generation_session(), self.row)


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


def settings_for(args):
    settings = json.loads(Path(args.settings).read_text(encoding="utf-8"))
    ar = settings["ar"]
    ar["model"].update(path=args.model, revision=None, weight_sha256=None, adapter=None, tokenizer=None,
                       tokenizer_revision=None, local_files_only=True, token_penalty=None)
    ar["prompt"]["chat_template_kwargs"] = {}
    ar["output"].update(thinking_mode="enabled", sampling_scope="thinking")
    ar["sampling"].update(temperature=args.temperature, top_p=1.0, top_k=None)
    ar["engine"].update(backend="vllm", dtype="bfloat16", device=args.device, context_window=args.max_new_tokens + 1024)
    ar["engine"]["continuous_batching"]["workers"] = 1
    vllm = ar["engine"]["vllm"]
    vllm.update(asynchronous=True, max_model_len=args.max_new_tokens + 1024, max_num_seqs=16,
                max_num_batched_tokens=2048, gpu_memory_utilization=args.memory_utilization, enforce_eager=args.eager,
                enable_prefix_caching=True, fused_logprobs=False, exact_scoring="none", parameter_count=args.parameter_count,
                tensor_parallel_size=args.tensor_parallel_size, data_parallel_size=args.data_parallel_size,
                async_scheduling=args.async_scheduling, quantization=None,
                engine_kwargs={"enable_chunked_prefill": True, **json.loads(args.engine_kwargs)})
    ar["engine"]["rollout_scheduler"].update(enabled=args.execution == "stream", max_in_flight=16)
    ar["algorithms"]["is"].update(candidate_count=args.candidate_count,
                                  block_size=args.block_size, block_first=False,
                                  sibling_replay=args.sibling_replay, completion_fraction=args.completion_fraction)
    if args.completion_fraction < 1:
        # Request-level counters can miss buffered work at cancellation.
        vllm["engine_kwargs"]["cudagraph_metrics"] = True
    settings["run"]["seed"] = args.seed
    ds = settings["datasets"][args.dataset]
    ds["max_new_tokens"] = args.max_new_tokens
    ds["selection"]["count"] = args.questions
    if args.selection_skip is not None:
        ds["selection"]["skip"] = args.selection_skip
    settings["rewards"]["consilience"].update(temperature=2.0, score_temperature=args.temperature, scope="thinking",
                                               top_k=5, window_fraction=0.2, window_tokens=None,
                                               skip_fraction=0.05, initial_penalty=3.0)
    return settings


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


def measured_solve(family, problem, seed, graph_measurements):
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
    if execution and execution["graph_execution"]["forward_token_slots"] < counters["generation_forward_token_slots"]:
        raise RuntimeError("native execution statistics omitted observed work")
    return result, elapsed, counters, execution


def compare_runs(paths):
    """Compare complete IS runs; report execution speedup only for matching work."""
    def load(path):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        metadata = next(row for row in rows if row["kind"] == "metadata")
        if metadata.get("mode", "is") != "is":
            raise ValueError("only complete IS measurements are supported")
        return metadata, {(row["repeat"], row["k"], row["problem_id"]): row
                          for row in rows if row["kind"] == "measurement"}

    def signature(row):
        return (row["output_sha256"], tuple(
            (tuple(step["selection"]), tuple(
                (key, job["seed"], job["limit"], job["prefix_sha256"], job["tokens"], job["digest"], job["reason"])
                for key, job in sorted(step["requests"].items()))) for step in row["rounds"]))

    (left_meta, left), (right_meta, right) = map(load, paths)
    for name in ("model", "sampling", "prompt", "output", "algorithms"):
        if left_meta["settings"]["ar"][name] != right_meta["settings"]["ar"][name]:
            raise ValueError(f"comparison changed ar.{name}")
    if left_meta["settings"]["rewards"]["consilience"] != right_meta["settings"]["rewards"]["consilience"]:
        raise ValueError("comparison changed the reward")
    if left_meta["dataset"] != right_meta["dataset"]:
        raise ValueError("comparison changed the dataset or selected questions")
    if not left or left.keys() != right.keys():
        raise ValueError("comparison requires the same nonempty set of measurement keys")
    for key, before in left.items():
        after = right[key]
        same = (signature(before) == signature(after)
                and before["counters"]["generated_tokens"] == after["counters"]["generated_tokens"])
        weight_error = (max(abs(a - b) for x, y in zip(before["rounds"], after["rounds"], strict=True)
                            for a, b in zip(x["candidate_log_weights"], y["candidate_log_weights"], strict=True))
                        if same else None)
        print(json.dumps({
            "repeat": key[0], "k": key[1], "problem_id": key[2], "same_trajectories_and_selection": same,
            "max_log_weight_error": weight_error,
            "seconds": [before["seconds"], after["seconds"]],
            "speedup": before["seconds"] / after["seconds"] if same and weight_error < 1e-8 else None,
            "tail_seconds": [before["tail_seconds"], after["tail_seconds"]],
            **{name: [before["counters"][name], after["counters"][name]] for name in (
                "generated_tokens", "generation_forward_token_slots", "estimated_dense_forward_flops")},
        }))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model")
    parser.add_argument("--parameter-count", type=int, default=None,
                        help="unique model parameters for the 2P-per-token FLOPs estimate")
    parser.add_argument("--label")
    parser.add_argument("--settings", default="settings/inference.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("BEFORE", "AFTER"),
                        help="compare existing measurements on CPU; no model is loaded")
    parser.add_argument("--questions", type=int, default=3)
    parser.add_argument("--dataset", choices=("gsm8k", "math500"), default="gsm8k")
    parser.add_argument("--selection-skip", type=int, default=None, help="MATH-500 selection offset")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--completion-fraction", type=float, default=1.0,
                        help="stop at this mean completed fraction of candidate groups; below 1 is length-biased")
    parser.add_argument("--sibling-replay", action="store_true", help="reuse the selected block's other complete rollouts")
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261018)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--device", choices=("cuda", "npu"), default="cuda")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--memory-utilization", type=float, default=0.85)
    parser.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--with-base", action="store_true",
                        help="measure ordinary thinking-mode sampling once per question/repeat, without reward statistics")
    parser.add_argument("--execution", choices=("stream", "batch"), default="stream")
    parser.add_argument("--engine-kwargs", default="{}", help="JSON engine options applied equally to both profiles")
    args = parser.parse_args()
    if args.compare:
        compare_runs(args.compare)
        return
    if not all((args.model, args.label, args.output)):
        parser.error("--model, --label and --output are required for an accelerator run")
    if min(args.tensor_parallel_size, args.data_parallel_size) < 1 or not 0 < args.memory_utilization < 1:
        parser.error("parallel sizes must be positive and memory utilization must be between 0 and 1")
    rollout_counts = args.k
    if (min(args.questions, args.repeats, args.max_new_tokens, args.block_size, *rollout_counts, args.candidate_count) < 1
            or max(rollout_counts) > 4 or args.temperature <= 0):
        parser.error("positive budgets and K <= 4 are required")
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    if args.selection_skip is not None and (args.dataset != "math500" or args.selection_skip < 0):
        parser.error("--selection-skip requires MATH-500 and a nonnegative offset")
    if not 0 < args.completion_fraction <= 1:
        parser.error("completion fraction must be in (0, 1]")
    if args.completion_fraction < 1 and args.execution != "stream":
        parser.error("completion quotas require streaming")
    import torch
    if args.device == "npu":
        import torch_npu  # noqa: F401
    accelerator = getattr(torch, args.device, None)
    if accelerator is None or not accelerator.is_available():
        raise RuntimeError(f"this benchmark requires an available {args.device} accelerator")
    settings = settings_for(args)
    dataset = (GSM8K if args.dataset == "gsm8k" else MATH500)(settings["datasets"][args.dataset])
    family = ARFamily(settings, SimpleNamespace(algorithm="is", reward="consilience"), dataset)
    rounds = []
    active_round = None
    original_step = ConditionalISAdapter.step
    original_batch_session = candidates.BatchGenerationSession

    def timed_step(adapter, state, step_index, seeds):
        nonlocal active_round
        row = {"step": step_index, "fixed_tokens": state.fixed, "started": time.perf_counter()}
        measured = replace(adapter, backend=TimedBackend(adapter.backend, row))
        active_round = row
        try:
            record, updated = original_step(measured, state, step_index, seeds)
        finally:
            active_round = None
        row["seconds"] = time.perf_counter() - row.pop("started")
        # Batch barriers delay outer delivery, so use native request completion there.
        finishes = sorted(row.get("native_finishes", [job["finished"] for job in row["requests"].values()]))
        row["tail_seconds"] = finishes[-1] - finishes[-2] if len(finishes) >= 2 else 0.0
        row["single_request_seconds"] = single_request_seconds(row["requests"]) if args.execution == "stream" else None
        row["selection"] = [record.selected_index, record.completion_index]
        row["candidate_log_weights"] = [candidate.log_weight for candidate in record.candidates]
        row["completion_log_weights"] = [
            [completion.log_weight for completion in candidate.rollouts] for candidate in record.candidates
        ]
        row["sampling_plan"] = record.sampling_plan
        row["candidate_indices"] = record.candidate_indices
        if args.completion_fraction < 1:
            width, count, rollouts = record.sampling_plan
            sources = record.candidate_indices or tuple(range(count))
            expected = [1 if width >= adapter.config.total_length - state.fixed else rollouts] * count
            completed = [0] * count
            for index, candidate in zip(sources, record.candidates, strict=True):
                completed[index] = len(candidate.rollouts)
                if len(candidate.rollouts) == 1 and not candidate.rollouts[0].token_ids:
                    expected[index] = 1
            row["completion_quota"] = {
                "fraction": args.completion_fraction, "candidate_sources": sources,
                "unit": "mean_candidate_completion_fraction",
                "completed_fraction": sum(done / planned for done, planned in zip(
                    completed, expected, strict=True)) / count,
                "expected_leaves": expected, "completed_leaves": completed,
                "discarded_requests": [key for key, job in row["requests"].items() if job["reason"] == "discarded"],
                "sampling_law": "length-biased; incomplete leaves have zero weight",
            }
        row["retained_tokens"] = len(updated.token_ids)
        rounds.append(row)
        print(json.dumps({"event": "round", "label": args.label, "k": record.sampling_plan[2],
                          "sampling_plan": record.sampling_plan,
                          "step": step_index, "seconds": row["seconds"], "tail_seconds": row["tail_seconds"],
                          "retained_tokens": row["retained_tokens"]}), flush=True)
        return record, updated

    ConditionalISAdapter.step = timed_step
    candidates.BatchGenerationSession = lambda backend: TimedSession(original_batch_session(backend), backend.row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graph_measurements = None
    try:
        family.load()
        graph_measurements = GraphMeasurements(family.raw)
        original_generate_one = family.raw._generate_one

        async def timed_generate_one(*request_args, **request_kwargs):
            result = await original_generate_one(*request_args, **request_kwargs)
            if active_round is not None:
                active_round.setdefault("native_finishes", []).append(time.perf_counter() - active_round["started"])
            return result

        family.raw._generate_one = timed_generate_one
        if args.execution == "stream" and not family.raw.supports_streaming_generation:
            raise RuntimeError("a streaming vLLM backend is required")
        metadata = {"kind": "metadata", "mode": "is",
                    "label": args.label, "arguments": vars(args) | {"output": str(args.output)},
                    "settings": settings, "dataset": dataset.describe(), "engine": engine_settings(family.raw),
                    "parameter_count": family.raw.parameter_count, "gpu": accelerator.get_device_name(),
                    "device": args.device,
                    "algorithm_sha256": {
                        name: hashlib.sha256((Path(candidates.__file__).parent / name).read_bytes()).hexdigest()
                        for name in ("candidates.py", "conditional_is.py", "config.py")
                    },
                    "environment": {key: os.environ.get(key) for key in (
                        "VLLM_USE_V2_MODEL_RUNNER", "VLLM_WSL2_ENABLE_PIN_MEMORY", "VLLM_BATCH_INVARIANT", "OMP_NUM_THREADS",
                        "TORCHINDUCTOR_COMPILE_THREADS")},
                    "python": platform.python_version(), "cuda": torch.version.cuda,
                    "versions": {name: importlib.metadata.version(name) for name in (
                        "torch", "vllm", "transformers", *(("torch-npu", "vllm-ascend") if args.device == "npu" else ()))}}
        with args.output.open("x", encoding="utf-8") as sink:
            sink.write(json.dumps(metadata, ensure_ascii=False) + "\n")
            sink.flush()
            # Warm every branch multiplicity; no timed run reuses warm-up KV entries.
            for k in dict.fromkeys(rollout_counts):
                family.max_new_tokens = 64
                family.config.update(rollout_count=k, block_size=16)
                warmed = family.solve(dataset.problems[0], SeedStream(args.seed))
                if k == args.k[0]:
                    print(json.dumps({"event": "warmup", "text": warmed["output"]["text"][:300]}), flush=True)
            family.max_new_tokens = args.max_new_tokens
            family.config["block_size"] = args.block_size
            for repeat in range(args.repeats):
                base_seconds = {}
                if args.with_base:
                    original = family.choices, family.config, family.reward_settings
                    try:
                        family.choices = SimpleNamespace(algorithm="sample", reward=None)
                        family.config, family.reward_settings = family.ar["algorithms"]["sample"], None
                        for problem in dataset.problems:
                            result, elapsed, counters, execution = measured_solve(
                                family, problem, args.seed, graph_measurements)
                            base_seconds[problem.id] = elapsed
                            row = {"kind": "base", "label": args.label, "repeat": repeat,
                                   "problem_id": problem.id, "seconds": elapsed,
                                   "counters": counters, **execution,
                                   "output_tokens": result["output"]["tokens"],
                                   "output_sha256": hashlib.sha256(result["output"]["text"].encode()).hexdigest(),
                                   "answer_text": result["answer_text"],
                                   "thinking_status": result["output"]["thinking_status"],
                                   "ended_by_eos": result["output"]["ended_by_eos"],
                                   "length_exhausted": result["output"]["length_exhausted"],
                                   "fallbacks": result["fallbacks"], "cost": result["cost"]}
                            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                            sink.flush()
                            print(json.dumps({key: row[key] for key in (
                                "kind", "label", "problem_id", "seconds", "output_tokens", "counters")}), flush=True)
                    finally:
                        family.choices, family.config, family.reward_settings = original
                for k in args.k:
                    family.config["rollout_count"] = k
                    for problem in dataset.problems:
                        rounds.clear()
                        result, elapsed, counters, execution = measured_solve(
                            family, problem, args.seed, graph_measurements)
                        row = {"kind": "measurement", "label": args.label, "repeat": repeat, "k": k,
                               "problem_id": problem.id, "seconds": elapsed, "rounds": list(rounds),
                               "base_seconds": base_seconds.get(problem.id),
                               "relative_to_base": (elapsed / base_seconds[problem.id] if problem.id in base_seconds else None),
                               "search_seconds": sum(step["seconds"] for step in rounds),
                               "tail_seconds": sum(step["tail_seconds"] for step in rounds),
                               "single_request_seconds": (sum(step["single_request_seconds"] for step in rounds)
                                                          if args.execution == "stream" else None),
                               "round_tail_median": statistics.median(step["tail_seconds"] for step in rounds),
                               "counters": counters, "output_tokens": result["output"]["tokens"],
                               "output_sha256": hashlib.sha256(result["output"]["text"].encode()).hexdigest(),
                               "output_text": result["output"]["text"],
                               "answer_text": result["answer_text"],
                               "thinking_status": result["output"]["thinking_status"],
                               "ended_by_eos": result["output"]["ended_by_eos"],
                               "length_exhausted": result["output"]["length_exhausted"],
                               "fallbacks": result["fallbacks"], "cost": result["cost"], "trace": result["trace"]}
                        row.update(execution)
                        sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                        sink.flush()
                        print(json.dumps({key: row[key] for key in ("label", "k", "problem_id", "seconds",
                                                                   "relative_to_base", "search_seconds",
                                                                   "tail_seconds", "counters")}), flush=True)
    finally:
        ConditionalISAdapter.step = original_step
        candidates.BatchGenerationSession = original_batch_session
        if graph_measurements is not None:
            graph_measurements.close()
        family.close()
        dataset.close()


if __name__ == "__main__":
    main()
