"""Paired multi-benchmark ablation runner for converged IS, rewards, sampling, and engine presets.

Supports coding (HumanEval+ / LiveCodeBench via frozen public manifest; judged
offline by ``scripts/judge_coding.py``) and math (MATH-500 L5 / GSM8K graded
inline). Multiple arms sharing the same model/engine weights execute inside one
loaded backend instance with identical per-problem seeds for paired comparisons.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import json
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

# Allow importing sibling script helpers (coding_common) when executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from coding_common import PublicCodingDataset, digest, extract_code, read_jsonl
from inference_scaling.app.ar import ARFamily
from inference_scaling.app.settings import load_settings, validate_settings
from inference_scaling.datasets.gsm8k import GSM8K
from inference_scaling.datasets.math500 import MATH500
from inference_scaling.shared.rng import SeedStream


def _arm(
    name: str,
    *,
    algorithm: str = "is",
    reward: str | None = "self_certainty",
    sampling: dict[str, Any] | None = None,
    finish_sampling: dict[str, Any] | None = None,
    reward_overrides: dict[str, Any] | None = None,
    is_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "algorithm": algorithm,
        "reward": reward,
        "sampling": sampling,
        "finish_sampling": finish_sampling,
        "reward_overrides": reward_overrides or {},
        "is_overrides": is_overrides or {},
    }


SUITES: dict[str, list[dict[str, Any]]] = {
    # Suite 1: Global sampling policy comparison (run once per token_penalty setting).
    "global_sampling": [
        _arm("base_t10_full", algorithm="sample", reward=None,
             sampling={"temperature": 1.0, "top_p": 1.0, "top_k": None}),
        _arm("base_t10_p95_k20", algorithm="sample", reward=None,
             sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20}),
        _arm("base_t08_p95_k20", algorithm="sample", reward=None,
             sampling={"temperature": 0.8, "top_p": 0.95, "top_k": 20}),
        _arm("base_t07_p95_k20", algorithm="sample", reward=None,
             sampling={"temperature": 0.7, "top_p": 0.95, "top_k": 20}),
        _arm("base_t08_p95_k20_fin06", algorithm="sample", reward=None,
             sampling={"temperature": 0.8, "top_p": 0.95, "top_k": 20},
             finish_sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20}),
        _arm("adapt_k1_t10_full",
             sampling={"temperature": 1.0, "top_p": 1.0, "top_k": None},
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5},
             is_overrides={"candidate_count": 4, "rollout_count": 1, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True}),
        _arm("adapt_k1_t08_p95_k20",
             sampling={"temperature": 0.8, "top_p": 0.95, "top_k": 20},
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5},
             is_overrides={"candidate_count": 4, "rollout_count": 1, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True}),
        _arm("adapt_k1_t10_p95_k20",
             sampling={"temperature": 1.0, "top_p": 0.95, "top_k": 20},
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5},
             is_overrides={"candidate_count": 4, "rollout_count": 1, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True}),
    ],
    # Suite 2: Reward本体 refinements (strictly self_certainty & consilience) + dynamic IS schedule ablation.
    "rewards_and_is": [
        _arm("base", algorithm="sample", reward=None),
        _arm("two_stage_sr_sc",
             reward="self_certainty",
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 0.0},
             is_overrides={"candidate_count": 4, "rollout_count": 1, "block_size": 1024,
                           "exploration_schedule": "two_stage", "selection_ready": True, "selection": "sir"}),
        _arm("two_stage_bmet_k2_sc",
             reward="self_certainty",
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 0.0},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "two_stage", "selection_ready": False,
                           "selection": "block_metropolized"}),
        _arm("adapt_k1_sc_mean",
             reward="self_certainty",
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5},
             is_overrides={"candidate_count": 4, "rollout_count": 1, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None}),
        _arm("adapt_k1_sc_tail",
             reward="self_certainty",
             reward_overrides={"aggregation": "tail_mean", "tail_fraction": 0.25, "incomplete_penalty": 1.5},
             is_overrides={"candidate_count": 4, "rollout_count": 1, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None}),
        _arm("adapt_k1_sc_fwin",
             reward="self_certainty",
             reward_overrides={"aggregation": "final_window", "final_weight": 0.5, "incomplete_penalty": 1.5},
             is_overrides={"candidate_count": 4, "rollout_count": 1, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None}),
        _arm("adapt_k2_sc_mean",
             reward="self_certainty",
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
        _arm("adapt_k2_sc_fwin",
             reward="self_certainty",
             reward_overrides={"aggregation": "final_window", "final_weight": 0.5, "incomplete_penalty": 1.5},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
        _arm("adapt_k2_sc_tail",
             reward="self_certainty",
             reward_overrides={"aggregation": "tail_mean", "tail_fraction": 0.25, "incomplete_penalty": 1.5},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
        _arm("adapt_k2_cons_p15",
             reward="consilience",
             reward_overrides={"initial_penalty": 1.5, "incomplete_penalty": 2.0},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
        _arm("adapt_k2_p512_sc",
             reward="self_certainty",
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": 512,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
        _arm("adapt_k2_nopilot_sc",
             reward="self_certainty",
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": False, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
        _arm("adapt_k2_sc_fin06",
             reward="self_certainty",
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5},
             finish_sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
        _arm("adapt_k2_sc_b1536",
             reward="self_certainty",
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1536,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
        _arm("adapt_k2_cons_base",
             reward="consilience",
             reward_overrides={"initial_penalty": 3.0, "incomplete_penalty": 0.0},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
        _arm("adapt_k2_sc_t08",
             reward="self_certainty",
             sampling={"temperature": 0.8, "top_p": 0.95, "top_k": 20},
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
    ],
    # Suite 3: Fast compact comparison (for quick Colab GPU / TP=2 vs TP=4 / concurrency checks).
    "quick_pareto": [
        _arm("base", algorithm="sample", reward=None),
        _arm("two_stage_sr_sc",
             reward="self_certainty",
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 0.0},
             is_overrides={"candidate_count": 4, "rollout_count": 1, "block_size": 1024,
                           "exploration_schedule": "two_stage", "selection_ready": True, "selection": "sir"}),
        _arm("adapt_k1_sc",
             reward="self_certainty",
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5},
             is_overrides={"candidate_count": 4, "rollout_count": 1, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True}),
        _arm("adapt_k2_sc",
             reward="self_certainty",
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5, "temperature_normalized": True},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True}),
    ],
    # Suite 4: Adaptive IS v2 (temperature-normalized SC + B/2 pilot + Stage-2 pruning & M-1 cap).
    "adaptive_v2": [
        _arm("base", algorithm="sample", reward=None),
        _arm("adapt_v2_k1_sc",
             reward="self_certainty",
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5, "temperature_normalized": True},
             is_overrides={"candidate_count": 4, "rollout_count": 1, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
        _arm("adapt_v2_k2_sc",
             reward="self_certainty",
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5, "temperature_normalized": True},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
        _arm("adapt_v2_k2_sc_fwin",
             reward="self_certainty",
             reward_overrides={"aggregation": "final_window", "final_weight": 0.5,
                               "incomplete_penalty": 1.5, "temperature_normalized": True},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
        _arm("adapt_v2_k2_sc_t08",
             reward="self_certainty",
             sampling={"temperature": 0.8, "top_p": 0.95, "top_k": 20},
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5, "temperature_normalized": True},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
        _arm("adapt_v2_k2_cons_p15",
             reward="consilience",
             reward_overrides={"initial_penalty": 1.5, "incomplete_penalty": 2.0},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1024,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
        _arm("adapt_v2_k2_b1536",
             reward="self_certainty",
             reward_overrides={"aggregation": "mean", "incomplete_penalty": 1.5, "temperature_normalized": True},
             is_overrides={"candidate_count": 4, "rollout_count": 2, "block_size": 1536,
                           "exploration_schedule": "adaptive", "selection_ready": False,
                           "selection": "block_metropolized", "pilot_gate": True, "pilot_tokens": None,
                           "selective_lookahead": True, "straggler_pruning": True, "adaptive_temperature": True}),
    ],
}


def apply_arm(
    family: ARFamily,
    arm: dict[str, Any],
    base_settings: dict[str, Any],
    *,
    is_coding: bool = False,
) -> None:
    """Switch algorithm, reward, sampling, domain prompt/budget, and IS config on one loaded ARFamily."""
    ar = copy.deepcopy(base_settings["ar"])
    if is_coding:
        ar["max_new_tokens"] = 8192
        ar["thinking"]["reasoning_effort"] = "low"
        ar["prompt"]["system"] = (
            "You are an expert Python programmer. You will be given a question (problem specification) "
            "and will generate a correct Python program that matches the specification and passes all tests."
        )
    else:
        ar["max_new_tokens"] = 12288
        ar["thinking"]["reasoning_effort"] = "medium"
        ar["prompt"]["system"] = None
    if arm["sampling"] is not None:
        ar["sampling"].update(arm["sampling"])
    if arm["finish_sampling"] is not None:
        ar["output"]["finish_sampling"] = dict(arm["finish_sampling"])
    else:
        ar["output"].pop("finish_sampling", None)
    if arm["is_overrides"]:
        ar["algorithms"]["is"].update(arm["is_overrides"])
    from inference_scaling.shared.model.thinking import resolve_thinking
    resolved_ar, resolved_thinking = resolve_thinking(ar)
    family.ar = resolved_ar
    family.thinking = resolved_thinking
    family.max_new_tokens = resolved_ar["max_new_tokens"]
    family.choices = SimpleNamespace(algorithm=arm["algorithm"], model="ar", reward=arm["reward"])
    family.config = resolved_ar["algorithms"][arm["algorithm"]]
    if arm["reward"] is None:
        family.reward_settings = None
    else:
        r_cfg = copy.deepcopy(base_settings["rewards"][arm["reward"]])
        r_cfg.update(arm["reward_overrides"])
        # Keep score_temperature synchronized with global sampling temperature for on-device scalar statistics.
        r_cfg["score_temperature"] = float(resolved_ar["sampling"]["temperature"])
        family.reward_settings = r_cfg


def load_one_dataset(
    name: str,
    settings: dict[str, Any],
    *,
    manifest: Path | None = None,
    dataset_path: Path | None = None,
    shard: int = 0,
    shards: int = 1,
    limit: int | None = None,
    skip: int | None = None,
):
    if name in {"humanevalplus", "livecodebench", "coding"}:
        if not manifest:
            raise ValueError("--manifest is required for coding datasets")
        ds = PublicCodingDataset(manifest, shard=shard, shards=shards)
        if name != "coding":
            filtered = tuple(p for p in ds.problems if p.metadata["dataset"] == name)
            ds = SimpleNamespace(
                name=name, oracle=False, problems=filtered, rows=ds.rows,
                source_sha256=ds.source_sha256, prompt=ds.prompt, grade=ds.grade, close=lambda: None,
            )
        if limit is not None:
            ds = SimpleNamespace(
                name=ds.name, oracle=ds.oracle, problems=tuple(ds.problems[:limit]),
                rows=ds.rows, source_sha256=ds.source_sha256, prompt=ds.prompt, grade=ds.grade, close=lambda: None,
            )
        return ds, True
    if name == "math500":
        cfg = copy.deepcopy(settings["datasets"]["math500"])
        if dataset_path:
            cfg["path"] = str(dataset_path)
            cfg["download"] = False
        if limit is not None:
            cfg["selection"]["count"] = limit
        if skip is not None:
            cfg["selection"]["skip"] = skip
        return MATH500(cfg), False
    if name == "gsm8k":
        cfg = copy.deepcopy(settings["datasets"]["gsm8k"])
        if dataset_path:
            cfg["path"] = str(dataset_path)
            cfg["download"] = False
        if limit is not None:
            cfg["selection"]["count"] = limit
        return GSM8K(cfg), False
    raise ValueError(f"unsupported dataset: {name}")


def summarize_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("kind") == "measurement" and row.get("status") == "complete":
            by_key.setdefault((row.get("dataset", "unknown"), row["mode"]), []).append(row)
    summary: dict[str, Any] = {}
    for (ds_name, arm_name), items in sorted(by_key.items()):
        n = len(items)
        secs = [r["seconds"] for r in items]
        slots = [r["cost"]["forward_token_slots"] for r in items if r.get("cost") and r["cost"].get("forward_token_slots") is not None]
        flops = [r["cost"]["flops"] for r in items if r.get("cost") and r["cost"].get("flops") is not None]
        graded = [r["correct"] for r in items if r.get("correct") is not None]
        pilot_exits = sum(
            1 for r in items
            if (steps := r.get("trace", {}).get("steps")) and len(steps) == 1
            and len(steps[0].get("candidate_log_weights", [])) == 1
        )
        summary.setdefault(ds_name, {})[arm_name] = {
            "n": n,
            "correct": sum(bool(x) for x in graded) if graded else None,
            "accuracy": (sum(bool(x) for x in graded) / len(graded)) if graded else None,
            "mean_seconds": sum(secs) / n,
            "total_seconds": sum(secs),
            "mean_output_tokens": sum(r["output_tokens"] for r in items) / n,
            "mean_slots": (sum(slots) / len(slots)) if slots else None,
            "total_flops": sum(flops) if flops else None,
            "length_exhausted": sum(bool(r.get("length_exhausted")) for r in items),
            "incomplete_thinking": sum(r.get("thinking_status") not in {"complete", "disabled", None} for r in items),
            "pilot_early_exits": pilot_exits,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--suite", choices=sorted(SUITES), required=True)
    parser.add_argument("--arms", nargs="*", help="optional subset of arm names inside --suite")
    parser.add_argument("--dataset", choices=("humanevalplus", "livecodebench", "coding", "math500", "gsm8k"))
    parser.add_argument("--datasets", nargs="*", help="multi-dataset spec like gsm8k:16 humanevalplus:24 livecodebench:16 math500:16")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip", type=int)
    parser.add_argument("--concurrency", type=int, default=1, help="concurrent problems per arm")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.dataset and not args.datasets:
        parser.error("provide either --dataset or --datasets")

    settings = load_settings(args.settings)
    validate_settings(settings)
    arms = SUITES[args.suite]
    if args.arms:
        wanted = set(args.arms)
        arms = [a for a in arms if a["name"] in wanted]
        if not arms:
            raise ValueError("no matching arms selected")

    dataset_specs: list[tuple[str, int | None]] = []
    if args.datasets:
        for spec in args.datasets:
            if ":" in spec:
                ds_name, lim = spec.split(":", 1)
                dataset_specs.append((ds_name, int(lim)))
            else:
                dataset_specs.append((spec, args.limit))
    else:
        dataset_specs.append((args.dataset, args.limit))

    loaded_datasets = [
        (ds_name, *load_one_dataset(
            ds_name, settings, manifest=args.manifest, dataset_path=args.dataset_path,
            shard=args.shard, shards=args.shards, limit=lim, skip=args.skip,
        ))
        for ds_name, lim in dataset_specs
    ]
    total_problems = sum(len(ds.problems) for _, ds, _ in loaded_datasets)
    if total_problems == 0:
        raise ValueError("empty problem set")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(args.output)
    done = {
        (r.get("dataset"), r["problem_id"], r["mode"])
        for r in existing if r.get("kind") == "measurement" and r.get("status") == "complete"
    }
    expected = total_problems * len(arms)
    if len(done) >= expected:
        print(json.dumps({"kind": "already_complete", "summary": summarize_results(existing)}, indent=2), flush=True)
        return

    # Load the 27B engine ONCE and reuse it across all datasets and arms.
    init_settings = copy.deepcopy(settings)
    init_settings["ar"]["algorithms"]["is"]["selection_ready"] = False
    if args.concurrency > 1:
        init_settings["ar"]["engine"]["continuous_batching"]["workers"] = args.concurrency
    family = ARFamily(init_settings, SimpleNamespace(algorithm="sample", model="ar", reward=None), loaded_datasets[0][1])
    lock = threading.Lock()
    grade_lock = threading.Lock()
    try:
        family.load()
        write_mode = "a" if args.output.exists() else "w"
        with args.output.open(write_mode, encoding="utf-8") as sink:
            def write(row: dict[str, Any]) -> None:
                with lock:
                    sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                    sink.flush()
                    os.fsync(sink.fileno())

            if not existing:
                write({
                    "kind": "metadata",
                    "suite": args.suite,
                    "datasets": [{ds_name: [p.id for p in ds.problems]} for ds_name, ds, _ in loaded_datasets],
                    "arms": [a["name"] for a in arms],
                    "settings": settings,
                    "capabilities": family.pipeline.capabilities().to_dict() if family.pipeline else {},
                    "execution_config": family.execution_config(),
                    "concurrency": args.concurrency,
                })

            # Short warmup on first problem so graphs/caches are ready before timed runs.
            first_ds, first_is_coding = loaded_datasets[0][1], loaded_datasets[0][2]
            family.dataset = first_ds
            apply_arm(family, arms[0], settings, is_coding=first_is_coding)
            saved_max = family.max_new_tokens
            try:
                family.max_new_tokens = 64
                family.solve(first_ds.problems[0], SeedStream(SeedStream(settings["run"]["seed"]).derive("warmup", 0)))
            finally:
                family.max_new_tokens = saved_max

            for ds_name, dataset, is_coding in loaded_datasets:
                family.dataset = dataset

                def run_one(problem, arm, _ds=dataset, _ds_name=ds_name, _is_coding=is_coding):
                    seed = SeedStream(settings["run"]["seed"]).derive("ablation-benchmark", problem.id, 0)
                    t0 = time.perf_counter()
                    solved = family.solve(problem, SeedStream(seed))
                    elapsed = time.perf_counter() - t0
                    answer_text = solved["answer_text"]
                    actual_ds = problem.metadata.get("dataset", _ds_name) if isinstance(problem.metadata, dict) else _ds_name
                    row: dict[str, Any] = {
                        "kind": "measurement",
                        "status": "complete",
                        "problem_id": problem.id,
                        "dataset": actual_ds,
                        "mode": arm["name"],
                        "seed": seed,
                        "seconds": elapsed,
                        "prompt_tokens": solved["prompt_tokens"],
                        "output_tokens": solved["output"]["tokens"],
                        "thinking_status": solved["output"]["thinking_status"],
                        "length_exhausted": solved["output"]["length_exhausted"],
                        "selected_reward": solved["reward"],
                        "cost": solved["cost"],
                        "fallbacks": solved["fallbacks"],
                        "trace": solved["trace"],
                    }
                    if _is_coding:
                        code = extract_code(answer_text)
                        row["submitted_code"] = code
                        row["submission_sha256"] = digest(code)
                        row["parseable"] = bool(code)
                        row["correct"] = None
                    else:
                        with grade_lock:
                            grade = _ds.grade(answer_text, problem)
                        row["answer"] = grade.answer
                        row["parseable"] = grade.parseable
                        row["correct"] = grade.correct
                    return row

                for arm in arms:
                    pending_problems = [
                        p for p in dataset.problems
                        if ((p.metadata.get("dataset", ds_name) if isinstance(p.metadata, dict) else ds_name), p.id, arm["name"]) not in done
                    ]
                    if not pending_problems:
                        continue
                    apply_arm(family, arm, settings, is_coding=is_coding)
                    arm_t0 = time.perf_counter()
                    arm_cost_before = family.cost_snapshot()
                    if args.concurrency <= 1:
                        for problem in pending_problems:
                            row = run_one(problem, arm)
                            write(row)
                            done.add((row["dataset"], problem.id, arm["name"]))
                            print(json.dumps({
                                "dataset": row["dataset"], "mode": arm["name"], "problem_id": problem.id,
                                "seconds": round(row["seconds"], 2), "tokens": row["output_tokens"],
                                "correct": row["correct"], "done": len(done), "expected": expected,
                            }), flush=True)
                    else:
                        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                            futures = {pool.submit(run_one, p, arm): p for p in pending_problems}
                            for future in as_completed(futures):
                                row = future.result()
                                write(row)
                                done.add((row["dataset"], row["problem_id"], arm["name"]))
                                print(json.dumps({
                                    "dataset": row["dataset"], "mode": arm["name"], "problem_id": row["problem_id"],
                                    "seconds": round(row["seconds"], 2), "tokens": row["output_tokens"],
                                    "correct": row["correct"], "done": len(done), "expected": expected,
                                }), flush=True)
                    arm_cost_after = family.cost_snapshot()
                    write({
                        "kind": "arm_complete", "dataset": ds_name, "mode": arm["name"],
                        "wall_seconds": time.perf_counter() - arm_t0, "concurrency": args.concurrency,
                        "arm_forward_token_slots": arm_cost_after["forward_token_slots"] - arm_cost_before["forward_token_slots"],
                        "arm_flops": arm_cost_after["flops"] - arm_cost_before["flops"],
                        "arm_unmeasured_work_events": (
                            arm_cost_after["unmeasured_work_events"] - arm_cost_before["unmeasured_work_events"]
                        ),
                    })

            all_rows = read_jsonl(args.output)
            summary = summarize_results(all_rows)
            write({"kind": "generation_complete", "summary": summary, "time": time.time()})
            print(json.dumps({"kind": "summary", "summary": summary}, indent=2), flush=True)
    finally:
        family.close()
        for _, ds, _ in loaded_datasets:
            ds.close()


if __name__ == "__main__":
    main()
