"""Temporal, fresh-request engineering pilot for Recycling Uno."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .hf_recycling_uno import HfRecyclingUnoRunner
from .hf_tree_uno import HfTreeUnoRunner
from .hf_replay_benchmark import _intervals
from .hf_uno import (
    ADAPTER_REVISION, ADAPTER_WEIGHT_SHA256, BASE_REVISION, BASE_WEIGHT_SHA256,
    _dtype, _package_version, _sha256, load_runtime,
)
from .recycling import RecyclingConfig
from .torch_sampling import SamplingConfig
from .tree_uno import TreeConfig
from .windows_execution import process_power_state


PILOT_WORKLOADS = (
    ("english", "Explain how a hash table handles collisions. Compare separate "
     "chaining with linear probing, and give one example of each."),
    ("chinese", "解释为什么数据库需要事务。用银行转账的具体例子说明原子性、一致性、"
     "隔离性和持久性，并解释常见的并发问题。"),
    ("code", "Write a Python function that merges overlapping intervals. Include "
     "type hints, explain its time complexity, and give three unit tests."),
    ("math", "Find the sum of the integers from 1 to 100. Derive the formula for "
     "the first n positive integers in two different ways and check small cases."),
)


def _method(value: str) -> tuple[str, RecyclingConfig | TreeConfig | None, int]:
    parts = value.split(":")
    if parts[0] in {"tree", "treeonline"} and len(parts) in {3, 4}:
        config = TreeConfig(
            block_size=int(parts[1]), nodes=int(parts[2]),
            top_k=int(parts[3]) if len(parts) == 4 else 4,
            online_rank=parts[0] == "treeonline",
        )
        config.validate()
        return value, config, config.block_size
    if parts[0] == "static" and len(parts) == 2:
        block = int(parts[1])
        if block < 2:
            raise ValueError("static width must be at least two")
        return value, None, block
    if parts[0] in {"warmstart", "scaled"} and len(parts) in {2, 3}:
        config = RecyclingConfig(
            block_size=int(parts[1]), policy=parts[0],
            noise_lora_scale=float(parts[2]) if len(parts) == 3 else 1.0,
        )
        config.validate()
        return value, config, config.block_size
    if parts[0] in {"always", "tps", "bounded", "disabled"} and len(parts) in {2, 3}:
        config = RecyclingConfig(
            block_size=int(parts[1]), policy=parts[0],
            max_recycle_depth=int(parts[2]) if len(parts) == 3 else 4,
        )
        config.validate()
        return value, config, config.block_size
    raise ValueError("method must be static:B or POLICY:B[:DEPTH]")


def gpu_snapshot() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu",
         "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    return result.stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", default="static:8,always:8,bounded:8:2,tps:8")
    parser.add_argument("--workloads", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--warmup-tokens", type=int, default=48)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20263005)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--scope", choices=["pilot", "confirmatory"], default="pilot")
    parser.add_argument("--background-note", default="unspecified")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--windows-disable-ecoqos", action="store_true")
    return parser


def summarize(records: list[dict], *, samples: int, seed: int) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["method"]].append(record)
    baseline_name = next(name for name in groups if name.startswith("static:"))
    baseline = {(r["workload"], r["seed"]): r for r in groups[baseline_name]}
    summary = {}
    for name, rows in groups.items():
        ratios, tpf, by_prompt = [], [], defaultdict(list)
        paired_base_seconds = []
        all_equal = True
        for row in rows:
            base = baseline.get((row["workload"], row["seed"]))
            if base is None:
                continue
            ratio = base["metrics"]["end_to_end_seconds"] / row["metrics"]["end_to_end_seconds"]
            ratios.append(ratio)
            by_prompt[row["workload"]].append(ratio)
            paired_base_seconds.append(base["metrics"]["end_to_end_seconds"])
            tpf.append(row["metrics"]["decoder_tokens_per_forward"] / base["metrics"]["decoder_tokens_per_forward"])
            all_equal &= row["metrics"]["output_token_ids"] == base["metrics"]["output_token_ids"]
        if not ratios:
            continue
        prompt_means = [float(np.mean(values)) for values in by_prompt.values()]
        total_seconds = sum(r["metrics"]["end_to_end_seconds"] for r in rows)
        summary[name] = {
            "pairs": len(ratios),
            "baseline": baseline_name,
            "absolute_e2e_tps": sum(r["metrics"]["output_tokens"] for r in rows) / total_seconds,
            "ratio_of_total_e2e_seconds": sum(paired_base_seconds) / total_seconds,
            "paired_e2e_speedup": _intervals(ratios, samples=samples, seed=seed),
            "prompt_cluster_mean_speedup": _intervals(prompt_means, samples=samples, seed=seed+1),
            "paired_tpf_ratio": _intervals(tpf, samples=samples, seed=seed+2),
            "greedy_token_ids_equal_to_baseline": all_equal,
            "per_workload_e2e_speedup": {k: float(np.mean(v)) for k, v in by_prompt.items()},
        }
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    execution_qos = process_power_state(disable_ecoqos=args.windows_disable_ecoqos)
    if args.repetitions < 1 or args.max_new_tokens < 2 or args.warmup_tokens < 2:
        raise ValueError("invalid repetition or token budget")
    methods = [_method(value) for value in args.methods.split(",")]
    if len({m[0] for m in methods}) != len(methods):
        raise ValueError("duplicate methods")
    if not any(m[0].startswith("static:") for m in methods):
        raise ValueError("one static baseline is required")
    hashes = {
        "base": _sha256(args.model_path / "model-00000-of-00001.safetensors"),
        "adapter": _sha256(args.adapter_path / "adapter_model.safetensors"),
    }
    if hashes != {"base": BASE_WEIGHT_SHA256, "adapter": ADAPTER_WEIGHT_SHA256}:
        raise RuntimeError("checkpoint hash differs from the immutable lock")
    workloads = (
        [(r["name"], r["prompt"]) for r in json.loads(args.workloads.read_text(encoding="utf-8"))]
        if args.workloads else PILOT_WORKLOADS
    )
    sampling = SamplingConfig(temperature=args.temperature, top_k=args.top_k, top_p=0.95)
    runtime = load_runtime(
        model_path=args.model_path.resolve(), adapter_path=args.adapter_path.resolve(),
        device=torch.device("cuda"), dtype=_dtype(args.dtype), sampling=sampling,
        mask_token_id=64256, stop_token_ids=[64019, 1], ignore_stop=True,
    )
    runner = HfRecyclingUnoRunner(runtime)
    tree_runner = HfTreeUnoRunner(runtime)
    encoded = [(name, prompt, runtime.encode_prompt(prompt)) for name, prompt in workloads]

    def run(method, ids, budget, seed):
        name, config, block = method
        if config is None:
            return {"metrics": asdict(runtime.generate_uno(
                ids, max_new_tokens=budget, block_size=block, seed=seed,
            )), "diagnostics": {}}
        if isinstance(config, TreeConfig):
            return asdict(tree_runner.generate(ids, max_new_tokens=budget, seed=seed, config=config))
        return asdict(runner.generate(
            ids, max_new_tokens=budget, seed=seed, config=config,
        ))

    for method in methods:
        run(method, encoded[0][2], args.warmup_tokens, args.seed - 1)
    # Deliberately warm every possible recycle length with ordinary base AR
    # forwards. These use disposable KV and are outside all timed results.
    for width in range(2, max(m[2] for m in methods) + 1):
        runtime.generate_uno(
            encoded[0][2], max_new_tokens=4, block_size=width, seed=args.seed - 2,
        )
    payload = {
        "schema_version": 2,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": args.scope,
        "backend": "Windows HF KV-cache; not official Nano-vLLM",
        "checkpoint": {"base_revision": BASE_REVISION, "adapter_revision": ADAPTER_REVISION, "sha256": hashes},
        "host": {
            "execution_qos": execution_qos,
            "platform": platform.platform(), "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
        },
        "design": {
            **vars(args), "model_path": str(args.model_path), "adapter_path": str(args.adapter_path),
            "output": str(args.output), "workloads": workloads,
            "sampling": asdict(sampling), "request_local": True,
            "method_order": "rotated and alternately reversed within paired seed",
            "fixed_output_tokens": True, "all_online_costs_inclusive": True,
            "e2e_timer": "full generation call, including init/close/text decode; excludes shared prompt encoding and benchmark I/O",
        },
        "records": [],
        "completed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    save()
    for prompt_index, (name, prompt, ids) in enumerate(encoded):
        for rep in range(args.repetitions):
            seed = args.seed + prompt_index * 100 + rep
            offset = (rep + prompt_index) % len(methods)
            order = methods[offset:] + methods[:offset]
            if (rep + prompt_index) % 2:
                order = list(reversed(order))
            for method in order:
                gpu_before = gpu_snapshot()
                call_start = time.perf_counter()
                result = run(method, ids, args.max_new_tokens, seed)
                torch.cuda.synchronize(runtime.device)
                call_seconds = time.perf_counter() - call_start
                metric = result["metrics"]
                metric["prefill_plus_decode_seconds"] = metric["end_to_end_seconds"]
                metric["end_to_end_seconds"] = call_seconds
                metric["end_to_end_tokens_per_second"] = metric["output_tokens"] / call_seconds
                payload["records"].append({
                    "method": method[0], "workload": name, "prompt": prompt,
                    "seed": seed, "method_order": [m[0] for m in order],
                    "gpu_before": gpu_before, "gpu_after": gpu_snapshot(), **result,
                })
                metric = result["metrics"]
                print(f'{name} seed={seed} {method[0]} '
                      f'TPS={metric["end_to_end_tokens_per_second"]:.2f} '
                      f'TPF={metric["decoder_tokens_per_forward"]:.3f}', flush=True)
                save()
    payload["summary"] = summarize(
        payload["records"], samples=args.bootstrap_samples, seed=args.seed,
    )
    if args.temperature > 0:
        for summary in payload["summary"].values():
            summary["greedy_token_ids_equal_to_baseline"] = None
    payload["completed"] = True
    save()
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
