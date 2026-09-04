"""Cross-request persistent fast-residual experiments for Stream-Uno."""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import time
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .fast_residual import FastResidualConfig
from .hf_online_uno import HfOnlineUnoRunner, OnlineRuntimeConfig
from .hf_uno import (
    ADAPTER_REVISION,
    ADAPTER_WEIGHT_SHA256,
    BASE_REVISION,
    BASE_WEIGHT_SHA256,
    _dtype,
    _parse_ints,
    _sha256,
    load_runtime,
)
from .torch_sampling import SamplingConfig


DEFAULT_PROMPT = (
    "Explain in three concise paragraphs why speculative decoding can be lossless."
)


def choose_snapshot(
    mean_tpf_ratios: dict[int, float],
    *,
    minimum_gain: float,
) -> dict[str, float | int | bool]:
    """Choose only from validation scores, falling back to the zero snapshot."""

    if 0 not in mean_tpf_ratios:
        raise ValueError("validation scores must include zero snapshot 0.")
    if minimum_gain < 0:
        raise ValueError("minimum_gain cannot be negative.")
    if not mean_tpf_ratios or any(
        index < 0 or not math.isfinite(score)
        for index, score in mean_tpf_ratios.items()
    ):
        raise ValueError("snapshot scores must be finite and non-negative indexed.")
    best_index, best_score = max(
        mean_tpf_ratios.items(),
        key=lambda item: (item[1], -item[0]),
    )
    threshold = 1.0 + minimum_gain
    selected_index = best_index if best_index > 0 and best_score >= threshold else 0
    return {
        "best_validation_snapshot": int(best_index),
        "best_validation_mean_tpf_ratio": float(best_score),
        "minimum_gain": float(minimum_gain),
        "selection_threshold": float(threshold),
        "selected_snapshot": int(selected_index),
        "nonzero_snapshot_selected": bool(selected_index > 0),
    }


def break_even_requests(
    training_cost_seconds: float,
    mean_future_saving_seconds: float,
) -> int | None:
    """Return future requests needed to amortize positive training cost."""

    if not math.isfinite(training_cost_seconds) or not math.isfinite(
        mean_future_saving_seconds
    ):
        raise ValueError("break-even inputs must be finite.")
    if mean_future_saving_seconds <= 0:
        return None
    return int(math.ceil(max(0.0, training_cost_seconds) / mean_future_saving_seconds))


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size:
        raise ValueError("summary requires at least one value.")
    return {
        "mean": float(np.mean(array)),
        "median": float(statistics.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _explicit_online_seconds(result: dict[str, Any]) -> float:
    diagnostics = result["diagnostics"]
    return sum(
        float(diagnostics[name])
        for name in (
            "update_seconds",
            "feedback_materialization_seconds",
            "head_forward_seconds",
            "candidate_head_forward_seconds",
        )
    )


def _frozen_config(
    *,
    block_size: int,
    feedback_top_k: int,
    fast: FastResidualConfig,
) -> OnlineRuntimeConfig:
    disabled_interval = 1_000_000_000
    return OnlineRuntimeConfig(
        block_size=block_size,
        update_stride=disabled_interval,
        feedback_top_k=feedback_top_k,
        supervision="on_policy",
        activation_mode="immediate",
        feedback_interval=disabled_interval,
        candidate_evaluation_interval=disabled_interval,
        fast=fast,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-prompt", action="append")
    parser.add_argument("--validation-prompt", action="append")
    parser.add_argument("--test-prompt", action="append")
    parser.add_argument("--training-requests", type=int, default=4)
    parser.add_argument("--validation-repetitions", type=int, default=2)
    parser.add_argument("--test-repetitions", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--warmup-tokens", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--update-stride", type=int, default=40)
    parser.add_argument("--feedback-interval", type=int, default=4)
    parser.add_argument("--feedback-top-k", type=int, default=50)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--selection-minimum-gain", type=float, default=0.002)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--mask-token-id", type=int, default=64256)
    parser.add_argument("--stop-token-ids", type=_parse_ints, default=[64019, 1])
    parser.add_argument("--ignore-stop", action="store_true")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--skip-hash-check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.training_requests < 1:
        raise ValueError("training_requests must be positive.")
    if args.validation_repetitions < 1 or args.test_repetitions < 2:
        raise ValueError("validation needs >=1 repetition and test needs >=2.")
    if args.max_new_tokens < 2:
        raise ValueError("max_new_tokens must be at least two.")

    model_path = args.model_path.resolve()
    adapter_path = args.adapter_path.resolve()
    base_weight = model_path / "model-00000-of-00001.safetensors"
    adapter_weight = adapter_path / "adapter_model.safetensors"
    for required in (base_weight, adapter_weight):
        if not required.is_file():
            raise FileNotFoundError(required)
    hashes = {
        "base_weight_sha256": _sha256(base_weight),
        "adapter_weight_sha256": _sha256(adapter_weight),
    }
    if not args.skip_hash_check:
        if hashes["base_weight_sha256"] != BASE_WEIGHT_SHA256:
            raise RuntimeError(
                "base checkpoint SHA-256 does not match the pinned revision."
            )
        if hashes["adapter_weight_sha256"] != ADAPTER_WEIGHT_SHA256:
            raise RuntimeError(
                "adapter checkpoint SHA-256 does not match the pinned revision."
            )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    sampling = SamplingConfig(
        temperature=float(args.temperature),
        top_k=int(args.top_k) if args.top_k > 0 else None,
        top_p=float(args.top_p),
    )
    runtime = load_runtime(
        model_path=model_path,
        adapter_path=adapter_path,
        device=device,
        dtype=_dtype(args.dtype),
        sampling=sampling,
        mask_token_id=args.mask_token_id,
        stop_token_ids=args.stop_token_ids,
        ignore_stop=args.ignore_stop,
    )
    runner = HfOnlineUnoRunner(runtime)
    train_prompts = args.train_prompt or [DEFAULT_PROMPT]
    validation_prompts = args.validation_prompt or [DEFAULT_PROMPT]
    test_prompts = args.test_prompt or [DEFAULT_PROMPT]
    encoded_train = [runtime.encode_prompt(prompt) for prompt in train_prompts]
    encoded_validation = [
        runtime.encode_prompt(prompt) for prompt in validation_prompts
    ]
    encoded_test = [runtime.encode_prompt(prompt) for prompt in test_prompts]
    routing_probe = runtime.routing_probe(
        encoded_train[0],
        block_size=args.block_size,
        seed=args.seed,
    )
    fast_config = FastResidualConfig(
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.learning_rate,
    )
    train_config = OnlineRuntimeConfig(
        block_size=args.block_size,
        update_stride=args.update_stride,
        feedback_top_k=args.feedback_top_k,
        supervision="on_policy",
        activation_mode="immediate",
        feedback_interval=args.feedback_interval,
        fast=fast_config,
    )
    frozen_config = _frozen_config(
        block_size=args.block_size,
        feedback_top_k=args.feedback_top_k,
        fast=fast_config,
    )

    runtime.generate_uno(
        encoded_train[0],
        max_new_tokens=max(2, args.warmup_tokens),
        block_size=args.block_size,
        seed=args.seed - 1,
    )
    runner.generate(
        encoded_train[0],
        max_new_tokens=max(args.block_size + 2, args.warmup_tokens),
        seed=args.seed - 1,
        initialization_seed=args.seed + 999_999,
        config=OnlineRuntimeConfig(
            block_size=args.block_size,
            update_stride=1,
            feedback_top_k=args.feedback_top_k,
            fast=fast_config,
        ),
    )

    persistent = runner.new_learner(
        train_config,
        initialization_seed=args.seed + 777_777,
    )
    snapshots = [persistent.clone()]
    training_runs: list[dict[str, Any]] = []
    for request_index in range(args.training_requests):
        prompt_index = request_index % len(encoded_train)
        input_ids = encoded_train[prompt_index]
        run_seed = args.seed + request_index
        static_first = request_index % 2 == 0
        static_metrics = None
        stream_result = None
        if static_first:
            static_metrics = runtime.generate_uno(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                block_size=args.block_size,
                seed=run_seed,
            )
        stream_result = runner.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            seed=run_seed,
            initialization_seed=args.seed + 888_888,
            config=train_config,
            persistent_learner=persistent,
        )
        if not static_first:
            static_metrics = runtime.generate_uno(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                block_size=args.block_size,
                seed=run_seed,
            )
        if static_metrics is None or stream_result is None:
            raise RuntimeError("training pair did not execute both methods.")
        snapshots.append(persistent.clone())
        training_runs.append(
            {
                "request_index": request_index,
                "prompt_index": prompt_index,
                "seed": run_seed,
                "order": ["static", "persistent_train"]
                if static_first
                else ["persistent_train", "static"],
                "static": asdict(static_metrics),
                "persistent_train": asdict(stream_result),
                "snapshot_after_request": request_index + 1,
            }
        )
        print(
            f"train request={request_index} "
            f"static_TPF={static_metrics.decoder_tokens_per_forward:.3f} "
            f"stream_TPF={stream_result.metrics.decoder_tokens_per_forward:.3f} "
            f"L2={stream_result.diagnostics.final_fast_weight_l2:.4f}",
            flush=True,
        )

    validation_static: dict[tuple[int, int], dict[str, Any]] = {}
    validation_runs: list[dict[str, Any]] = []
    snapshot_ratios: dict[int, list[float]] = {
        index: [] for index in range(len(snapshots))
    }
    for repetition in range(args.validation_repetitions):
        for prompt_index, input_ids in enumerate(encoded_validation):
            run_seed = args.seed + 100_000 + 1_000 * repetition + prompt_index
            static_metrics = runtime.generate_uno(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                block_size=args.block_size,
                seed=run_seed,
            )
            validation_static[(repetition, prompt_index)] = asdict(static_metrics)
            snapshot_order = list(range(len(snapshots)))
            shift = (repetition + prompt_index) % len(snapshot_order)
            snapshot_order = snapshot_order[shift:] + snapshot_order[:shift]
            for snapshot_index in snapshot_order:
                result = runner.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    seed=run_seed,
                    initialization_seed=args.seed + 999_999,
                    config=frozen_config,
                    persistent_learner=snapshots[snapshot_index],
                )
                ratio = (
                    result.metrics.decoder_tokens_per_forward
                    / static_metrics.decoder_tokens_per_forward
                )
                snapshot_ratios[snapshot_index].append(ratio)
                validation_runs.append(
                    {
                        "snapshot_index": snapshot_index,
                        "repetition": repetition,
                        "prompt_index": prompt_index,
                        "seed": run_seed,
                        "tpf_ratio_over_static": ratio,
                        "result": asdict(result),
                    }
                )
    validation_scores = {
        index: float(np.mean(ratios)) for index, ratios in snapshot_ratios.items()
    }
    selection = choose_snapshot(
        validation_scores,
        minimum_gain=args.selection_minimum_gain,
    )
    selected_index = int(selection["selected_snapshot"])
    selection["all_snapshot_mean_tpf_ratios"] = {
        str(index): score for index, score in validation_scores.items()
    }
    selection["all_snapshot_tpf_ratio_summaries"] = {
        str(index): _summary(ratios) for index, ratios in snapshot_ratios.items()
    }
    print(
        f"validation best={selection['best_validation_snapshot']} "
        f"ratio={selection['best_validation_mean_tpf_ratio']:.4f} "
        f"selected={selected_index}",
        flush=True,
    )

    selected = snapshots[selected_index].clone()
    del snapshots, persistent
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    test_runs: list[dict[str, Any]] = []
    test_tpf_ratios = []
    test_speed_ratios = []
    test_savings_seconds = []
    for repetition in range(args.test_repetitions):
        for prompt_index, input_ids in enumerate(encoded_test):
            run_seed = args.seed + 200_000 + 1_000 * repetition + prompt_index
            static_first = (repetition + prompt_index) % 2 == 0
            static_metrics = None
            frozen_result = None
            if static_first:
                static_metrics = runtime.generate_uno(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    block_size=args.block_size,
                    seed=run_seed,
                )
            frozen_result = runner.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                seed=run_seed,
                initialization_seed=args.seed + 999_999,
                config=frozen_config,
                persistent_learner=selected,
            )
            if not static_first:
                static_metrics = runtime.generate_uno(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    block_size=args.block_size,
                    seed=run_seed,
                )
            if static_metrics is None or frozen_result is None:
                raise RuntimeError("test pair did not execute both methods.")
            tpf_ratio = (
                frozen_result.metrics.decoder_tokens_per_forward
                / static_metrics.decoder_tokens_per_forward
            )
            speed_ratio = (
                frozen_result.metrics.decode_tokens_per_second
                / static_metrics.decode_tokens_per_second
            )
            saving = (
                static_metrics.decode_seconds - frozen_result.metrics.decode_seconds
            )
            test_tpf_ratios.append(tpf_ratio)
            test_speed_ratios.append(speed_ratio)
            test_savings_seconds.append(saving)
            test_runs.append(
                {
                    "repetition": repetition,
                    "prompt_index": prompt_index,
                    "seed": run_seed,
                    "order": ["static", "persistent_frozen"]
                    if static_first
                    else ["persistent_frozen", "static"],
                    "tpf_ratio": tpf_ratio,
                    "decode_tps_ratio": speed_ratio,
                    "serving_time_saving_seconds": saving,
                    "static": asdict(static_metrics),
                    "persistent_frozen": asdict(frozen_result),
                }
            )
            print(
                f"test rep={repetition} prompt={prompt_index} "
                f"TPF_ratio={tpf_ratio:.4f} TPS_ratio={speed_ratio:.4f}",
                flush=True,
            )

    observed_training_increment = sum(
        float(run["persistent_train"]["metrics"]["decode_seconds"])
        - float(run["static"]["decode_seconds"])
        for run in training_runs
    )
    instrumented_training_cost = sum(
        _explicit_online_seconds(run["persistent_train"]) for run in training_runs
    )
    mean_future_saving = float(np.mean(test_savings_seconds))
    analysis = {
        "validation_zero_snapshot_exact_tpf": all(
            math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-12)
            for value in snapshot_ratios[0]
        ),
        "test": {
            "pairs": len(test_runs),
            "paired_tpf_ratio": _summary(test_tpf_ratios),
            "paired_decode_tps_ratio": _summary(test_speed_ratios),
            "serving_time_saving_seconds": _summary(test_savings_seconds),
        },
        "amortization": {
            "observed_training_increment_seconds": observed_training_increment,
            "instrumented_training_cost_seconds": instrumented_training_cost,
            "mean_future_request_saving_seconds": mean_future_saving,
            "observed_break_even_future_requests": break_even_requests(
                observed_training_increment,
                mean_future_saving,
            ),
            "instrumented_break_even_future_requests": break_even_requests(
                instrumented_training_cost,
                mean_future_saving,
            ),
            "validation_benchmark_reruns_excluded": True,
        },
    }

    output = {
        "schema_version": 1,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "execution_backend": "huggingface_pytorch_kv_cache_stream_fast_residual",
        "claim_scope": {
            "sampling": "exact filtered linear Psi-Spec using each cycle's saved q",
            "state": "one persistent rank-r logit residual across ordered requests",
            "selection": "validation TPF only; test is not used for checkpoint choice",
            "performance": "actual wall-clock on Windows HF fallback, not Nano-vLLM",
        },
        "checkpoint": {
            "base_id": "IFM/K2-Horizon-0.9B",
            "base_revision": BASE_REVISION,
            "adapter_id": "IFM/K2-Horizon-0.9B-Uno",
            "adapter_revision": ADAPTER_REVISION,
            **hashes,
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "sampling": {
            **asdict(sampling),
            "mask_token_id": args.mask_token_id,
            "stop_token_ids": args.stop_token_ids,
            "ignore_stop": args.ignore_stop,
        },
        "design": {
            "train_prompts": train_prompts,
            "validation_prompts": validation_prompts,
            "test_prompts": test_prompts,
            "training_requests": args.training_requests,
            "validation_repetitions": args.validation_repetitions,
            "test_repetitions": args.test_repetitions,
            "max_new_tokens": args.max_new_tokens,
            "warmup_tokens": args.warmup_tokens,
            "block_size": args.block_size,
            "update_stride": args.update_stride,
            "feedback_interval": args.feedback_interval,
            "feedback_top_k": args.feedback_top_k,
            "selection_minimum_gain": args.selection_minimum_gain,
            "fast": asdict(fast_config),
            "seed": args.seed,
            "seed_partitions": {
                "train_offset": 0,
                "validation_offset": 100_000,
                "test_offset": 200_000,
            },
        },
        "routing_probe": routing_probe,
        "adapter_load_report": runtime.adapter_load_report,
        "training_runs": training_runs,
        "validation": {
            "static_runs": [
                {
                    "repetition": repetition,
                    "prompt_index": prompt_index,
                    "metrics": metrics,
                }
                for (repetition, prompt_index), metrics in sorted(
                    validation_static.items()
                )
            ],
            "snapshot_runs": validation_runs,
            "selection": selection,
        },
        "test_runs": test_runs,
        "analysis": analysis,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
