"""Engineering benchmark for one-request-trained Verifier-Replay Uno.

The benchmark deliberately separates the cache-building request from future
frozen-cache evaluation requests.  It reports both response latency and the
one-time cache-indexing cost, and labels all wall-clock results as Hugging Face
fallback measurements rather than official Nano-vLLM results.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .hf_replay_uno import (
    HfReplayUnoRunner,
    ReplayRunResult,
    ReplayRuntimeConfig,
)
from .hf_uno import (
    ADAPTER_REVISION,
    ADAPTER_WEIGHT_SHA256,
    BASE_REVISION,
    BASE_WEIGHT_SHA256,
    HfUnoRuntime,
    _dtype,
    _package_version,
    _parse_ints,
    _sha256,
    load_runtime,
)
from .replay_cache import (
    CostAwareReplayRouter,
    ReplayCacheConfig,
    ReplayRouteConfig,
    VerifierReplayCache,
)
from .stage2_analysis import bootstrap_interval
from .torch_sampling import SamplingConfig


DEFAULT_PROMPT = (
    "Explain why speculative decoding can be lossless. Give a concise derivation "
    "of the acceptance and residual-correction rule, then state its assumptions."
)


def _run_result_dict(result: ReplayRunResult) -> dict[str, Any]:
    diagnostics = asdict(result.diagnostics)
    diagnostics["replay_tokens_per_forward"] = (
        result.diagnostics.replay_tokens_per_forward
    )
    diagnostics["static_tokens_per_forward"] = (
        result.diagnostics.static_tokens_per_forward
    )
    return {
        "metrics": asdict(result.metrics),
        "diagnostics": diagnostics,
    }


def _intervals(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be non-empty and finite.")
    return {
        "mean": bootstrap_interval(
            array,
            statistic=np.mean,
            samples=samples,
            seed=seed,
        ),
        "median": bootstrap_interval(
            array,
            statistic=np.median,
            samples=samples,
            seed=seed + 1,
        ),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _break_even_requests(*, one_time_overhead: float, mean_future_saving: float) -> int | None:
    if not math.isfinite(one_time_overhead) or not math.isfinite(mean_future_saving):
        raise ValueError("break-even inputs must be finite.")
    if one_time_overhead <= 0:
        return 0
    if mean_future_saving <= 0:
        return None
    return int(math.ceil(one_time_overhead / mean_future_saving))


def _namespace(*, sampling: SamplingConfig) -> str:
    return (
        f"IFM/K2-Horizon-0.9B@{BASE_REVISION}|"
        f"uno@{ADAPTER_REVISION}|temperature={sampling.temperature:g}|"
        f"top_k={sampling.top_k}|top_p={sampling.top_p}"
    )


def _new_runner(
    runtime: HfUnoRuntime,
    *,
    sampling: SamplingConfig,
    block_size: int,
    min_suffix_length: int,
    max_suffix_length: int,
) -> HfReplayUnoRunner:
    namespace = _namespace(sampling=sampling)
    cache = VerifierReplayCache(
        namespace=namespace,
        config=ReplayCacheConfig(
            min_suffix_length=min_suffix_length,
            max_suffix_length=max_suffix_length,
            max_continuation_length=block_size - 1,
            min_observations=1,
            min_confidence=0.75,
        ),
    )
    router = CostAwareReplayRouter(
        namespace=namespace,
        config=ReplayRouteConfig(
            min_match_length=min_suffix_length,
            min_proposal_tokens=1,
            min_cache_confidence=0.75,
            exploration_trials_per_match_length=1,
            probe_interval=64,
            ema_decay=0.9,
            throughput_margin=0.02,
        ),
    )
    return HfReplayUnoRunner(
        runtime,
        replay_cache=cache,
        router=router,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=24)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--mask-token-id", type=int, default=64256)
    parser.add_argument("--stop-token-ids", type=_parse_ints, default=[64019, 1])
    parser.add_argument(
        "--ignore-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-suffix-length", type=int, default=8)
    parser.add_argument("--max-suffix-length", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--skip-hash-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.block_size < 2:
        raise ValueError("replay benchmark requires block_size >= 2.")
    if args.max_new_tokens < 2 or args.warmup_tokens < 2:
        raise ValueError("token budgets must be at least two.")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive.")
    if args.bootstrap_samples < 1_000:
        raise ValueError("use at least 1,000 bootstrap resamples.")
    if args.min_suffix_length < 1 or args.max_suffix_length < args.min_suffix_length:
        raise ValueError("suffix bounds are invalid.")

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
            raise RuntimeError("base checkpoint SHA-256 does not match the lock.")
        if hashes["adapter_weight_sha256"] != ADAPTER_WEIGHT_SHA256:
            raise RuntimeError("Uno adapter SHA-256 does not match the lock.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
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
    prompts = args.prompt or [DEFAULT_PROMPT]
    encoded_prompts = [runtime.encode_prompt(prompt) for prompt in prompts]
    routing_probe = runtime.routing_probe(
        encoded_prompts[0],
        block_size=min(args.block_size, 8),
        seed=args.seed - 2,
    )

    # Warm AR, static Uno, and the replay fast path with disposable state.
    runtime.generate_ar(
        encoded_prompts[0],
        max_new_tokens=args.warmup_tokens,
        seed=args.seed - 1,
    )
    runtime.generate_uno(
        encoded_prompts[0],
        max_new_tokens=args.warmup_tokens,
        block_size=args.block_size,
        seed=args.seed - 1,
    )
    warm_runner = _new_runner(
        runtime,
        sampling=sampling,
        block_size=args.block_size,
        min_suffix_length=args.min_suffix_length,
        max_suffix_length=args.max_suffix_length,
    )
    warm_runner.generate(
        encoded_prompts[0],
        max_new_tokens=args.warmup_tokens,
        seed=args.seed - 1,
        config=ReplayRuntimeConfig(block_size=args.block_size),
    )
    warm_runner.generate(
        encoded_prompts[0],
        max_new_tokens=args.warmup_tokens,
        seed=args.seed - 1,
        config=ReplayRuntimeConfig(
            block_size=args.block_size,
            observe_after_request=False,
        ),
    )

    prompt_results: list[dict[str, Any]] = []
    all_tpf_ratios: list[float] = []
    all_decode_tps_ratios: list[float] = []
    all_e2e_tps_ratios: list[float] = []
    all_request_savings: list[float] = []
    all_output_equal = True

    for prompt_index, input_ids in enumerate(encoded_prompts):
        runner = _new_runner(
            runtime,
            sampling=sampling,
            block_size=args.block_size,
            min_suffix_length=args.min_suffix_length,
            max_suffix_length=args.max_suffix_length,
        )
        train_seed = args.seed + 100_000 * prompt_index
        ar = runtime.generate_ar(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            seed=train_seed,
        )
        train_static = runtime.generate_uno(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            seed=train_seed,
        )
        train_replay = runner.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            seed=train_seed,
            config=ReplayRuntimeConfig(block_size=args.block_size),
        )
        train_equal = train_static.output_token_ids == train_replay.metrics.output_token_ids
        if sampling.temperature <= 0:
            train_equal = train_equal and ar.output_token_ids == train_static.output_token_ids
        all_output_equal &= train_equal

        pairs: list[dict[str, Any]] = []
        tpf_ratios: list[float] = []
        decode_tps_ratios: list[float] = []
        e2e_tps_ratios: list[float] = []
        request_savings: list[float] = []
        for repetition in range(args.repetitions):
            run_seed = train_seed + 1_000 + repetition
            if repetition % 2 == 0:
                static = runtime.generate_uno(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    block_size=args.block_size,
                    seed=run_seed,
                )
                replay = runner.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    seed=run_seed,
                    config=ReplayRuntimeConfig(
                        block_size=args.block_size,
                        observe_after_request=False,
                    ),
                )
                method_order = ["static", "replay"]
            else:
                replay = runner.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    seed=run_seed,
                    config=ReplayRuntimeConfig(
                        block_size=args.block_size,
                        observe_after_request=False,
                    ),
                )
                static = runtime.generate_uno(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    block_size=args.block_size,
                    seed=run_seed,
                )
                method_order = ["replay", "static"]

            output_equal = static.output_token_ids == replay.metrics.output_token_ids
            if sampling.temperature <= 0:
                output_equal = output_equal and ar.output_token_ids == static.output_token_ids
            all_output_equal &= output_equal
            tpf_ratio = (
                replay.metrics.decoder_tokens_per_forward
                / static.decoder_tokens_per_forward
            )
            decode_tps_ratio = (
                replay.metrics.decode_tokens_per_second
                / static.decode_tokens_per_second
            )
            e2e_tps_ratio = (
                replay.metrics.end_to_end_tokens_per_second
                / static.end_to_end_tokens_per_second
            )
            request_saving = static.end_to_end_seconds - replay.metrics.end_to_end_seconds
            tpf_ratios.append(tpf_ratio)
            decode_tps_ratios.append(decode_tps_ratio)
            e2e_tps_ratios.append(e2e_tps_ratio)
            request_savings.append(request_saving)
            pairs.append(
                {
                    "repetition": repetition,
                    "seed": run_seed,
                    "method_order": method_order,
                    "output_token_ids_equal": output_equal,
                    "tpf_ratio": tpf_ratio,
                    "decode_tps_ratio": decode_tps_ratio,
                    "end_to_end_tps_ratio": e2e_tps_ratio,
                    "end_to_end_saving_seconds": request_saving,
                    "static": asdict(static),
                    "replay": _run_result_dict(replay),
                }
            )
            print(
                f"prompt={prompt_index} rep={repetition} "
                f"TPF={tpf_ratio:.3f}x decode={decode_tps_ratio:.3f}x "
                f"replay_cycles={replay.diagnostics.replay_cycles} "
                f"static_cycles={replay.diagnostics.static_cycles}",
                flush=True,
            )

        one_time_hybrid_seconds = (
            train_replay.metrics.end_to_end_seconds
            + train_replay.diagnostics.cache_update_seconds
        )
        one_time_static_seconds = train_static.end_to_end_seconds
        one_time_overhead = one_time_hybrid_seconds - one_time_static_seconds
        mean_future_saving = float(statistics.fmean(request_savings))
        break_even = _break_even_requests(
            one_time_overhead=one_time_overhead,
            mean_future_saving=mean_future_saving,
        )
        cumulative_static = one_time_static_seconds + sum(
            pair["static"]["end_to_end_seconds"] for pair in pairs
        )
        cumulative_replay = one_time_hybrid_seconds + sum(
            pair["replay"]["metrics"]["end_to_end_seconds"] for pair in pairs
        )

        prompt_results.append(
            {
                "prompt_index": prompt_index,
                "prompt": prompts[prompt_index],
                "prompt_tokens": int(input_ids.size(1)),
                "ar_reference": asdict(ar),
                "cache_build_request": {
                    "output_token_ids_equal": train_equal,
                    "static": asdict(train_static),
                    "hybrid": _run_result_dict(train_replay),
                    "static_end_to_end_seconds": one_time_static_seconds,
                    "hybrid_response_seconds": train_replay.metrics.end_to_end_seconds,
                    "cache_update_seconds": train_replay.diagnostics.cache_update_seconds,
                    "hybrid_inclusive_seconds": one_time_hybrid_seconds,
                    "incremental_overhead_seconds": one_time_overhead,
                },
                "future_pairs": pairs,
                "summary": {
                    "paired_tpf_ratio": _intervals(
                        tpf_ratios,
                        samples=args.bootstrap_samples,
                        seed=args.seed + 10 * prompt_index,
                    ),
                    "paired_decode_tps_ratio": _intervals(
                        decode_tps_ratios,
                        samples=args.bootstrap_samples,
                        seed=args.seed + 10 * prompt_index + 2,
                    ),
                    "paired_end_to_end_tps_ratio": _intervals(
                        e2e_tps_ratios,
                        samples=args.bootstrap_samples,
                        seed=args.seed + 10 * prompt_index + 4,
                    ),
                    "future_request_saving_seconds": _intervals(
                        request_savings,
                        samples=args.bootstrap_samples,
                        seed=args.seed + 10 * prompt_index + 6,
                    ),
                    "mean_future_request_saving_seconds": mean_future_saving,
                    "break_even_future_requests": break_even,
                    "cumulative_speedup_including_cache_build": (
                        cumulative_static / cumulative_replay
                        if cumulative_replay > 0
                        else 0.0
                    ),
                },
            }
        )
        all_tpf_ratios.extend(tpf_ratios)
        all_decode_tps_ratios.extend(decode_tps_ratios)
        all_e2e_tps_ratios.extend(e2e_tps_ratios)
        all_request_savings.extend(request_savings)

    result = {
        "schema_version": 1,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "execution_backend": "huggingface_pytorch_kv_cache_verifier_replay",
        "claim_scope": {
            "stage": "engineering pilot; not preregistered confirmatory evidence",
            "algorithmic_fidelity": (
                "real checkpoint, real KV rollback, exact greedy/filtered replay, "
                "two-forward Uno fallback"
            ),
            "performance_fidelity": (
                "Windows Hugging Face fallback only; not official Nano-vLLM, "
                "FlashAttention, Triton, paged KV, or CUDA graphs"
            ),
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
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_compute_capability": (
                list(torch.cuda.get_device_capability(device))
                if device.type == "cuda"
                else None
            ),
        },
        "sampling": {
            **asdict(sampling),
            "mask_token_id": args.mask_token_id,
            "stop_token_ids": args.stop_token_ids,
            "ignore_stop": args.ignore_stop,
        },
        "design": {
            "prompts": prompts,
            "block_size": args.block_size,
            "max_new_tokens": args.max_new_tokens,
            "warmup_tokens": args.warmup_tokens,
            "repetitions": args.repetitions,
            "cache_build_requests_per_prompt": 1,
            "future_cache_frozen": True,
            "min_suffix_length": args.min_suffix_length,
            "max_suffix_length": args.max_suffix_length,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "paired_method_order": "alternating",
        },
        "routing_probe": routing_probe,
        "integrity": {
            "all_paired_output_token_ids_equal": all_output_equal,
            "greedy_ar_reference_required": sampling.temperature <= 0,
            "routing_seed_rows_unchanged": routing_probe["clean_rows_match"],
            "routing_noise_rows_changed": routing_probe["noise_rows_changed"],
        },
        "aggregate": {
            "paired_tpf_ratio": _intervals(
                all_tpf_ratios,
                samples=args.bootstrap_samples,
                seed=args.seed + 101,
            ),
            "paired_decode_tps_ratio": _intervals(
                all_decode_tps_ratios,
                samples=args.bootstrap_samples,
                seed=args.seed + 103,
            ),
            "paired_end_to_end_tps_ratio": _intervals(
                all_e2e_tps_ratios,
                samples=args.bootstrap_samples,
                seed=args.seed + 105,
            ),
            "future_request_saving_seconds": _intervals(
                all_request_savings,
                samples=args.bootstrap_samples,
                seed=args.seed + 107,
            ),
        },
        "prompt_results": prompt_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
