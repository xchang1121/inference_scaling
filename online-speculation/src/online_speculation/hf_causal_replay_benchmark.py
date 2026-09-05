"""First-request benchmark for causal verified-past replay on Uno-1B."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .hf_replay_benchmark import _intervals, _new_runner, _run_result_dict
from .hf_replay_uno import ReplayRuntimeConfig
from .hf_uno import (
    ADAPTER_REVISION,
    ADAPTER_WEIGHT_SHA256,
    BASE_REVISION,
    BASE_WEIGHT_SHA256,
    _dtype,
    _package_version,
    _parse_ints,
    _sha256,
    load_runtime,
)
from .torch_sampling import SamplingConfig


DEFAULT_WORKLOADS = (
    (
        "natural_answer",
        "Explain why speculative decoding can be lossless. Give a concise derivation "
        "of the acceptance and residual-correction rule, then state its assumptions.",
    ),
    (
        "explicit_repetition",
        "Write the following sentence exactly eight times, one sentence per line: "
        "Verification makes speculative decoding lossless.",
    ),
    (
        "code_template",
        "Write six Python functions named step_1 through step_6. Each function must "
        "use the same three-line body: assign value = input_value + 1, print value, "
        "and return value. Do not abbreviate any function.",
    ),
)


def _parse_workload(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("workload must have NAME=PROMPT form")
    name, prompt = value.split("=", 1)
    if not name.strip() or not prompt.strip():
        raise argparse.ArgumentTypeError("workload name and prompt cannot be empty")
    return name.strip(), prompt.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", type=_parse_workload, action="append")
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--warmup-tokens", type=int, default=24)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--mask-token-id", type=int, default=64256)
    parser.add_argument("--stop-token-ids", type=_parse_ints, default=[64019, 1])
    parser.add_argument(
        "--ignore-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--min-suffix-length", type=int, default=8)
    parser.add_argument("--max-suffix-length", type=int, default=32)
    parser.add_argument("--match-length-bucket-width", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260915)
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
        raise ValueError("causal replay requires block_size >= 2")
    if args.max_new_tokens < 2 or args.warmup_tokens < 2:
        raise ValueError("token budgets must be at least two")
    if args.repetitions < 1:
        raise ValueError("repetitions must be positive")
    if args.bootstrap_samples < 1_000:
        raise ValueError("use at least 1,000 bootstrap resamples")
    if args.min_suffix_length < 1 or args.max_suffix_length < args.min_suffix_length:
        raise ValueError("suffix bounds are invalid")
    if args.match_length_bucket_width < 1:
        raise ValueError("match-length bucket width must be positive")

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
            raise RuntimeError("base checkpoint SHA-256 does not match the lock")
        if hashes["adapter_weight_sha256"] != ADAPTER_WEIGHT_SHA256:
            raise RuntimeError("Uno adapter SHA-256 does not match the lock")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    sampling = SamplingConfig(temperature=0.0, top_k=1, top_p=1.0)
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
    workloads = tuple(args.workload) if args.workload else DEFAULT_WORKLOADS
    encoded = [(name, prompt, runtime.encode_prompt(prompt)) for name, prompt in workloads]
    routing_probe = runtime.routing_probe(
        encoded[0][2],
        block_size=min(args.block_size, 8),
        seed=args.seed - 2,
    )

    # Warm each tensor shape and both static/replay model branches with disposable state.
    runtime.generate_ar(
        encoded[0][2],
        max_new_tokens=args.warmup_tokens,
        seed=args.seed - 1,
    )
    runtime.generate_uno(
        encoded[0][2],
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
        match_length_bucket_width=args.match_length_bucket_width,
    )
    warm_runner.generate(
        encoded[0][2],
        max_new_tokens=args.warmup_tokens,
        seed=args.seed - 1,
        config=ReplayRuntimeConfig(block_size=args.block_size),
    )
    warm_runner.generate(
        encoded[0][2],
        max_new_tokens=args.warmup_tokens,
        seed=args.seed - 1,
        config=ReplayRuntimeConfig(
            block_size=args.block_size,
            observe_after_request=False,
        ),
    )

    workload_results: list[dict[str, Any]] = []
    aggregate_tpf: list[float] = []
    aggregate_decode: list[float] = []
    aggregate_inclusive: list[float] = []
    all_causal_static_equal = True
    all_static_ar_equal = True

    for workload_index, (name, prompt, input_ids) in enumerate(encoded):
        reference_seed = args.seed + 100_000 * workload_index
        ar = runtime.generate_ar(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            seed=reference_seed,
        )
        pairs: list[dict[str, Any]] = []
        tpf_ratios: list[float] = []
        decode_ratios: list[float] = []
        inclusive_ratios: list[float] = []
        replay_cycles: list[int] = []
        static_fallback_cycles: list[int] = []
        for repetition in range(args.repetitions):
            run_seed = reference_seed + 1_000 + repetition
            runner = _new_runner(
                runtime,
                sampling=sampling,
                block_size=args.block_size,
                min_suffix_length=args.min_suffix_length,
                max_suffix_length=args.max_suffix_length,
                match_length_bucket_width=args.match_length_bucket_width,
            )
            if repetition % 2 == 0:
                static = runtime.generate_uno(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    block_size=args.block_size,
                    seed=run_seed,
                )
                causal = runner.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    seed=run_seed,
                    config=ReplayRuntimeConfig(
                        block_size=args.block_size,
                        observe_after_request=False,
                        causal_within_request=True,
                    ),
                )
                method_order = ["static", "causal"]
            else:
                causal = runner.generate(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    seed=run_seed,
                    config=ReplayRuntimeConfig(
                        block_size=args.block_size,
                        observe_after_request=False,
                        causal_within_request=True,
                    ),
                )
                static = runtime.generate_uno(
                    input_ids,
                    max_new_tokens=args.max_new_tokens,
                    block_size=args.block_size,
                    seed=run_seed,
                )
                method_order = ["causal", "static"]

            causal_static_equal = (
                causal.metrics.output_token_ids == static.output_token_ids
            )
            static_ar_equal = static.output_token_ids == ar.output_token_ids
            all_causal_static_equal &= causal_static_equal
            all_static_ar_equal &= static_ar_equal
            inclusive_seconds = (
                causal.metrics.end_to_end_seconds
                + causal.diagnostics.cache_close_seconds
            )
            tpf_ratio = (
                causal.metrics.decoder_tokens_per_forward
                / static.decoder_tokens_per_forward
            )
            decode_ratio = (
                causal.metrics.decode_tokens_per_second
                / static.decode_tokens_per_second
            )
            inclusive_ratio = static.end_to_end_seconds / inclusive_seconds
            tpf_ratios.append(tpf_ratio)
            decode_ratios.append(decode_ratio)
            inclusive_ratios.append(inclusive_ratio)
            replay_cycles.append(causal.diagnostics.replay_cycles)
            static_fallback_cycles.append(causal.diagnostics.static_cycles)
            pairs.append(
                {
                    "repetition": repetition,
                    "seed": run_seed,
                    "method_order": method_order,
                    "causal_static_token_ids_equal": causal_static_equal,
                    "static_ar_token_ids_equal": static_ar_equal,
                    "tpf_ratio": tpf_ratio,
                    "decode_tps_ratio": decode_ratio,
                    "inclusive_end_to_end_tps_ratio": inclusive_ratio,
                    "static": asdict(static),
                    "causal": _run_result_dict(causal),
                }
            )
            print(
                f"{name} rep={repetition} TPF={tpf_ratio:.3f}x "
                f"decode={decode_ratio:.3f}x replay={causal.diagnostics.replay_cycles} "
                f"fallback={causal.diagnostics.static_cycles}",
                flush=True,
            )

        workload_results.append(
            {
                "workload_index": workload_index,
                "name": name,
                "prompt": prompt,
                "prompt_tokens": int(input_ids.size(1)),
                "ar_reference": asdict(ar),
                "pairs": pairs,
                "summary": {
                    "paired_tpf_ratio": _intervals(
                        tpf_ratios,
                        samples=args.bootstrap_samples,
                        seed=args.seed + 10 * workload_index,
                    ),
                    "paired_decode_tps_ratio": _intervals(
                        decode_ratios,
                        samples=args.bootstrap_samples,
                        seed=args.seed + 10 * workload_index + 2,
                    ),
                    "paired_inclusive_end_to_end_tps_ratio": _intervals(
                        inclusive_ratios,
                        samples=args.bootstrap_samples,
                        seed=args.seed + 10 * workload_index + 4,
                    ),
                    "mean_replay_cycles": sum(replay_cycles) / len(replay_cycles),
                    "mean_static_fallback_cycles": (
                        sum(static_fallback_cycles) / len(static_fallback_cycles)
                    ),
                },
            }
        )
        aggregate_tpf.extend(tpf_ratios)
        aggregate_decode.extend(decode_ratios)
        aggregate_inclusive.extend(inclusive_ratios)

    result = {
        "schema_version": 1,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "execution_backend": "huggingface_pytorch_kv_cache_causal_replay",
        "claim_scope": {
            "stage": "engineering pilot; first-request repetition workloads",
            "algorithmic_fidelity": (
                "request-local verified-past overlay, exact greedy replay, real KV "
                "rollback, two-forward Uno fallback"
            ),
            "performance_fidelity": (
                "Windows Hugging Face fallback only; not official Nano-vLLM"
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
        },
        "sampling": {
            **asdict(sampling),
            "mask_token_id": args.mask_token_id,
            "stop_token_ids": args.stop_token_ids,
            "ignore_stop": args.ignore_stop,
        },
        "design": {
            "workloads": [
                {"name": name, "prompt": prompt} for name, prompt in workloads
            ],
            "block_size": args.block_size,
            "max_new_tokens": args.max_new_tokens,
            "warmup_tokens": args.warmup_tokens,
            "repetitions": args.repetitions,
            "fresh_empty_global_cache_per_pair": True,
            "request_local_overlay_discarded_after_pair": True,
            "min_suffix_length": args.min_suffix_length,
            "max_suffix_length": args.max_suffix_length,
            "match_length_bucket_width": args.match_length_bucket_width,
            "bootstrap_samples": args.bootstrap_samples,
            "seed": args.seed,
            "paired_method_order": "alternating",
        },
        "routing_probe": routing_probe,
        "integrity": {
            "all_causal_static_token_ids_equal": all_causal_static_equal,
            "all_static_ar_token_ids_equal": all_static_ar_equal,
            "finite_precision_target_note": (
                "Static Uno block verification can differ from one-token AR when "
                "BF16 kernel shape rounding flips a near-tied greedy argmax; the "
                "causal exactness gate is paired equality to the unchanged static "
                "Uno verifier path."
            ),
            "routing_seed_rows_unchanged": routing_probe["clean_rows_match"],
            "routing_noise_rows_changed": routing_probe["noise_rows_changed"],
        },
        "aggregate": {
            "paired_tpf_ratio": _intervals(
                aggregate_tpf,
                samples=args.bootstrap_samples,
                seed=args.seed + 101,
            ),
            "paired_decode_tps_ratio": _intervals(
                aggregate_decode,
                samples=args.bootstrap_samples,
                seed=args.seed + 103,
            ),
            "paired_inclusive_end_to_end_tps_ratio": _intervals(
                aggregate_inclusive,
                samples=args.bootstrap_samples,
                seed=args.seed + 105,
            ),
        },
        "workload_results": workload_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
