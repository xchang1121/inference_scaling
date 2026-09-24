"""Run the GSM8K comparison with resumable per-example records.

Method assembly lives in :mod:`experiments.arllm.assembly.method_runners`; this entry
point owns the command line, manifest, resumable records and summary.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import statistics
import sys
import tomllib
from pathlib import Path
from typing import Any, Sequence

import torch
import transformers

from experiments.arllm.assembly.common import (
    fraction_text,
    installed_package_version,
    load_backend,
    prompt_tokens,
    sample_one,
    timed,
)
from experiments.arllm.assembly.method_runners import run_method
from experiments.arllm.assembly.reward_sources import REWARD_SOURCES
from experiments.arllm.assembly.runtime import model_metadata, set_rl_adapter_override, validate_model_artifacts
from experiments.shared.artifacts import (
    dataclass_snapshot_delta,
    implementation_hashes,
    json_fingerprint,
    load_jsonl,
)
from experiments.shared.methods import AR_ARCHIVED_METHODS, AR_METHODS
from experiments.shared.model_cli import add_model_output_arguments, apply_model_output_overrides
from experiments.shared.statistics import wilson_interval
from inference_scaling.arllm.backends import (
    BACKEND_CHOICES,
    close_backend,
    configured_backend,
    set_backend_override,
)
from inference_scaling.shared.evaluation import extract_numeric_answer, load_gsm8k, select_problems
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.rewards.verifier import replace_verifier_from_file, verifier_spec_from_config

# Archived methods stay runnable by name to reproduce reported results.
METHODS = AR_METHODS + tuple(name for name in AR_ARCHIVED_METHODS if name not in AR_METHODS)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
# Files under src/inference_scaling and experiments/shared are hashed automatically.
IMPLEMENTATION_FILES = (
    "experiments/arllm/gsm8k_reproduction.py",
    "experiments/arllm/assembly/common.py",
    "experiments/arllm/assembly/method_runners.py",
    "experiments/arllm/assembly/reward_sources.py",
)


def _summary(
    records: Sequence[dict[str, Any]], manifest: dict[str, Any]
) -> dict[str, Any]:
    count = len(records)
    correct = sum(bool(record["correct"]) for record in records)
    low, high = wilson_interval(correct, count)
    elapsed = [float(record["elapsed_seconds"]) for record in records]
    output_lengths = [int(record["output_tokens"]) for record in records]
    generated = sum(
        int(record["backend_delta"].get("generated_tokens", 0)) for record in records
    )
    scored = sum(
        int(record["backend_delta"].get("scored_tokens", 0)) for record in records
    )
    proposal_generated = sum(
        int(record.get("proposal_backend_delta", {}).get("generated_tokens", 0))
        for record in records
    )
    base_generation_slots = sum(
        int(record["backend_delta"].get("generation_forward_token_slots", 0))
        for record in records
    )
    base_shared_prefill_saved = sum(
        int(record["backend_delta"].get("shared_prefill_tokens_saved", 0))
        for record in records
    )
    base_score_slots = sum(
        int(record["backend_delta"].get("score_forward_token_slots", 0))
        for record in records
    )
    base_flops = sum(
        int(record["backend_delta"].get("estimated_dense_forward_flops", 0))
        for record in records
    )
    proposal_generation_slots = sum(
        int(
            record.get("proposal_backend_delta", {}).get(
                "generation_forward_token_slots", 0
            )
        )
        for record in records
    )
    proposal_shared_prefill_saved = sum(
        int(
            record.get("proposal_backend_delta", {}).get(
                "shared_prefill_tokens_saved", 0
            )
        )
        for record in records
    )
    proposal_score_slots = sum(
        int(
            record.get("proposal_backend_delta", {}).get("score_forward_token_slots", 0)
        )
        for record in records
    )
    proposal_flops = sum(
        int(
            record.get("proposal_backend_delta", {}).get(
                "estimated_dense_forward_flops", 0
            )
        )
        for record in records
    )
    direct_slots = sum(
        int(
            record.get("diagnostics", {}).get(
                "direct_generation_forward_token_slots", 0
            )
        )
        for record in records
    )
    direct_flops = sum(
        int(
            record.get("diagnostics", {}).get("direct_estimated_dense_forward_flops", 0)
        )
        for record in records
    )
    total_forward_slots = (
        base_generation_slots
        + base_score_slots
        + proposal_generation_slots
        + proposal_score_slots
        + direct_slots
    )
    total_flops = base_flops + proposal_flops + direct_flops
    return {
        "schema_version": 3,
        "manifest_fingerprint": manifest["fingerprint"],
        "method": manifest["method"],
        "tag": manifest["tag"],
        "examples": count,
        "correct": correct,
        "accuracy": correct / count,
        "accuracy_wilson_95": [low, high],
        "sum_example_seconds": sum(elapsed),
        "mean_example_seconds": statistics.fmean(elapsed),
        "median_example_seconds": statistics.median(elapsed),
        "mean_selected_output_tokens": statistics.fmean(output_lengths),
        "median_selected_output_tokens": statistics.median(output_lengths),
        "maximum_selected_output_tokens": max(output_lengths),
        "base_generated_tokens": generated,
        "proposal_generated_tokens": proposal_generated,
        "base_scored_tokens": scored,
        "base_generation_forward_token_slots": base_generation_slots,
        "base_shared_prefill_tokens_saved": base_shared_prefill_saved,
        "base_score_forward_token_slots": base_score_slots,
        "proposal_generation_forward_token_slots": proposal_generation_slots,
        "proposal_shared_prefill_tokens_saved": proposal_shared_prefill_saved,
        "proposal_score_forward_token_slots": proposal_score_slots,
        "direct_generation_forward_token_slots": direct_slots,
        "total_forward_token_slots": total_forward_slots,
        "total_shared_prefill_tokens_saved": (
            base_shared_prefill_saved + proposal_shared_prefill_saved
        ),
        "estimated_dense_forward_flops": total_flops,
        "estimated_dense_forward_petaflops": total_flops / 1e15,
        "compute_definition": (
            "forward token slots count every non-cached model-input position actually "
            "submitted by the benchmark, including repeated prompt/scoring positions; "
            "dominant dense FLOPs = 2 * model parameter count * forward token slots, "
            "summed separately for the 1.5B and 0.5B models"
        ),
        "compute_exclusions": (
            "quadratic attention, elementwise kernels, tokenization, sampling, and host work"
        ),
    }


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    apply_model_output_overrides(config, args)
    set_backend_override(config, args.backend)
    if getattr(args, "vllm_mh_fused_logprobs", False):
        if args.method != "mh":
            raise ValueError("--vllm-mh-fused-logprobs requires --method mh")
        config.setdefault("vllm", {}).setdefault("base", {})[
            "mh_fused_logprobs"
        ] = True
    set_rl_adapter_override(config, getattr(args, "rl_adapter", None))
    replace_verifier_from_file(config, getattr(args, "verifier_config", None))
    if args.limit is not None:
        config["run"]["sample_count"] = args.limit
    if args.sampling_temperature is not None:
        config.setdefault("sampling", {})["temperature"] = args.sampling_temperature
    if args.num_beams is not None:
        config["beam"]["num_beams"] = args.num_beams
    if args.best_of_n_samples is not None:
        config["best_of_n"]["samples"] = args.best_of_n_samples
    if args.conditional_reward is not None:
        config["conditional_is"]["reward"] = args.conditional_reward
        config.setdefault("iterated_is", {})["reward"] = args.conditional_reward
        config.setdefault("reward", {})["source"] = args.conditional_reward
    if args.reward_temperature is not None:
        config["conditional_is"]["reward_temperature"] = args.reward_temperature
        config.setdefault("reward", {})["temperature"] = args.reward_temperature
    if getattr(args, "logprob_reward_scale", None) is not None:
        config["conditional_is"]["logprob_reward_scale"] = args.logprob_reward_scale
        config.setdefault("reward", {})["logprob_scale"] = args.logprob_reward_scale
    for argument, setting in (
        ("consilience_top_k", "consilience_top_k"),
        ("consilience_window_fraction", "consilience_window_fraction"),
        ("consilience_window_tokens", "consilience_window_tokens"),
        ("consilience_skip_fraction", "consilience_skip_fraction"),
        ("consilience_initial_penalty", "consilience_initial_penalty"),
        ("consilience_reward_scale", "consilience_reward_scale"),
        ("consilience_reasoning_end_text", "consilience_reasoning_end_text"),
    ):
        value = getattr(args, argument, None)
        if value is not None:
            config["conditional_is"][setting] = value
            if setting == "consilience_reasoning_end_text":
                config.setdefault("output", {})["thinking_end_text"] = value
            else:
                name = "scale" if setting == "consilience_reward_scale" else setting.removeprefix("consilience_")
                config.setdefault("reward", {}).setdefault("consilience", {})[name] = value
    if args.importance_log_ratio_clip is not None:
        value = args.importance_log_ratio_clip.strip().lower()
        parsed_clip = None if value == "none" else float(value)
        if parsed_clip is not None and parsed_clip <= 0:
            raise ValueError("importance log-ratio clip must be positive or 'none'")
        config["conditional_is"]["importance_log_ratio_clip"] = parsed_clip
    if args.disable_importance_correction:
        if not args.method.endswith("small_proposal"):
            raise ValueError(
                "--disable-importance-correction requires a small-proposal method"
            )
        config["conditional_is"]["apply_importance_correction"] = False
        config["conditional_is"]["importance_log_ratio_clip"] = None
    if args.mh_alpha is not None:
        config["mh"]["alpha"] = args.mh_alpha
    if args.mh_steps is not None:
        config["mh"]["steps_per_block"] = args.mh_steps
    if getattr(args, "mh_suffix_schedule", None) is not None:
        config["mh"]["suffix_schedule"] = args.mh_suffix_schedule
    if args.candidate_count is not None:
        config["conditional_is"]["candidate_count"] = args.candidate_count
    if args.rollout_count is not None:
        config["conditional_is"]["rollout_count"] = args.rollout_count
    if getattr(args, "iterated_pool_size", None) is not None:
        config.setdefault("iterated_is", {})["pool_size"] = args.iterated_pool_size
    if getattr(args, "iterated_updates", None) is not None:
        config.setdefault("iterated_is", {})["updates"] = args.iterated_updates
    if getattr(args, "consensus_pilot_samples", None) is not None:
        config.setdefault("iterated_is", {})["pilot_samples"] = (
            args.consensus_pilot_samples
        )
    if args.block_size is not None:
        if args.method in {"mh", "verifier_mh", "reward_mh"}:
            config["mh"]["block_size"] = args.block_size
        else:
            config["conditional_is"]["block_size"] = args.block_size


def _model_metadata(config: dict[str, Any], method: str) -> dict[str, Any]:
    return model_metadata(config, "rl" if method.startswith("rl_") else "base")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/arllm.toml"))
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        help="override runtime.backend before the experiment fingerprint is computed",
    )
    parser.add_argument(
        "--vllm-mh-fused-logprobs",
        action="store_true",
        help=(
            "return proposal and base token probabilities from one vLLM decode; "
            "requires --backend vllm-sync and vLLM 0.26.x"
        ),
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--sampling-scope", choices=("full", "thinking"))
    parser.add_argument("--thinking-mode", choices=("auto", "enabled", "disabled"))
    parser.add_argument("--thinking-format", choices=("auto", "tags", "json", "xml"))
    parser.add_argument("--thinking-path", help="structured thinking field, e.g. response.reasoning")
    parser.add_argument("--content-path", help="structured final-content field, e.g. response.answer")
    parser.add_argument("--thinking-start-text")
    parser.add_argument("--thinking-end-text")
    parser.add_argument("--starts-in-thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--tag", default="default")
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("results/gsm8k"))
    parser.add_argument("--rl-adapter", type=Path)
    parser.add_argument(
        "--verifier-config",
        type=Path,
        help="standalone TOML file whose [verifier] table replaces the default",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--sampling-temperature", type=float)
    parser.add_argument("--num-beams", type=int)
    parser.add_argument("--best-of-n-samples", type=int)
    parser.add_argument(
        "--conditional-reward", "--reward",
        choices=REWARD_SOURCES,
        help=(
            "reward used by Best-of-N and conditional methods; verifier uses "
            "the separately configured [verifier] component"
        ),
    )
    parser.add_argument("--reward-temperature", type=float)
    parser.add_argument(
        "--logprob-reward-scale",
        type=float,
        help=(
            "c in r=c*log p(y|x); combined with reward temperature tau this "
            "targets the power 1+c/tau"
        ),
    )
    parser.add_argument("--consilience-top-k", type=int)
    parser.add_argument("--consilience-window-fraction", type=float)
    parser.add_argument("--consilience-window-tokens", type=int)
    parser.add_argument("--consilience-skip-fraction", type=float)
    parser.add_argument("--consilience-initial-penalty", type=float)
    parser.add_argument("--consilience-reward-scale", type=float)
    parser.add_argument(
        "--consilience-reasoning-end-text",
        help=(
            "optional model-specific delimiter after the reasoning phase; "
            "the delimiter and final-answer tokens are excluded from the reward"
        ),
    )
    parser.add_argument(
        "--importance-log-ratio-clip",
        help="positive symmetric clip or 'none' for exact untruncated weights",
    )
    parser.add_argument(
        "--disable-importance-correction",
        action="store_true",
        help=(
            "skip base-model rescoring of small-model rollouts and use proposal-model "
            "continuation reward weighting as a biased lookahead signal"
        ),
    )
    parser.add_argument("--mh-alpha", type=float)
    parser.add_argument("--mh-steps", type=int)
    parser.add_argument(
        "--mh-suffix-schedule",
        choices=("uniform", "inverse_length", "multiscale"),
    )
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--rollout-count", type=int)
    parser.add_argument("--iterated-pool-size", type=int)
    parser.add_argument("--iterated-updates", type=int)
    parser.add_argument("--consensus-pilot-samples", type=int)
    parser.add_argument("--block-size", type=int)
    parser.add_argument(
        "--draw-index",
        type=int,
        default=0,
        help="independent sampling replicate included in the request-level seed",
    )
    add_model_output_arguments(parser)
    args = parser.parse_args()

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    _apply_overrides(config, args)
    if (
        str(config["runtime"]["device"]).startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    all_problems = load_gsm8k(args.data)
    problems = select_problems(
        all_problems,
        int(config["run"]["sample_count"]),
        seed=int(config["run"]["subset_seed"]),
    )
    roles = {"base"}
    if args.method.startswith("rl_"):
        roles.add("rl")
    if args.method.endswith("small_proposal"):
        roles.add("proposal")
    input_artifacts = validate_model_artifacts(config, roles)
    input_weight_hashes = input_artifacts["weight_sha256"]
    actual_base_hash = input_weight_hashes["base"]
    actual_adapter_hash = input_weight_hashes.get("rl_adapter")
    actual_proposal_hash = input_weight_hashes.get("proposal")
    effective = {
        "config": config,
        "method": args.method,
        "tag": args.tag,
        "draw_index": args.draw_index,
        "input_weight_sha256": input_weight_hashes,
        "input_metadata_sha256": input_artifacts["metadata_sha256"],
        "input_adapter_sha256": input_artifacts["adapter_sha256"],
        "implementation_sha256": implementation_hashes(
            REPOSITORY_ROOT,
            entrypoints=IMPLEMENTATION_FILES,
        ),
        "problem_indices": [problem.index for problem in problems],
    }
    fingerprint = json_fingerprint(effective)
    run_dir = (
        args.output_root / str(config["run"]["name"]) / f"{args.method}-{args.tag}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "records.jsonl"
    summary_path = run_dir / "summary.json"
    manifest = {
        "schema_version": 2,
        "fingerprint": fingerprint,
        "method": args.method,
        "tag": args.tag,
        "effective": effective,
        "dataset": {
            "name": "GSM8K official test split",
            "path": str(args.data),
            "rows_in_public_split": len(all_problems),
        },
        "model": _model_metadata(config, args.method),
        "configured_verifier": (
            verifier_spec_from_config(config).as_dict()
            if "verifier" in config
            else None
        ),
        "proposal_model": (
            model_metadata(config, "proposal")
            if args.method.endswith("small_proposal")
            else None
        ),
        "environment": {
            "backend": configured_backend(config),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "vllm": (
                installed_package_version("vllm")
                if configured_backend(config).startswith("vllm")
                else None
            ),
        },
    }
    previous = manifest
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["fingerprint"] != fingerprint:
            raise ValueError(
                f"{run_dir} contains a different experiment; choose a new --tag"
            )
    elif records_path.is_file():
        raise ValueError(f"{run_dir} contains records without a manifest")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    existing_records = load_jsonl(records_path)
    completed = {int(record["problem_index"]) for record in existing_records}
    pending = [problem for problem in problems if problem.index not in completed]
    if not pending:
        manifest = previous
        selected = [
            record
            for record in existing_records
            if int(record["problem_index"]) in {problem.index for problem in problems}
        ]
        summary = _summary(selected, manifest)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return

    model_key = "rl" if args.method.startswith("rl_") else "base"
    adapter_base = None
    if model_key == "rl" and config["models"].get("rl_kind") == "peft_adapter":
        adapter_base = str(config["models"]["rl_base"])
    backend = None
    proposal_backend = None
    try:
        backend = load_backend(
            str(config["models"][model_key]),
            config,
            adapter_base=adapter_base, role=model_key,
        )
        proposal_backend = None
        if args.method.endswith("small_proposal"):
            proposal_backend = load_backend(str(config["models"]["proposal"]), config, role="proposal")
            if backend.tokenizer.get_vocab() != proposal_backend.tokenizer.get_vocab():
                raise ValueError(
                    "base and proposal tokenizers do not have identical vocabularies"
                )

        manifest["model"]["parameter_count"] = backend.parameter_count
        manifest["model"]["verified_base_weight_sha256"] = actual_base_hash
        if model_key == "rl" and config["models"].get("rl_kind") == "peft_adapter":
            assert actual_adapter_hash is not None
            manifest["model"]["adapter_weight_sha256"] = actual_adapter_hash
            manifest["model"]["base_weight_sha256"] = str(
                input_weight_hashes.get("rl_base", actual_base_hash)
            )
        if proposal_backend is not None and manifest["proposal_model"] is not None:
            assert actual_proposal_hash is not None
            manifest["proposal_model"]["parameter_count"] = proposal_backend.parameter_count
            manifest["proposal_model"]["verified_weight_sha256"] = actual_proposal_hash
        manifest["compute_accounting"] = {
            "primary_units": ["forward_token_slots", "estimated_dense_forward_flops"],
            "flop_formula": "2 * model_parameter_count * forward_token_slots",
            "wall_time_role": "hardware-dependent supplemental measurement",
        }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        warm_prompt = prompt_tokens(backend, pending[0], config)
        sample_one(
            backend,
            warm_prompt,
            max_new_tokens=2,
            temperature=1.0,
            seed=int(config["run"]["seed"]),
            request_id="warmup",
        )
        seeds = SeedStream(
            SeedStream(int(config["run"]["seed"])).derive("draw", args.draw_index)
        )
        with records_path.open("a", encoding="utf-8", buffering=1) as sink:
            for ordinal, problem in enumerate(pending, 1):
                prompt = prompt_tokens(backend, problem, config)
                before = backend.snapshot()
                proposal_before = proposal_backend.snapshot() if proposal_backend else None
                (tokens, diagnostics), elapsed = timed(
                    lambda backend=backend, problem=problem, prompt=prompt, proposal_backend=proposal_backend: (
                        run_method(
                            args.method,
                            backend,
                            problem,
                            prompt,
                            config,
                            seeds,
                            proposal_backend,
                        )
                    )
                )
                after = backend.snapshot()
                proposal_after = proposal_backend.snapshot() if proposal_backend else None
                text = backend.decode(tokens)
                segments = diagnostics["output_segments"]
                prediction = extract_numeric_answer(segments["content_text"])
                record = {
                    "schema_version": 2,
                    "method": args.method,
                    "tag": args.tag,
                    "draw_index": args.draw_index,
                    "problem_index": problem.index,
                    "question_sha256": hashlib.sha256(
                        problem.question.encode()
                    ).hexdigest(),
                    "gold_answer": fraction_text(problem.gold_answer),
                    "prediction": fraction_text(prediction),
                    "correct": prediction == problem.gold_answer,
                    "output": text,
                    "thinking": segments["thinking_text"],
                    "content": segments["content_text"],
                    "output_tokens": len(tokens),
                    "prompt_tokens": len(prompt),
                    "elapsed_seconds": elapsed,
                    "backend_delta": dataclass_snapshot_delta(before, after),
                    "diagnostics": diagnostics,
                }
                if proposal_before is not None and proposal_after is not None:
                    record["proposal_backend_delta"] = dataclass_snapshot_delta(
                        proposal_before, proposal_after
                    )
                sink.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                fallback = segments.get("sampling_fallback_reason") or segments.get("reward_fallback_reason")
                if fallback is not None:
                    print(
                        f"gsm8k_index={problem.index} sampling_scope={segments['sampling_scope']} "
                        f"reward_scope={segments.get('reward_scope', 'unchanged')} full_fallback={fallback}",
                        flush=True,
                    )
                print(
                    f"[{ordinal}/{len(pending)}] method={args.method} "
                    f"gsm8k_index={problem.index} correct={record['correct']} "
                    f"seconds={elapsed:.3f}",
                    flush=True,
                )

        records = load_jsonl(records_path)
        selected = [
            record
            for record in records
            if int(record["problem_index"]) in {p.index for p in problems}
        ]
        summary = _summary(selected, manifest)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    finally:
        try:
            close_backend(proposal_backend)
        finally:
            close_backend(backend)
            backend = proposal_backend = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
