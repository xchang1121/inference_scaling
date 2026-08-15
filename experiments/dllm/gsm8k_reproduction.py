"""Run the paired LLaDA GSM8K quality methods with resumable records."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import time
import tomllib
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Sequence

from experiments.shared.paired_protocol import load_pairing
from inference_scaling.dllm.algorithms import (
    run_conditional_diffusion_is,
    run_diffusion_block_beam,
    run_diffusion_reward_mh,
    run_diffusion_trajectory_power_mh,
)
from inference_scaling.dllm.backends import load_llada_backend
from inference_scaling.dllm.config import (
    DiffusionBlockBeamConfig,
    DiffusionISConfig,
    DiffusionMHConfig,
    DiffusionPowerMHConfig,
    DiffusionSamplingConfig,
)
from inference_scaling.dllm.types import DiffusionGenerationRequest
from inference_scaling.shared.evaluation import (
    CumulativeConsensusReward,
    GSM8KProblem,
    consensus_index,
    extract_numeric_answer,
    gsm8k_prompt,
    load_gsm8k,
    modal_answer,
    select_problems,
)
from inference_scaling.shared.metrics import importance_effective_sample_size
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence


METHODS = (
    "base",
    "block_beam",
    "best_of_n",
    "trajectory_power_mh",
    "conditional_is",
    "conditional_is_reduced_layer_proposal",
    "conditional_is_reduced_layer_proposal_unclipped",
    "conditional_is_reduced_layer_proposal_uncorrected",
    "verifier_mh",
    "verifier_conditional_is",
    "verifier_conditional_is_reduced_layer_proposal",
    "vrpo_sample",
    "vrpo_greedy",
)
IMPLEMENTATION_FILES = (
    "experiments/dllm/gsm8k_reproduction.py",
    "src/inference_scaling/dllm/algorithms/is_sampling.py",
    "src/inference_scaling/dllm/algorithms/mh.py",
    "src/inference_scaling/dllm/algorithms/search.py",
    "src/inference_scaling/dllm/backends/llada.py",
    "src/inference_scaling/dllm/backends/loader.py",
    "src/inference_scaling/dllm/config.py",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sampling(section: dict[str, Any]) -> DiffusionSamplingConfig:
    return DiffusionSamplingConfig(
        block_length=int(section["block_length"]),
        steps_per_block=int(section.get("denoising_steps", section.get("steps_per_block"))),
        temperature=float(section.get("temperature", 0.0)),
        top_k=int(section.get("top_k", 0)),
        top_p=float(section.get("top_p", 1.0)),
        cfg_scale=float(section.get("cfg_scale", 0.0)),
        remasking=str(section.get("remasking", "low_confidence")),
        confidence_threshold=float(section.get("confidence_threshold", 0.85)),
        mask_token_id=(
            int(section["mask_token_id"])
            if section.get("mask_token_id") is not None
            else None
        ),
    )


def _capped_generation_length(
    *,
    prompt_length: int,
    maximum: int,
    sampling: DiffusionSamplingConfig,
) -> int:
    """Use at most the AR budget and retain complete diffusion blocks."""

    del prompt_length
    remainder = maximum % sampling.block_length
    length = maximum - remainder
    if length <= 0:
        raise ValueError("generation budget is too small to complete a diffusion block")
    return length


def _sample_one(
    backend: Any,
    prompt: TokenSequence,
    *,
    generation_length: int,
    sampling: DiffusionSamplingConfig,
    seed: int,
    request_id: str,
) -> TokenSequence:
    samples = backend.sample_batch(
        [
            DiffusionGenerationRequest(
                prefix=prompt,
                generation_length=generation_length,
                sampling=sampling,
                seed=seed,
                request_id=request_id,
            )
        ]
    )
    if len(samples) != 1:
        raise RuntimeError("backend returned an invalid number of LLaDA samples")
    return samples[0].token_ids


def _run_best_of_n(
    backend: Any,
    prompt: TokenSequence,
    *,
    generation_length: int,
    sampling: DiffusionSamplingConfig,
    samples: int,
    seeds: SeedStream,
    problem_index: int,
) -> tuple[TokenSequence, dict[str, Any]]:
    requests = [
        DiffusionGenerationRequest(
            prefix=prompt,
            generation_length=generation_length,
            sampling=sampling,
            seed=seeds.derive("dllm-best-of-n", problem_index, draw),
            request_id=f"dllm-best-of-n:{problem_index}:{draw}",
        )
        for draw in range(samples)
    ]
    candidates = backend.sample_batch(requests)
    if len(candidates) != samples:
        raise RuntimeError("backend returned an invalid number of Best-of-N samples")
    texts = [backend.decode(candidate.token_ids) for candidate in candidates]
    answers = [extract_numeric_answer(text) for text in texts]
    tie_break_scores = [
        float(candidate.trajectory_logprob)
        if candidate.trajectory_logprob is not None
        else 0.0
        for candidate in candidates
    ]
    selected_index = consensus_index(texts, tie_break_scores)
    mode = modal_answer(answers)
    return candidates[selected_index].token_ids, {
        "candidate_count": samples,
        "selected_index": selected_index,
        "selector": "modal_numeric_answer_then_trajectory_score_then_draw_order",
        "trajectory_score_available": all(
            candidate.trajectory_logprob is not None for candidate in candidates
        ),
        "modal_answer": str(mode) if mode is not None else None,
    }


def _conditional_diagnostics(result: Any) -> dict[str, Any]:
    rollout_weights: list[list[float]] = []
    rewards: list[float] = []
    raw_corrections: list[float] = []
    applied_corrections: list[float] = []
    for step in result.steps:
        for candidate in step.candidates:
            rollout_weights.append([rollout.log_weight for rollout in candidate.rollouts])
            for rollout in candidate.rollouts:
                rewards.append(float(rollout.reward))
                if rollout.raw_log_importance_ratio is not None:
                    raw_corrections.append(float(rollout.raw_log_importance_ratio))
                    applied_corrections.append(
                        float(rollout.applied_log_importance_ratio)
                    )
    return {
        "decision_stages": len(result.steps),
        "rollout_evaluations": sum(len(values) for values in rollout_weights),
        "mean_rollout_ess": (
            statistics.fmean(
                importance_effective_sample_size(values) for values in rollout_weights
            )
            if rollout_weights
            else 0.0
        ),
        "mean_rollout_reward": statistics.fmean(rewards) if rewards else 0.0,
        "importance_corrected_rollouts": len(raw_corrections),
        "mean_absolute_raw_log_importance_correction": (
            statistics.fmean(abs(value) for value in raw_corrections)
            if raw_corrections
            else 0.0
        ),
        "mean_absolute_applied_log_importance_correction": (
            statistics.fmean(abs(value) for value in applied_corrections)
            if applied_corrections
            else 0.0
        ),
        "clipped_rollout_corrections": sum(
            raw != applied
            for raw, applied in zip(
                raw_corrections, applied_corrections, strict=True
            )
        ),
    }


def run_method(
    method: str,
    backend: Any,
    problem: GSM8KProblem,
    prompt: TokenSequence,
    config: dict[str, Any],
    *,
    seed: int,
    proposal_backend: Any | None = None,
) -> tuple[TokenSequence, dict[str, Any]]:
    if method not in METHODS:
        raise ValueError(f"unknown LLaDA method {method!r}")
    generation_budget = int(config["generation"]["max_new_tokens"])
    generation_sampling = _sampling(config["generation"])
    exact_sampling = _sampling(config["exact_policy"])
    generation_length = _capped_generation_length(
        prompt_length=len(prompt),
        maximum=generation_budget,
        sampling=generation_sampling,
    )
    seeds = SeedStream(seed)

    if method in {"base", "vrpo_sample", "vrpo_greedy"}:
        sampling = generation_sampling
        if method == "vrpo_greedy":
            sampling = replace(sampling, temperature=0.0)
        tokens = _sample_one(
            backend,
            prompt,
            generation_length=generation_length,
            sampling=sampling,
            seed=seeds.derive(method, problem.index),
            request_id=f"{method}:{problem.index}",
        )
        return tokens, {"sampling_policy": sampling.policy_id}

    if method == "block_beam":
        search = config["search"]
        result = run_diffusion_block_beam(
            backend=backend,
            prompt=prompt,
            config=DiffusionBlockBeamConfig(
                total_length=generation_length,
                decision_block_size=int(search["decision_block_size"]),
                width=int(search["width"]),
                branching_factor=int(search["branching_factor"]),
            ),
            sampling=exact_sampling,
            seed=seeds.derive(method, problem.index),
        )
        return result.best.token_ids, {
            "width": int(search["width"]),
            "branching_factor": int(search["branching_factor"]),
            "decision_stages": len(result.stages),
            "sampled_proposals": sum(stage.proposals for stage in result.stages),
            "selected_trajectory_logprob": result.best.trajectory_logprob,
            "search_kind": "sampled_reverse_trajectory_block_beam",
        }

    if method == "best_of_n":
        return _run_best_of_n(
            backend,
            prompt,
            generation_length=generation_length,
            sampling=generation_sampling,
            samples=int(config["best_of_n"]["samples"]),
            seeds=seeds,
            problem_index=problem.index,
        )

    if method == "trajectory_power_mh":
        mh = config["mh"]
        result = run_diffusion_trajectory_power_mh(
            backend=backend,
            prompt=prompt,
            config=DiffusionPowerMHConfig(
                total_length=generation_length,
                decision_block_size=int(mh["decision_block_size"]),
                updates_per_stage=int(mh["updates_per_stage"]),
                alpha=float(mh["alpha"]),
            ),
            sampling=exact_sampling,
            seed=seeds.derive(method, problem.index),
        )
        return result.final.token_ids, {
            "target": "exact_reverse_trajectory_probability_to_alpha",
            "alpha": result.alpha,
            "decision_block_size": int(mh["decision_block_size"]),
            "updates_per_stage": int(mh["updates_per_stage"]),
            "updates": len(result.steps),
            "accepted": sum(step.accepted for step in result.steps),
            "acceptance_rate": result.acceptance_rate,
        }

    def exact_reward(_: TokenSequence, continuation: TokenSequence) -> float:
        prediction = extract_numeric_answer(backend.decode(continuation))
        return float(prediction == problem.gold_answer)

    if method == "verifier_mh":
        mh = config["mh"]
        result = run_diffusion_reward_mh(
            backend=backend,
            prompt=prompt,
            config=DiffusionMHConfig(
                total_length=generation_length,
                updates=int(mh["updates"]),
                reward_temperature=float(mh["reward_temperature"]),
            ),
            sampling=generation_sampling,
            reward=exact_reward,
            seed=seeds.derive(method, problem.index),
        )
        return result.final.token_ids, {
            "target": "base_reverse_process_times_exp_exact_reward_over_temperature",
            "updates": len(result.steps),
            "accepted": sum(step.accepted for step in result.steps),
            "acceptance_rate": result.acceptance_rate,
            "final_reward": result.final_reward,
            "uses_test_gold_oracle": True,
        }

    conditional = config["conditional_is"]
    reduced = "reduced_layer_proposal" in method
    if reduced and proposal_backend is None:
        raise ValueError("reduced-layer conditional IS requires its proposal backend")
    verifier = method.startswith("verifier_")
    uncorrected = method.endswith("_uncorrected")
    unclipped = method.endswith("_unclipped")
    rollout_backend = proposal_backend if reduced else backend
    assert rollout_backend is not None
    apply_correction = reduced and not uncorrected
    clip = (
        None
        if not apply_correction or unclipped
        else float(conditional["importance_log_ratio_clip"])
    )
    reward_batch = None if verifier else CumulativeConsensusReward(backend.decode)
    result = run_conditional_diffusion_is(
        base_backend=backend,
        prompt=prompt,
        config=DiffusionISConfig(
            candidate_count=int(conditional["candidate_count"]),
            rollout_count=int(conditional["rollout_count"]),
            block_size=int(conditional["decision_block_size"]),
            total_length=generation_length,
            reward_temperature=(
                float(config["mh"]["reward_temperature"])
                if verifier
                else float(conditional["reward_temperature"])
            ),
            importance_log_ratio_clip=clip,
        ),
        base_sampling=generation_sampling,
        rollout_backend=rollout_backend,
        rollout_sampling=exact_sampling,
        target_rollout_backend=backend,
        target_rollout_sampling=exact_sampling,
        apply_importance_correction=apply_correction,
        reward=exact_reward if verifier else None,
        reward_batch=reward_batch,
        seed=seeds.derive(method, problem.index),
    )
    diagnostics = _conditional_diagnostics(result)
    diagnostics.update(
        {
            "candidate_source": "full_llada_moe",
            "rollout_source": (
                "shared_prefix_layer_llada" if reduced else "full_llada_moe"
            ),
            "reward_source": "exact_verifier" if verifier else "self_consistency",
            "uses_test_gold_oracle": verifier,
            "apply_importance_correction": apply_correction,
            "importance_log_ratio_clip": clip,
            "target": (
                "base_candidates_with_uncorrected_proposal_rollout_energy"
                if uncorrected
                else "base_candidates_with_target_reverse_trajectory_rollout_energy"
            ),
        }
    )
    return result.token_ids, diagnostics


def _snapshot_delta(before: Any, after: Any) -> dict[str, Any]:
    left = asdict(before)
    right = asdict(after)
    constants = {"total_parameters", "active_parameters"}
    result = {
        key: right[key] if key in constants else right[key] - left[key]
        for key in right
    }
    result["estimated_active_flops"] = (
        2 * result["active_parameters"] * result["model_token_slots"]
    )
    result["estimated_sample_active_flops"] = (
        2 * result["active_parameters"] * result["sample_model_token_slots"]
    )
    result["estimated_score_active_flops"] = (
        2 * result["active_parameters"] * result["score_model_token_slots"]
    )
    return result


def _wilson(correct: int, count: int, z: float = 1.959963984540054) -> list[float]:
    proportion = correct / count
    denominator = 1 + z * z / count
    center = (proportion + z * z / (2 * count)) / denominator
    radius = z * (
        (proportion * (1 - proportion) / count + z * z / (4 * count * count)) ** 0.5
    ) / denominator
    return [center - radius, center + radius]


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot summarize an empty LLaDA experiment")
    correct = sum(bool(record["correct"]) for record in records)

    def total(role: str, field: str) -> float:
        return sum(float(record[role][field]) for record in records)

    main_flops = total("main_compute", "estimated_active_flops")
    proposal_flops = total("proposal_compute", "estimated_active_flops")
    return {
        "method": records[0]["method"],
        "examples": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "accuracy_wilson_95": _wilson(correct, len(records)),
        "wall_clock_seconds": sum(float(record["elapsed_seconds"]) for record in records),
        "main_generation_forward_token_slots": total(
            "main_compute", "sample_model_token_slots"
        ),
        "main_exact_rescoring_forward_token_slots": total(
            "main_compute", "score_model_token_slots"
        ),
        "proposal_generation_forward_token_slots": total(
            "proposal_compute", "sample_model_token_slots"
        ),
        "proposal_exact_rescoring_forward_token_slots": total(
            "proposal_compute", "score_model_token_slots"
        ),
        "main_estimated_active_flops": main_flops,
        "proposal_estimated_active_flops": proposal_flops,
        "total_estimated_active_flops": main_flops + proposal_flops,
        "total_estimated_active_petaflops": (main_flops + proposal_flops) / 1e15,
        "compute_definition": (
            "2 * active model parameters * model-input token slots; generation and "
            "exact trajectory rescoring are recorded separately for the full LLaDA-MoE "
            "model and the shared early-exit proposal"
        ),
    }


def _zero_compute() -> dict[str, int | float]:
    return {
        "sample_requests": 0,
        "score_requests": 0,
        "forward_calls": 0,
        "model_sequences": 0,
        "model_token_slots": 0,
        "generated_tokens": 0,
        "elapsed_seconds": 0.0,
        "total_parameters": 0,
        "active_parameters": 0,
        "sample_forward_calls": 0,
        "score_forward_calls": 0,
        "sample_model_sequences": 0,
        "score_model_sequences": 0,
        "sample_model_token_slots": 0,
        "score_model_token_slots": 0,
        "sample_elapsed_seconds": 0.0,
        "score_elapsed_seconds": 0.0,
        "estimated_active_flops": 0,
        "estimated_sample_active_flops": 0,
        "estimated_score_active_flops": 0,
    }


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _synchronize_cuda(device: str) -> None:
    if not device.startswith("cuda"):
        return
    import torch

    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/gsm8k_llada_moe_3090.toml")
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/dllm/gsm8k")
    )
    parser.add_argument("--tag", default="paired")
    parser.add_argument("--draw-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config, _ = load_pairing(args.config)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        config["run"]["sample_count"] = args.limit
    if args.draw_index < 0:
        raise ValueError("--draw-index must be non-negative")
    device = str(config["runtime"]["device"])
    if device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    model_dir = Path(str(config["model"]["path"]))
    weight_files = tuple(str(value) for value in config["model"]["weight_files"])
    expected_hashes = tuple(str(value) for value in config["model"]["weight_sha256"])
    expected_sizes = tuple(int(value) for value in config["model"]["weight_bytes"])
    if not (len(weight_files) == len(expected_hashes) == len(expected_sizes)):
        raise ValueError("LLaDA weight manifest columns have different lengths")
    actual_hashes: dict[str, str] = {}
    for name, expected_hash, expected_size in zip(
        weight_files, expected_hashes, expected_sizes, strict=True
    ):
        weight = model_dir / name
        if not weight.is_file():
            raise FileNotFoundError(
                f"pinned LLaDA weight is absent: {weight}; run experiments/dllm/download_llada.py"
            )
        if weight.stat().st_size != expected_size:
            raise ValueError(f"LLaDA weight size does not match the manifest: {weight}")
        actual_hash = _file_sha256(weight)
        if actual_hash != expected_hash:
            raise ValueError(f"LLaDA weight hash does not match the manifest: {weight}")
        actual_hashes[name] = actual_hash
    problems = select_problems(
        load_gsm8k(args.data),
        int(config["run"]["sample_count"]),
        seed=int(config["run"]["subset_seed"]),
    )
    effective = {
        "config": config,
        "method": args.method,
        "tag": args.tag,
        "draw_index": args.draw_index,
        "model_weight_sha256": actual_hashes,
        "problem_indices": [problem.index for problem in problems],
        "implementation_sha256": {
            path: _file_sha256(Path(path)) for path in IMPLEMENTATION_FILES
        },
    }
    fingerprint = _fingerprint(effective)
    run_dir = args.output_root / args.tag / args.method / f"draw-{args.draw_index}"
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    records_path = run_dir / "records.jsonl"
    summary_path = run_dir / "summary.json"
    manifest = {"fingerprint": fingerprint, **effective}
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint:
            raise ValueError(f"existing run has a different fingerprint: {run_dir}")
    else:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    role = "aligned" if args.method.startswith("vrpo_") else "base"
    backend = load_llada_backend(config, role)
    proposal_backend = (
        load_llada_backend(config, "proposal", base_backend=backend)
        if "reduced_layer_proposal" in args.method
        else None
    )
    prior_records = _load_records(records_path)
    completed = {int(record["problem_index"]) for record in prior_records}
    seeds = SeedStream(int(config["run"]["seed"]))
    try:
        with records_path.open("a", encoding="utf-8", buffering=1) as output:
            for problem in problems:
                if problem.index in completed:
                    continue
                prompt = backend.encode_chat(gsm8k_prompt(problem.question))
                main_before = backend.snapshot()
                proposal_before = (
                    proposal_backend.snapshot() if proposal_backend is not None else None
                )
                _synchronize_cuda(device)
                started = time.perf_counter()
                tokens, diagnostics = run_method(
                    args.method,
                    backend,
                    problem,
                    prompt,
                    config,
                    seed=seeds.derive(
                        "dllm-gsm8k", args.method, args.draw_index, problem.index
                    ),
                    proposal_backend=proposal_backend,
                )
                _synchronize_cuda(device)
                elapsed = time.perf_counter() - started
                main_compute = _snapshot_delta(main_before, backend.snapshot())
                proposal_compute = (
                    _snapshot_delta(proposal_before, proposal_backend.snapshot())
                    if proposal_backend is not None and proposal_before is not None
                    else _zero_compute()
                )
                text = backend.decode(tokens)
                prediction = extract_numeric_answer(text)
                record = {
                    "fingerprint": fingerprint,
                    "method": args.method,
                    "draw_index": args.draw_index,
                    "problem_index": problem.index,
                    "gold_answer": str(problem.gold_answer),
                    "prediction": str(prediction) if prediction is not None else None,
                    "correct": prediction == problem.gold_answer,
                    "prompt_tokens": len(prompt),
                    "selected_output_tokens": len(tokens),
                    "output_token_ids": list(tokens),
                    "output_text": text,
                    "elapsed_seconds": elapsed,
                    "main_compute": main_compute,
                    "proposal_compute": proposal_compute,
                    "diagnostics": diagnostics,
                }
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                prior_records.append(record)
                completed.add(problem.index)
                print(
                    f"{args.method} {len(completed)}/{len(problems)} "
                    f"index={problem.index} correct={int(record['correct'])} "
                    f"seconds={elapsed:.3f}",
                    flush=True,
                )
        ordered = sorted(prior_records, key=lambda record: int(record["problem_index"]))
        summary = {"fingerprint": fingerprint, **summarize(ordered)}
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        del proposal_backend
        del backend
        gc.collect()
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
