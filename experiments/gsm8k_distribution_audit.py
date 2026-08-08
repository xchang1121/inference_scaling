"""Empirical answer-distribution audit for GRPO and same-target MH/IS.

Sequence-level distributions are too large to enumerate for a language model.
This diagnostic therefore states its limitation explicitly and compares the
induced distributions of parsed final answers on fixed public GSM8K prompts.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import statistics
import time
import tomllib
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

import torch

from gsm8k_reproduction import (
    IMPLEMENTATION_FILES,
    _file_sha256,
    _load_backend,
    _prompt_tokens,
    _run_method,
    _sample_one,
    _snapshot_delta,
)
from inference_scaling.evaluation import extract_numeric_answer, load_gsm8k, select_problems
from inference_scaling.rng import SeedStream

DEFAULT_METHODS = (
    "base",
    "rl_sample",
    "verifier_mh",
    "verifier_conditional_is",
    "verifier_conditional_is_small_proposal",
)
AUDIT_IMPLEMENTATION_FILES = (
    *IMPLEMENTATION_FILES,
    "experiments/gsm8k_distribution_audit.py",
)


def _answer_key(answer: Fraction | None) -> str:
    if answer is None:
        return "[invalid]"
    if answer.denominator == 1:
        return str(answer.numerator)
    return f"{answer.numerator}/{answer.denominator}"


def _distribution(counts: Mapping[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {answer: count / total for answer, count in counts.items()}


def _tv(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    support = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in support)


def _js(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    support = set(left) | set(right)
    midpoint = {key: (left.get(key, 0.0) + right.get(key, 0.0)) / 2 for key in support}

    def kl(distribution: Mapping[str, float]) -> float:
        return sum(
            probability * math.log2(probability / midpoint[key])
            for key, probability in distribution.items()
            if probability > 0
        )

    return 0.5 * (kl(left) + kl(right))


def _method_backend(config: dict[str, Any], method: str):
    if method.startswith("rl_"):
        adapter_base = (
            str(config["models"]["rl_base"])
            if config["models"].get("rl_kind") == "peft_adapter"
            else None
        )
        return _load_backend(
            str(config["models"]["rl"]), config, adapter_base=adapter_base
        )
    return _load_backend(str(config["models"]["base"]), config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_quick.toml"))
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--problem-count", type=int, default=4)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument(
        "--output", type=Path, default=Path("results/gsm8k_distribution_audit.json")
    )
    args = parser.parse_args()
    if args.problem_count <= 0 or args.draws <= 0:
        raise ValueError("problem-count and draws must be positive")

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    problems = select_problems(
        load_gsm8k(args.data),
        args.problem_count,
        seed=int(config["run"]["subset_seed"]),
    )
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = set(methods) - set(DEFAULT_METHODS)
    if unknown:
        raise ValueError(f"unknown distribution-audit methods: {sorted(unknown)}")

    input_weight_hashes = {
        "base": _file_sha256(
            Path(str(config["models"]["base"])) / "model.safetensors"
        )
    }
    if input_weight_hashes["base"] != str(config["models"]["base_weight_sha256"]):
        raise ValueError("base model weight hash does not match the pinned configuration")
    if "rl_sample" in methods:
        input_weight_hashes["rl_adapter"] = _file_sha256(
            Path(str(config["models"]["rl"])) / "adapter_model.safetensors"
        )
    if any(method.endswith("small_proposal") for method in methods):
        input_weight_hashes["proposal"] = _file_sha256(
            Path(str(config["models"]["proposal"])) / "model.safetensors"
        )
        if input_weight_hashes["proposal"] != str(
            config["models"]["proposal_weight_sha256"]
        ):
            raise ValueError(
                "proposal model weight hash does not match the pinned configuration"
            )

    records: dict[str, dict[int, Counter[str]]] = {}
    runtimes: dict[str, float] = {}
    correctness: dict[str, tuple[int, int]] = {}
    compute: dict[str, dict[str, int | float]] = {}
    for method in methods:
        backend = _method_backend(config, method)
        proposal_backend = None
        if method.endswith("small_proposal"):
            proposal_backend = _load_backend(str(config["models"]["proposal"]), config)
            if backend.tokenizer.get_vocab() != proposal_backend.tokenizer.get_vocab():
                raise ValueError("base and proposal tokenizers do not have identical vocabularies")
        warm_prompt = _prompt_tokens(backend, problems[0])
        _sample_one(
            backend,
            warm_prompt,
            max_new_tokens=2,
            temperature=1.0,
            seed=int(config["run"]["seed"]),
            request_id=f"distribution-audit-warmup:{method}",
        )
        if proposal_backend is not None:
            _sample_one(
                proposal_backend,
                warm_prompt,
                max_new_tokens=2,
                temperature=1.0,
                seed=int(config["run"]["seed"]),
                request_id=f"distribution-audit-proposal-warmup:{method}",
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        method_counts: dict[int, Counter[str]] = {
            problem.index: Counter() for problem in problems
        }
        correct = 0
        backend_before = backend.snapshot()
        proposal_before = proposal_backend.snapshot() if proposal_backend else None
        started = time.perf_counter()
        for draw in range(args.draws):
            draw_config = copy.deepcopy(config)
            draw_config["run"]["seed"] = int(config["run"]["seed"]) + 1_000_003 * draw
            seeds = SeedStream(int(draw_config["run"]["seed"]))
            for problem in problems:
                prompt = _prompt_tokens(backend, problem)
                tokens, _ = _run_method(
                    method,
                    backend,
                    problem,
                    prompt,
                    draw_config,
                    seeds,
                    proposal_backend,
                )
                answer = extract_numeric_answer(backend.decode(tokens))
                method_counts[problem.index][_answer_key(answer)] += 1
                correct += int(answer == problem.gold_answer)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        runtimes[method] = time.perf_counter() - started
        backend_after = backend.snapshot()
        proposal_after = proposal_backend.snapshot() if proposal_backend else None
        base_delta = _snapshot_delta(backend_before, backend_after)
        proposal_delta = (
            _snapshot_delta(proposal_before, proposal_after)
            if proposal_before is not None and proposal_after is not None
            else {}
        )
        total_slots = (
            int(base_delta["generation_forward_token_slots"])
            + int(base_delta["score_forward_token_slots"])
            + int(proposal_delta.get("generation_forward_token_slots", 0))
            + int(proposal_delta.get("score_forward_token_slots", 0))
        )
        total_flops = int(base_delta["estimated_dense_forward_flops"]) + int(
            proposal_delta.get("estimated_dense_forward_flops", 0)
        )
        compute[method] = {
            "forward_token_slots": total_slots,
            "estimated_dense_forward_flops": total_flops,
            "estimated_dense_forward_petaflops": total_flops / 1e15,
        }
        correctness[method] = (correct, args.problem_count * args.draws)
        records[method] = method_counts
        del proposal_backend
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    reference = "rl_sample"
    if reference not in records:
        raise ValueError("the audit requires rl_sample as its GRPO reference")
    comparisons: dict[str, Any] = {}
    for method in methods:
        if method == reference:
            continue
        per_problem = []
        for problem in problems:
            left = _distribution(records[method][problem.index])
            right = _distribution(records[reference][problem.index])
            per_problem.append(
                {
                    "problem_index": problem.index,
                    "jensen_shannon_bits": _js(left, right),
                    "total_variation": _tv(left, right),
                }
            )
        comparisons[f"{method}_vs_{reference}"] = {
            "mean_jensen_shannon_bits": statistics.fmean(
                item["jensen_shannon_bits"] for item in per_problem
            ),
            "mean_total_variation": statistics.fmean(
                item["total_variation"] for item in per_problem
            ),
            "per_problem": per_problem,
        }

    report = {
        "schema_version": 1,
        "public_dataset": "OpenAI GSM8K official test split",
        "problem_indices": [problem.index for problem in problems],
        "input_weight_sha256": input_weight_hashes,
        "implementation_sha256": {
            path: _file_sha256(Path(path)) for path in AUDIT_IMPLEMENTATION_FILES
        },
        "draws_per_problem": args.draws,
        "distribution_level": "parsed final answer, not full token sequence",
        "target_by_method": {
            "base": "base_probability",
            "rl_sample": (
                "finite-step GRPO approximation to base_probability_times_exp_exact_reward_over_beta"
            ),
            "verifier_mh": "base_probability_times_exp_exact_reward_over_beta",
            "verifier_conditional_is": (
                "finite-rollout approximation to base_probability_times_exp_exact_reward_over_beta"
            ),
            "verifier_conditional_is_small_proposal": (
                "clipped off-policy finite-rollout approximation to "
                "base_probability_times_exp_exact_reward_over_beta"
            ),
        },
        "reward_temperature_beta": float(config["matched_target"]["reward_temperature"]),
        "methods": {
            method: {
                "accuracy_over_draws": correctness[method][0] / correctness[method][1],
                "correct": correctness[method][0],
                "samples": correctness[method][1],
                "seconds_excluding_model_load": runtimes[method],
                **compute[method],
                "answer_counts": {
                    str(index): dict(sorted(counts.items()))
                    for index, counts in records[method].items()
                },
            }
            for method in methods
        },
        "comparisons": comparisons,
        "compute_definition": (
            "2 * each model's parameter count * observed padded forward token slots; "
            "the small proposal is accounted separately before summation"
        ),
        "limitation": (
            "Finite draws estimate answer-level distributions only. They cannot establish "
            "equality of the underlying full sequence distributions."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
