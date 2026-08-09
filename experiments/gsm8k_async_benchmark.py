"""Compare synchronous and continuous-batched execution across TTS baselines."""

from __future__ import annotations

import argparse
import json
import statistics
import tomllib
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

if __package__:
    from experiments.gsm8k_reproduction import (
        IMPLEMENTATION_FILES,
        _file_sha256,
        _load_backend,
        _prompt_tokens,
        _snapshot_delta,
        _timed,
    )
else:
    from gsm8k_reproduction import (
        IMPLEMENTATION_FILES,
        _file_sha256,
        _load_backend,
        _prompt_tokens,
        _snapshot_delta,
        _timed,
    )
from inference_scaling.algorithms import run_conditional_is
from inference_scaling.backends import (
    BACKEND_CHOICES,
    ContinuousBatchingBackend,
    ScoreCachingBackend,
    close_backend,
    set_backend_override,
)
from inference_scaling.config import ConditionalEnergyConfig, SamplingConfig
from inference_scaling.evaluation import (
    CumulativeConsensusReward,
    consensus_index,
    extract_numeric_answer,
    load_gsm8k,
    select_problems,
)
from inference_scaling.rng import SeedStream
from inference_scaling.types import GenerationRequest, TokenSequence

METHODS = (
    "base",
    "best_of_n",
    "conditional_is",
    "conditional_is_small_proposal",
)
ASYNC_IMPLEMENTATION_FILES = (
    *IMPLEMENTATION_FILES,
    "experiments/gsm8k_async_benchmark.py",
    "src/inference_scaling/backends/batching.py",
)


def _sampling(raw_backend, config: dict[str, Any]) -> SamplingConfig:
    return SamplingConfig(
        temperature=float(config.get("sampling", {}).get("temperature", 1.0)),
        eos_token_id=raw_backend.tokenizer.eos_token_id,
    )


def _run_one(
    method: str,
    base_backend,
    proposal_backend,
    raw_backend,
    prompt: TokenSequence,
    problem,
    config: dict[str, Any],
    root_seed: int,
) -> TokenSequence:
    maximum = int(config["generation"]["max_new_tokens"])
    sampling = _sampling(raw_backend, config)
    seeds = SeedStream(
        SeedStream(root_seed).derive("async-benchmark", method, problem.index)
    )
    if method == "base":
        return base_backend.sample_batch(
            [
                GenerationRequest(
                    prompt,
                    maximum,
                    sampling,
                    seeds.derive("sample"),
                    f"async-base:{problem.index}",
                )
            ]
        )[0].token_ids

    if method == "best_of_n":
        count = int(config["best_of_n"]["samples"])
        samples = base_backend.sample_batch(
            [
                GenerationRequest(
                    prompt,
                    maximum,
                    sampling,
                    seeds.derive("candidate", candidate_index),
                    f"async-best-of-n:{problem.index}:{candidate_index}",
                )
                for candidate_index in range(count)
            ]
        )
        texts = [raw_backend.decode(sample.token_ids) for sample in samples]
        selected = consensus_index(texts, [sample.logprob for sample in samples])
        return samples[selected].token_ids

    if method not in {"conditional_is", "conditional_is_small_proposal"}:
        raise ValueError(f"unknown asynchronous benchmark method: {method}")
    if method.endswith("small_proposal") and proposal_backend is None:
        raise ValueError("small-proposal method requires a proposal backend")
    section = config["conditional_is"]
    reward = CumulativeConsensusReward(raw_backend.decode)
    result = run_conditional_is(
        base_backend,
        prompt,
        ConditionalEnergyConfig(
            candidate_count=int(section["candidate_count"]),
            rollout_count=int(section["rollout_count"]),
            block_size=int(section["block_size"]),
            total_length=maximum,
            reward_temperature=float(section["reward_temperature"]),
            importance_log_ratio_clip=(
                float(section["importance_log_ratio_clip"])
                if method.endswith("small_proposal")
                and section.get("importance_log_ratio_clip") is not None
                else None
            ),
        ),
        None,
        seeds,
        base_sampling=sampling,
        rollout_backend=(
            proposal_backend
            if method.endswith("small_proposal")
            else base_backend
        ),
        rollout_sampling=sampling,
        reward_batch=reward,
    )
    return result.token_ids


def _accuracy(raw_backend, outputs, problems) -> float:
    return sum(
        extract_numeric_answer(raw_backend.decode(tokens)) == problem.gold_answer
        for tokens, problem in zip(outputs, problems, strict=True)
    ) / len(problems)


def _common_prefix_length(left: TokenSequence, right: TokenSequence) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def _output_agreement(raw_backend, synchronous, asynchronous, problems) -> dict[str, Any]:
    if len(synchronous) != len(asynchronous) or len(synchronous) != len(problems):
        raise ValueError("paired output diagnostics require equally sized inputs")
    exact = [left == right for left, right in zip(synchronous, asynchronous, strict=True)]
    synchronous_answers = [
        extract_numeric_answer(raw_backend.decode(tokens)) for tokens in synchronous
    ]
    asynchronous_answers = [
        extract_numeric_answer(raw_backend.decode(tokens)) for tokens in asynchronous
    ]
    answer_equal = [
        left == right
        for left, right in zip(synchronous_answers, asynchronous_answers, strict=True)
    ]
    common_prefix_lengths = [
        _common_prefix_length(left, right)
        for left, right in zip(synchronous, asynchronous, strict=True)
    ]
    common_prefix_fractions = [
        common / max(1, len(left), len(right))
        for common, left, right in zip(
            common_prefix_lengths,
            synchronous,
            asynchronous,
            strict=True,
        )
    ]
    mismatches = [
        {
            "gsm8k_index": problem.index,
            "common_prefix_tokens": common,
            "synchronous_tokens": len(left),
            "asynchronous_tokens": len(right),
        }
        for problem, left, right, common, is_exact in zip(
            problems,
            synchronous,
            asynchronous,
            common_prefix_lengths,
            exact,
            strict=True,
        )
        if not is_exact
    ]
    count = len(problems)
    return {
        "outputs_bitwise_equal": all(exact),
        "output_exact_match_count": sum(exact),
        "output_exact_match_fraction": sum(exact) / count,
        "answer_match_count": sum(answer_equal),
        "answer_match_fraction": sum(answer_equal) / count,
        "both_answers_parseable_count": sum(
            left is not None and right is not None
            for left, right in zip(
                synchronous_answers,
                asynchronous_answers,
                strict=True,
            )
        ),
        "mean_common_prefix_fraction": statistics.fmean(common_prefix_fractions),
        "median_common_prefix_fraction": statistics.median(common_prefix_fractions),
        "mismatches": mismatches,
    }


def _compute_delta(base_before, base_after, proposal_before, proposal_after):
    base = _snapshot_delta(base_before, base_after)
    proposal = (
        _snapshot_delta(proposal_before, proposal_after)
        if proposal_before is not None and proposal_after is not None
        else {}
    )
    # A high-water mark is a gauge, not an additive counter.  Report the
    # observed value itself so a warm-up request does not turn concurrency 4
    # into the misleading delta 3.
    if hasattr(base_after, "maximum_in_flight_requests"):
        base["maximum_in_flight_requests"] = int(
            base_after.maximum_in_flight_requests
        )
    if proposal_after is not None and hasattr(
        proposal_after, "maximum_in_flight_requests"
    ):
        proposal["maximum_in_flight_requests"] = int(
            proposal_after.maximum_in_flight_requests
        )
    slots = int(base["generation_forward_token_slots"]) + int(
        base["score_forward_token_slots"]
    )
    slots += int(proposal.get("generation_forward_token_slots", 0)) + int(
        proposal.get("score_forward_token_slots", 0)
    )
    flops = int(base["estimated_dense_forward_flops"]) + int(
        proposal.get("estimated_dense_forward_flops", 0)
    )
    return {
        "base_backend": base,
        "proposal_backend": proposal or None,
        "total_forward_token_slots": slots,
        "estimated_dense_forward_flops": flops,
    }


def _batching_context(stack: ExitStack, raw_backend, config: dict[str, Any]):
    return stack.enter_context(
        ContinuousBatchingBackend(
            raw_backend,
            max_batch_size=int(config["runtime"]["max_batch_size"]),
            max_batch_tokens=int(config["runtime"]["max_batch_tokens"]),
            batch_wait_seconds=0.01,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_quick.toml"))
    parser.add_argument("--backend", choices=BACKEND_CHOICES)
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/gsm8k_async.json"))
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--methods", default=",".join(METHODS))
    args = parser.parse_args()

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    set_backend_override(config, args.backend)
    config["run"]["sample_count"] = args.limit
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    unknown = sorted(set(methods) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown asynchronous methods: {unknown}")
    problems = select_problems(
        load_gsm8k(args.data),
        args.limit,
        seed=int(config["run"]["subset_seed"]),
    )

    base_weight_hash = _file_sha256(
        Path(str(config["models"]["base"])) / "model.safetensors"
    )
    if base_weight_hash != str(config["models"]["base_weight_sha256"]):
        raise ValueError("base model weight hash does not match the pinned configuration")
    proposal_weight_hash = _file_sha256(
        Path(str(config["models"]["proposal"])) / "model.safetensors"
    )
    if proposal_weight_hash != str(config["models"]["proposal_weight_sha256"]):
        raise ValueError("proposal model weight hash does not match the pinned configuration")

    raw_backend = _load_backend(str(config["models"]["base"]), config)
    raw_proposal = _load_backend(str(config["models"]["proposal"]), config)
    if raw_backend.tokenizer.get_vocab() != raw_proposal.tokenizer.get_vocab():
        raise ValueError("base and proposal tokenizers do not have identical vocabularies")
    prompts = [_prompt_tokens(raw_backend, problem) for problem in problems]
    root_seed = int(config["run"]["seed"])
    warm_sampling = _sampling(raw_backend, config)
    raw_backend.sample_batch(
        [GenerationRequest(prompts[0], 2, warm_sampling, root_seed, "warmup-base")]
    )
    raw_proposal.sample_batch(
        [GenerationRequest(prompts[0], 2, warm_sampling, root_seed, "warmup-proposal")]
    )

    workers = min(args.workers, len(problems))
    method_reports: dict[str, Any] = {}
    for method in methods:
        uses_proposal = method.endswith("small_proposal")
        synchronous_base = ScoreCachingBackend(raw_backend)
        synchronous_proposal = (
            ScoreCachingBackend(raw_proposal) if uses_proposal else None
        )
        base_before = raw_backend.snapshot()
        proposal_before = raw_proposal.snapshot() if uses_proposal else None
        synchronous, synchronous_seconds = _timed(
            lambda: [
                _run_one(
                    method,
                    synchronous_base,
                    synchronous_proposal,
                    raw_backend,
                    prompt,
                    problem,
                    config,
                    root_seed,
                )
                for prompt, problem in zip(prompts, problems, strict=True)
            ]
        )
        base_after = raw_backend.snapshot()
        proposal_after = raw_proposal.snapshot() if uses_proposal else None
        synchronous_compute = _compute_delta(
            base_before, base_after, proposal_before, proposal_after
        )

        async_base_before = raw_backend.snapshot()
        async_proposal_before = raw_proposal.snapshot() if uses_proposal else None
        with ExitStack() as stack:
            base_batching = _batching_context(stack, raw_backend, config)
            proposal_batching = (
                _batching_context(stack, raw_proposal, config)
                if uses_proposal
                else None
            )
            asynchronous_base = ScoreCachingBackend(base_batching)
            asynchronous_proposal = (
                ScoreCachingBackend(proposal_batching)
                if proposal_batching is not None
                else None
            )

            def run_parallel():
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(
                            _run_one,
                            method,
                            asynchronous_base,
                            asynchronous_proposal,
                            raw_backend,
                            prompt,
                            problem,
                            config,
                            root_seed,
                        )
                        for prompt, problem in zip(prompts, problems, strict=True)
                    ]
                    return [future.result() for future in futures]

            asynchronous, asynchronous_seconds = _timed(run_parallel)
            batching = {
                "base": asdict(base_batching.snapshot()),
                "proposal": (
                    asdict(proposal_batching.snapshot())
                    if proposal_batching is not None
                    else None
                ),
            }
        async_base_after = raw_backend.snapshot()
        async_proposal_after = raw_proposal.snapshot() if uses_proposal else None
        asynchronous_compute = _compute_delta(
            async_base_before,
            async_base_after,
            async_proposal_before,
            async_proposal_after,
        )
        agreement = _output_agreement(
            raw_backend,
            synchronous,
            asynchronous,
            problems,
        )
        method_reports[method] = {
            "synchronous_seconds": synchronous_seconds,
            "asynchronous_continuous_batching_seconds": asynchronous_seconds,
            "wall_time_speedup_synchronous_over_asynchronous": (
                synchronous_seconds / asynchronous_seconds
            ),
            "wall_time_gain_fraction": 1 - asynchronous_seconds / synchronous_seconds,
            "synchronous_compute": synchronous_compute,
            "asynchronous_compute": asynchronous_compute,
            "synchronous_over_asynchronous_flops": (
                synchronous_compute["estimated_dense_forward_flops"]
                / asynchronous_compute["estimated_dense_forward_flops"]
            ),
            **agreement,
            "synchronous_accuracy": _accuracy(raw_backend, synchronous, problems),
            "asynchronous_accuracy": _accuracy(raw_backend, asynchronous, problems),
            "synchronous_mean_output_tokens": statistics.fmean(
                len(item) for item in synchronous
            ),
            "asynchronous_mean_output_tokens": statistics.fmean(
                len(item) for item in asynchronous
            ),
            "continuous_batching": batching,
        }

    report = {
        "schema_version": 4,
        "benchmark": "GSM8K cross-request scheduling for source-aligned TTS methods",
        "examples": len(problems),
        "workers": workers,
        "runtime_backend": config["runtime"].get("backend", "transformers"),
        "methods": method_reports,
        "algorithm_config": {
            "sampling": config.get("sampling", {"temperature": 1.0}),
            "best_of_n": config["best_of_n"],
            "conditional_is": config["conditional_is"],
            "max_new_tokens": config["generation"]["max_new_tokens"],
        },
        "models": {
            "base": {
                "path": str(config["models"]["base"]),
                "weight_sha256": base_weight_hash,
                "parameter_count": raw_backend.parameter_count,
            },
            "proposal": {
                "path": str(config["models"]["proposal"]),
                "weight_sha256": proposal_weight_hash,
                "parameter_count": raw_proposal.parameter_count,
            },
        },
        "implementation_sha256": {
            path: _file_sha256(Path(path)) for path in ASYNC_IMPLEMENTATION_FILES
        },
        "compute_interpretation": (
            "Continuous batching is applied separately but uniformly to Base, "
            "Best-of-N, conditional IS, and small-proposal conditional IS. Its intended "
            "benefit is higher hardware utilization, not lower algorithmic FLOPs; any "
            "token-slot difference is physical padding, batch formation, or a numerically "
            "diverged live sampling path. Request-local seeds fix the random streams, but "
            "CUDA logits can still vary slightly with batch shape. Exact token agreement, "
            "numeric-answer agreement, common-prefix overlap, and mismatch indices are "
            "therefore reported rather than assumed. A speedup with non-identical outputs "
            "is a representative live-workload comparison, not a fixed-trace comparison."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    close_backend(raw_proposal)
    close_backend(raw_backend)


if __name__ == "__main__":
    main()
