"""Reproducible real-model checks for the unified inference-scaling stack.

This script deliberately uses FP32 by default.  On the reference RTX 3090,
reduced-precision logits varied enough with batch shape to distort importance
weights even though ordinary text generation still looked reasonable.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import fmean
from typing import Any, TypeVar

import torch
import transformers

from inference_scaling.algorithms.base_replay import base_replay_step
from inference_scaling.algorithms.conditional_energy import run_conditional_is
from inference_scaling.algorithms.dynamic_is import CandidateProposal, run_dynamic_is
from inference_scaling.algorithms.mh import run_mh_chains
from inference_scaling.backends import (
    ContinuousBatchingBackend,
    ScoreCachingBackend,
    TransformersBackend,
)
from inference_scaling.config import (
    BaseReplayConfig,
    ConditionalEnergyConfig,
    DynamicISConfig,
    MHConfig,
    SamplingConfig,
)
from inference_scaling.metrics import importance_effective_sample_size
from inference_scaling.replay import (
    BehaviorPolicy,
    BehaviorRegistry,
    InMemoryReplayStore,
    ReplayKey,
    ReplaySampleRequest,
    sample_replay_records,
)
from inference_scaling.rng import SeedStream
from inference_scaling.types import GenerationRequest, ScoreRequest, TokenSequence

MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_WEIGHT_SHA256 = "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe"
T = TypeVar("T")

try:
    from experiments.shared.artifacts import (
        dataclass_snapshot_delta as _snapshot_delta,
        file_sha256 as _file_sha256,
    )
except ModuleNotFoundError:  # direct execution from experiments/
    from shared.artifacts import (
        dataclass_snapshot_delta as _snapshot_delta,
        file_sha256 as _file_sha256,
    )


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(function: Callable[[], T]) -> tuple[T, float]:
    _synchronize()
    start = time.perf_counter()
    value = function()
    _synchronize()
    return value, time.perf_counter() - start


def _chat_prefix(backend: TransformersBackend, question: str) -> TokenSequence:
    text = backend.tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return backend.encode(text, add_special_tokens=False)


def _integer_values(text: str) -> tuple[int, ...]:
    values: list[int] = []
    for match in re.findall(r"(?<![\w.])-?\d[\d,]*(?![\w.])", text):
        try:
            values.append(int(match.replace(",", "")))
        except ValueError:
            continue
    return tuple(values)


def _is_correct(text: str, answer: int) -> bool:
    values = _integer_values(text)
    return bool(values) and values[-1] == answer


def _reward_function(
    backend: TransformersBackend, answer: int
) -> Callable[[TokenSequence, TokenSequence], float]:
    def reward(_prompt: TokenSequence, generated: TokenSequence) -> float:
        return 1.0 if _is_correct(backend.decode(generated), answer) else 0.0

    return reward


def _environment(backend: TransformersBackend, dtype: str) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "model_path": backend.model_id,
        "model_revision": MODEL_REVISION,
        "model_weight_sha256": MODEL_WEIGHT_SHA256,
        "model_weight_sha256_verified": True,
        "dtype": dtype,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_memory_bytes": properties.total_memory,
    }


def _verify_model_weight(model_path: str) -> None:
    weight_path = Path(model_path) / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"missing pinned model weight: {weight_path}")
    actual = _file_sha256(weight_path)
    if actual != MODEL_WEIGHT_SHA256:
        raise ValueError(
            "model.safetensors does not match the pinned reproduction weight: "
            f"expected {MODEL_WEIGHT_SHA256}, got {actual}"
        )


def run_backend_checks(backend: TransformersBackend) -> dict[str, Any]:
    prefix = _chat_prefix(
        backend, "Compute 37 * 48. Give only the integer, with no explanation."
    )
    sampling = SamplingConfig()
    warmup = [
        GenerationRequest(prefix, 4, sampling, 10 + index, f"warmup-{index}")
        for index in range(2)
    ]
    backend.sample_batch(warmup)
    torch.cuda.reset_peak_memory_stats()

    requests = [
        GenerationRequest(prefix, 24, sampling, 100 + index, f"benchmark-{index}")
        for index in range(8)
    ]
    before_sequential = backend.snapshot()
    sequential, sequential_seconds = _timed(
        lambda: [backend.sample_batch([request])[0] for request in requests]
    )
    after_sequential = backend.snapshot()
    batched, batched_seconds = _timed(lambda: backend.sample_batch(requests))
    after_batched = backend.snapshot()
    rescored, score_seconds = _timed(
        lambda: backend.score_batch(
            [ScoreRequest(prefix, tuple(sample.token_ids for sample in batched), sampling)]
        )
    )
    errors = [
        abs(generated_logprob - rescored_logprob)
        for sample, token_scores in zip(batched, rescored, strict=True)
        for generated_logprob, rescored_logprob in zip(
            sample.token_logprobs, token_scores, strict=True
        )
    ]

    asynchronous_requests = [
        GenerationRequest(prefix, 12, sampling, 800 + index, f"async-{index}")
        for index in range(8)
    ]
    with ContinuousBatchingBackend(
        backend,
        max_batch_size=8,
        max_batch_tokens=4096,
        batch_wait_seconds=0.01,
    ) as batching:
        asynchronous, asynchronous_seconds = _timed(
            lambda: _concurrent_single_request_generation(
                batching, asynchronous_requests
            )
        )
        batching_snapshot = asdict(batching.snapshot())

    return {
        "batch_size": len(requests),
        "tokens_per_request": requests[0].max_new_tokens,
        "sequential_seconds": sequential_seconds,
        "batched_seconds": batched_seconds,
        "batch_speedup": sequential_seconds / batched_seconds,
        "batched_tokens_per_second": sum(len(item.token_ids) for item in batched)
        / batched_seconds,
        "batched_score_seconds": score_seconds,
        "max_generation_rescore_logprob_error": max(errors, default=0.0),
        "mean_generation_rescore_logprob_error": fmean(errors) if errors else 0.0,
        "sequential_and_batched_tokens_equal": [item.token_ids for item in sequential]
        == [item.token_ids for item in batched],
        "sequential_backend_delta": _snapshot_delta(
            before_sequential, after_sequential
        ),
        "batched_backend_delta": _snapshot_delta(after_sequential, after_batched),
        "asynchronous_seconds": asynchronous_seconds,
        "asynchronous_outputs": len(asynchronous),
        "continuous_batching": batching_snapshot,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
    }


def _concurrent_single_request_generation(
    backend: ContinuousBatchingBackend,
    requests: Sequence[GenerationRequest],
) -> list[Any]:
    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = [executor.submit(backend.sample_batch, [request]) for request in requests]
        return [future.result()[0] for future in futures]


def run_mh_check(
    backend: TransformersBackend,
    cached_backend: ScoreCachingBackend,
) -> dict[str, Any]:
    prefix = _chat_prefix(
        backend, "Complete this answer in a short, conventional way: The capital of France is"
    )
    proposal = SamplingConfig()
    config = MHConfig(
        alpha=2.0,
        total_length=16,
        block_size=8,
        steps_per_block=3,
        chains=4,
    )
    base_requests = [
        GenerationRequest(prefix, config.total_length, proposal, 200 + index, f"base-{index}")
        for index in range(config.chains)
    ]
    base_samples, base_seconds = _timed(
        lambda: cached_backend.sample_batch(base_requests)
    )
    chains, mh_seconds = _timed(
        lambda: run_mh_chains(
            cached_backend,
            prefix,
            config,
            proposal,
            SeedStream(20260808),
        )
    )
    base_logprob_per_token = [
        sample.logprob / len(sample.token_ids) for sample in base_samples
    ]
    mh_logprob_per_token = [
        sum(chain.base_token_logprobs) / len(chain.token_ids) for chain in chains
    ]
    return {
        "config": asdict(config),
        "base_seconds": base_seconds,
        "mh_seconds": mh_seconds,
        "acceptance_rate": sum(chain.accepted for chain in chains)
        / sum(chain.attempts for chain in chains),
        "base_mean_logprob_per_token": fmean(base_logprob_per_token),
        "mh_mean_logprob_per_token": fmean(mh_logprob_per_token),
        "mean_logprob_gain_per_token": fmean(mh_logprob_per_token)
        - fmean(base_logprob_per_token),
        "base_outputs": [backend.decode(sample.token_ids) for sample in base_samples],
        "mh_outputs": [backend.decode(chain.token_ids) for chain in chains],
    }


ARITHMETIC_TASKS = (
    ("Compute 29 + 37. Give only the integer, with no explanation.", 66),
    ("Compute 27 * 14. Give only the integer, with no explanation.", 378),
    ("Compute 123 + 456. Give only the integer, with no explanation.", 579),
    ("Compute 99 - 37. Give only the integer, with no explanation.", 62),
)


def _conditional_diagnostics(result: Any) -> tuple[float, float, float]:
    effective_sizes: list[float] = []
    raw_corrections: list[float] = []
    applied_corrections: list[float] = []
    for step in result.steps:
        for candidate in step.candidates:
            effective_sizes.append(
                importance_effective_sample_size(
                    [rollout.log_weight for rollout in candidate.rollouts]
                )
            )
            raw_corrections.extend(
                rollout.raw_log_importance_ratio for rollout in candidate.rollouts
            )
            applied_corrections.extend(
                rollout.applied_log_importance_ratio
                for rollout in candidate.rollouts
            )
    return (
        fmean(effective_sizes) if effective_sizes else 0.0,
        fmean(abs(value) for value in raw_corrections) if raw_corrections else 0.0,
        fmean(abs(value) for value in applied_corrections)
        if applied_corrections
        else 0.0,
    )


def run_conditional_checks(
    backend: TransformersBackend,
    cached_backend: ScoreCachingBackend,
) -> dict[str, Any]:
    eos = backend.tokenizer.eos_token_id
    base_sampling = SamplingConfig(eos_token_id=eos)
    config = ConditionalEnergyConfig(
        candidate_count=4,
        rollout_count=4,
        block_size=8,
        total_length=24,
        reward_temperature=0.25,
    )
    prefixes = [_chat_prefix(backend, question) for question, _ in ARITHMETIC_TASKS]
    baseline_requests = [
        GenerationRequest(prefix, config.total_length, base_sampling, 300 + index, f"baseline-{index}")
        for index, prefix in enumerate(prefixes)
    ]
    baseline, baseline_seconds = _timed(
        lambda: cached_backend.sample_batch(baseline_requests)
    )

    def run_policy(temperature: float) -> tuple[list[Any], float]:
        rollout_sampling = SamplingConfig(
            temperature=temperature,
            eos_token_id=eos,
        )

        def run_all() -> list[Any]:
            return [
                run_conditional_is(
                    cached_backend,
                    prefix,
                    config,
                    _reward_function(backend, answer),
                    SeedStream(4000 + index),
                    base_sampling=base_sampling,
                    rollout_backend=cached_backend,
                    rollout_sampling=rollout_sampling,
                )
                for index, (prefix, (_, answer)) in enumerate(
                    zip(prefixes, ARITHMETIC_TASKS, strict=True)
                )
            ]

        return _timed(run_all)

    on_policy, on_policy_seconds = run_policy(1.0)
    off_policy, off_policy_seconds = run_policy(0.7)

    def summarize(results: Sequence[Any], seconds: float) -> dict[str, Any]:
        texts = [backend.decode(result.token_ids) for result in results]
        diagnostics = [_conditional_diagnostics(result) for result in results]
        return {
            "seconds": seconds,
            "accuracy": sum(
                _is_correct(text, answer)
                for text, (_, answer) in zip(texts, ARITHMETIC_TASKS, strict=True)
            )
            / len(ARITHMETIC_TASKS),
            "average_completion_ess": fmean(value[0] for value in diagnostics),
            "mean_absolute_raw_log_importance_correction": fmean(
                value[1] for value in diagnostics
            ),
            "mean_absolute_applied_log_importance_correction": fmean(
                value[2] for value in diagnostics
            ),
            "outputs": texts,
        }

    baseline_texts = [backend.decode(sample.token_ids) for sample in baseline]
    return {
        "config": asdict(config),
        "tasks": [
            {"question": question, "answer": answer}
            for question, answer in ARITHMETIC_TASKS
        ],
        "baseline": {
            "seconds": baseline_seconds,
            "accuracy": sum(
                _is_correct(text, answer)
                for text, (_, answer) in zip(
                    baseline_texts, ARITHMETIC_TASKS, strict=True
                )
            )
            / len(ARITHMETIC_TASKS),
            "outputs": baseline_texts,
        },
        "on_policy": summarize(on_policy, on_policy_seconds),
        "off_policy_temperature_0_7": summarize(
            off_policy, off_policy_seconds
        ),
    }


def run_replay_check(
    backend: TransformersBackend,
    cached_backend: ScoreCachingBackend,
) -> dict[str, Any]:
    question, answer = ARITHMETIC_TASKS[0]
    prompt = _chat_prefix(backend, question)
    reward = _reward_function(backend, answer)
    eos = backend.tokenizer.eos_token_id
    base_sampling = SamplingConfig(eos_token_id=eos)
    behavior_sampling = SamplingConfig(temperature=0.7, eos_token_id=eos)
    config = BaseReplayConfig(
        candidate_count=4,
        block_size=1,
        total_length=8,
        reward_temperature=0.25,
        max_history_per_candidate=2,
        fresh_rollouts=1,
        truncation=8.0,
    )
    seeds = SeedStream(5000)
    probe_registry = BehaviorRegistry()
    probe_store = InMemoryReplayStore()
    probe, probe_seconds = _timed(
        lambda: base_replay_step(
            base_backend=cached_backend,
            registry=probe_registry,
            store=probe_store,
            prompt=prompt,
            generated_prefix=(),
            config=config,
            base_sampling=base_sampling,
            reward=reward,
            reward_version="integer-match-v1",
            seeds=seeds,
            step_index=0,
        )
    )
    behavior = BehaviorPolicy.for_backend(
        cached_backend, behavior_sampling, label="temperature-0.7-history"
    )
    registry = BehaviorRegistry()
    registry.register(behavior)
    store = InMemoryReplayStore()
    history_requests: list[ReplaySampleRequest] = []
    for candidate_index, candidate in enumerate(probe.candidates):
        if candidate.token_ids[-1] == eos:
            continue
        key = ReplayKey(prompt, (), candidate.token_ids, "integer-match-v1")
        for history_index in range(config.max_history_per_candidate):
            history_requests.append(
                ReplaySampleRequest(
                    key=key,
                    max_new_tokens=config.total_length - config.block_size,
                    seed=seeds.derive(
                        "controlled-history", candidate_index, history_index
                    ),
                    record_id=f"controlled-history:{candidate_index}:{history_index}",
                )
            )
    records, history_generation_seconds = _timed(
        lambda: sample_replay_records(behavior, history_requests, reward)
    )
    for record in records:
        store.add_evaluation(record)
    evaluation_before = store.evaluation_count
    replayed, replay_seconds = _timed(
        lambda: base_replay_step(
            base_backend=cached_backend,
            registry=registry,
            store=store,
            prompt=prompt,
            generated_prefix=(),
            config=config,
            base_sampling=base_sampling,
            reward=reward,
            reward_version="integer-match-v1",
            seeds=seeds,
            step_index=0,
        )
    )
    return {
        "config": asdict(config),
        "probe_seconds": probe_seconds,
        "history_generation_seconds": history_generation_seconds,
        "replay_decision_seconds": replay_seconds,
        "candidate_draws_reproduced": [item.token_ids for item in probe.candidates]
        == [item.token_ids for item in replayed.candidates],
        "history_records_available_before_decision": evaluation_before,
        "history_counts_used": [
            candidate.estimate.history_count for candidate in replayed.candidates
        ],
        "fresh_counts_used": [
            candidate.estimate.fresh_count for candidate in replayed.candidates
        ],
        "history_ess": [
            candidate.estimate.history_ess for candidate in replayed.candidates
        ],
        "evaluation_records_after_decision": store.evaluation_count,
        "reserved_records_after_decision": store.reserved_count,
        "design_records_after_decision": store.design_count,
        "selected_candidate_text": backend.decode(replayed.selected.token_ids),
    }


def run_dynamic_check(
    backend: TransformersBackend,
    cached_backend: ScoreCachingBackend,
) -> dict[str, Any]:
    question, answer = ARITHMETIC_TASKS[1]
    prompt = _chat_prefix(backend, question)
    eos = backend.tokenizer.eos_token_id
    base_sampling = SamplingConfig(eos_token_id=eos)
    auxiliary = CandidateProposal.for_backend(
        cached_backend,
        SamplingConfig(temperature=0.7, eos_token_id=eos),
        label="temperature-0.7-candidate-proposal",
    )
    config = DynamicISConfig(
        candidate_count=8,
        block_size=2,
        total_length=12,
        reward_temperature=0.25,
        max_history_per_candidate=0,
        rollout_budget=8.0,
        auxiliary_mixture=0.5,
        minimum_fresh_per_candidate=1,
    )
    before_cache = cached_backend.snapshot()
    result, seconds = _timed(
        lambda: run_dynamic_is(
            cached_backend,
            BehaviorRegistry(),
            InMemoryReplayStore(),
            prompt,
            config,
            _reward_function(backend, answer),
            "integer-match-v1",
            SeedStream(6000),
            base_sampling=base_sampling,
            auxiliary_proposal=auxiliary,
        )
    )
    after_cache = cached_backend.snapshot()
    text = backend.decode(result.token_ids)
    source_counts = Counter(
        candidate.draw.source
        for step in result.steps
        for candidate in step.candidates
    )
    return {
        "config": asdict(config),
        "seconds": seconds,
        "output": text,
        "correct": _is_correct(text, answer),
        "steps": len(result.steps),
        "candidate_source_counts": dict(sorted(source_counts.items())),
        "outer_weight_ess_by_step": [
            importance_effective_sample_size(
                [candidate.log_weight for candidate in step.candidates]
            )
            for step in result.steps
        ],
        "history_allocations": [
            [candidate.allocation.history_count for candidate in step.candidates]
            for step in result.steps
        ],
        "fresh_allocations": [
            [candidate.allocation.fresh_count for candidate in step.candidates]
            for step in result.steps
        ],
        "score_cache_delta": _snapshot_delta(before_cache, after_cache),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="models/Qwen2.5-0.5B-Instruct",
        help="Local Transformers model directory",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument(
        "--section",
        choices=("all", "backend", "algorithms"),
        default="all",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    _verify_model_weight(args.model)
    backend, load_seconds = _timed(
        lambda: TransformersBackend.from_pretrained(
            args.model,
            device=args.device,
            dtype=args.dtype,
            local_files_only=True,
        )
    )
    cached_backend = ScoreCachingBackend(backend)
    report: dict[str, Any] = {
        "environment": _environment(backend, args.dtype),
        "model_load_seconds": load_seconds,
    }
    if args.section in ("all", "backend"):
        report["backend"] = run_backend_checks(backend)
    if args.section in ("all", "algorithms"):
        report["mh"] = run_mh_check(backend, cached_backend)
        report["conditional_importance_sampling"] = run_conditional_checks(
            backend, cached_backend
        )
        report["off_policy_replay"] = run_replay_check(backend, cached_backend)
        report["dynamic_candidate_importance_sampling"] = run_dynamic_check(
            backend, cached_backend
        )
        report["final_score_cache"] = asdict(cached_backend.snapshot())

    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(serialized)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
