"""Run resumable multi-draw GSM8K comparisons for conditional IS variants."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import statistics
import tomllib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

if __package__:
    from experiments.gsm8k_passk import (
        PASSK_IMPLEMENTATION_FILES,
        _MethodBackend,
        _chunk_plan,
        _estimated_pass_at_k,
        _load_jsonl,
        _prepare_manifest,
        _summarize_method,
        _validate_chunks,
    )
    from experiments.gsm8k_reproduction import (
        _file_sha256,
        _load_backend,
        _prompt_tokens,
        _run_method,
        _sample_one,
        _snapshot_delta,
        _timed,
    )
else:
    from gsm8k_passk import (
        PASSK_IMPLEMENTATION_FILES,
        _MethodBackend,
        _chunk_plan,
        _estimated_pass_at_k,
        _load_jsonl,
        _prepare_manifest,
        _summarize_method,
        _validate_chunks,
    )
    from gsm8k_reproduction import (
        _file_sha256,
        _load_backend,
        _prompt_tokens,
        _run_method,
        _sample_one,
        _snapshot_delta,
        _timed,
    )
from inference_scaling.backends import ContinuousBatchingBackend
from inference_scaling.evaluation import (
    extract_numeric_answer,
    load_gsm8k,
    select_problems,
)
from inference_scaling.rng import SeedStream


IS_PASSK_METHODS = (
    "conditional_is",
    "conditional_is_small_proposal",
)
IS_PASSK_IMPLEMENTATION_FILES = tuple(
    dict.fromkeys((*PASSK_IMPLEMENTATION_FILES, "experiments/gsm8k_is_passk.py"))
)
_BATCHING_SUM_FIELDS = (
    "sample_batches",
    "score_batches",
    "sample_requests",
    "score_sequences",
)
_BATCHING_MAX_FIELDS = ("maximum_sample_batch", "maximum_score_batch")


def _combine_numeric_deltas(
    base: dict[str, int | float],
    proposal: dict[str, int | float] | None,
) -> dict[str, int | float]:
    """Sum physical counters while preserving model-specific deltas separately."""

    proposal = proposal or {}
    if set(proposal) - set(base):
        raise ValueError("proposal backend delta contains fields absent from base delta")
    return {name: value + proposal.get(name, 0) for name, value in base.items()}


def _combine_batching_snapshots(
    base: dict[str, int | float],
    proposal: dict[str, int | float] | None,
) -> dict[str, int | float]:
    proposal = proposal or {}
    return {
        **{
            name: int(base[name]) + int(proposal.get(name, 0))
            for name in _BATCHING_SUM_FIELDS
        },
        **{
            name: max(int(base[name]), int(proposal.get(name, 0)))
            for name in _BATCHING_MAX_FIELDS
        },
    }


def _summarize_model_compute(
    chunks: Sequence[dict[str, Any]], key: str
) -> dict[str, int | float] | None:
    deltas = [chunk.get(key) for chunk in chunks if chunk.get(key) is not None]
    if not deltas:
        return None
    field_names = set(deltas[0])
    if any(set(delta) != field_names for delta in deltas):
        raise ValueError(f"inconsistent {key} fields across IS pass@k chunks")
    totals = {
        name: sum(delta[name] for delta in deltas)
        for name in sorted(field_names)
    }
    total_slots = int(totals["generation_forward_token_slots"]) + int(
        totals["score_forward_token_slots"]
    )
    totals["total_forward_token_slots"] = total_slots
    totals["estimated_dense_forward_petaflops"] = (
        int(totals["estimated_dense_forward_flops"]) / 1e15
    )
    return totals


def _summarize_batching_by_model(
    chunks: Sequence[dict[str, Any]], model: str
) -> dict[str, int] | None:
    snapshots = [
        chunk["continuous_batching_by_model"].get(model)
        for chunk in chunks
        if chunk["continuous_batching_by_model"].get(model) is not None
    ]
    if not snapshots:
        return None
    return {
        **{
            name: sum(int(snapshot[name]) for snapshot in snapshots)
            for name in _BATCHING_SUM_FIELDS
        },
        **{
            name: max(int(snapshot[name]) for snapshot in snapshots)
            for name in _BATCHING_MAX_FIELDS
        },
    }


def _input_weight_hashes(
    config: dict[str, Any], methods: Sequence[str]
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    model_keys = ["base"]
    if any(method.endswith("small_proposal") for method in methods):
        model_keys.append("proposal")
    for key in model_keys:
        path = Path(str(config["models"][key])) / "model.safetensors"
        digest = _file_sha256(path)
        expected = str(config["models"][f"{key}_weight_sha256"])
        if digest != expected:
            raise ValueError(f"{key} model weight hash does not match the pinned configuration")
        hashes[key] = digest
    return hashes


def _batching_backend(stack: ExitStack, raw_backend: Any, config: dict[str, Any]):
    return stack.enter_context(
        ContinuousBatchingBackend(
            raw_backend,
            max_batch_size=int(config["runtime"]["max_batch_size"]),
            max_batch_tokens=int(config["runtime"]["max_batch_tokens"]),
            batch_wait_seconds=0.01,
        )
    )


def _run_chunk(
    *,
    method: str,
    chunk_index: int,
    task_keys: Sequence[tuple[int, int]],
    problems_by_index: dict[int, Any],
    prompts_by_index: dict[int, tuple[int, ...]],
    raw_base: Any,
    raw_proposal: Any | None,
    config: dict[str, Any],
    workers: int,
    fingerprint: str,
) -> dict[str, Any]:
    if method.endswith("small_proposal") != (raw_proposal is not None):
        raise ValueError("proposal backend presence does not match the IS method")

    base_before = raw_base.snapshot()
    proposal_before = raw_proposal.snapshot() if raw_proposal is not None else None
    with ExitStack() as stack:
        base_batching = _batching_backend(stack, raw_base, config)
        proposal_batching = (
            _batching_backend(stack, raw_proposal, config)
            if raw_proposal is not None
            else None
        )
        base = _MethodBackend(base_batching, raw_base)
        proposal = (
            _MethodBackend(proposal_batching, raw_proposal)
            if proposal_batching is not None and raw_proposal is not None
            else None
        )

        def run_one(task_key: tuple[int, int]):
            draw, problem_index = task_key
            seeds = SeedStream(
                SeedStream(int(config["run"]["seed"])).derive("draw", draw)
            )
            return _run_method(
                method,
                base,
                problems_by_index[problem_index],
                prompts_by_index[problem_index],
                config,
                seeds,
                proposal,
            )

        def run_parallel():
            with ThreadPoolExecutor(
                max_workers=min(workers, len(task_keys))
            ) as executor:
                return list(executor.map(run_one, task_keys))

        outputs, elapsed = _timed(run_parallel)
        base_batching_snapshot = asdict(base_batching.snapshot())
        proposal_batching_snapshot = (
            asdict(proposal_batching.snapshot())
            if proposal_batching is not None
            else None
        )

    base_after = raw_base.snapshot()
    proposal_after = raw_proposal.snapshot() if raw_proposal is not None else None
    base_delta = _snapshot_delta(base_before, base_after)
    proposal_delta = (
        _snapshot_delta(proposal_before, proposal_after)
        if proposal_before is not None and proposal_after is not None
        else None
    )

    records = []
    for (draw, problem_index), (tokens, diagnostics) in zip(
        task_keys, outputs, strict=True
    ):
        problem = problems_by_index[problem_index]
        text = raw_base.decode(tokens)
        prediction = extract_numeric_answer(text)
        records.append(
            {
                "draw_index": draw,
                "problem_index": problem_index,
                "question_sha256": hashlib.sha256(
                    problem.question.encode("utf-8")
                ).hexdigest(),
                "prediction": str(prediction) if prediction is not None else None,
                "correct": prediction == problem.gold_answer,
                "output_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "output_tokens": len(tokens),
                "diagnostics": diagnostics,
            }
        )

    return {
        "schema_version": 1,
        "manifest_fingerprint": fingerprint,
        "method": method,
        "chunk_index": chunk_index,
        "records": records,
        "seconds_excluding_model_load": elapsed,
        "backend_delta": _combine_numeric_deltas(base_delta, proposal_delta),
        "base_backend_delta": base_delta,
        "proposal_backend_delta": proposal_delta,
        "continuous_batching": _combine_batching_snapshots(
            base_batching_snapshot, proposal_batching_snapshot
        ),
        "continuous_batching_by_model": {
            "base": base_batching_snapshot,
            "proposal": proposal_batching_snapshot,
        },
    }


def _run_pending_chunks(
    *,
    raw_path: Path,
    fingerprint: str,
    plan: dict[tuple[str, int], tuple[tuple[int, int], ...]],
    completed: dict[tuple[str, int], dict[str, Any]],
    methods: Sequence[str],
    problems_by_index: dict[int, Any],
    config: dict[str, Any],
    workers: int,
) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    for method in methods:
        pending = [key for key in plan if key[0] == method and key not in completed]
        if not pending:
            continue
        raw_base = _load_backend(str(config["models"]["base"]), config)
        raw_proposal = (
            _load_backend(str(config["models"]["proposal"]), config)
            if method.endswith("small_proposal")
            else None
        )
        if (
            raw_proposal is not None
            and raw_base.tokenizer.get_vocab() != raw_proposal.tokenizer.get_vocab()
        ):
            raise ValueError("base and proposal tokenizers do not have identical vocabularies")
        prompts_by_index = {
            index: _prompt_tokens(raw_base, problem)
            for index, problem in problems_by_index.items()
        }
        if raw_proposal is not None:
            proposal_prompts = {
                index: _prompt_tokens(raw_proposal, problem)
                for index, problem in problems_by_index.items()
            }
            if proposal_prompts != prompts_by_index:
                raise ValueError("base and proposal tokenizers render different prompts")
        first_prompt = prompts_by_index[next(iter(problems_by_index))]
        _sample_one(
            raw_base,
            first_prompt,
            max_new_tokens=2,
            temperature=1.0,
            seed=int(config["run"]["seed"]),
            request_id=f"is-passk-warmup:{method}:base",
        )
        if raw_proposal is not None:
            _sample_one(
                raw_proposal,
                first_prompt,
                max_new_tokens=2,
                temperature=1.0,
                seed=int(config["run"]["seed"]),
                request_id=f"is-passk-warmup:{method}:proposal",
            )

        with raw_path.open("a", encoding="utf-8", buffering=1) as sink:
            for key in pending:
                chunk = _run_chunk(
                    method=method,
                    chunk_index=key[1],
                    task_keys=plan[key],
                    problems_by_index=problems_by_index,
                    prompts_by_index=prompts_by_index,
                    raw_base=raw_base,
                    raw_proposal=raw_proposal,
                    config=config,
                    workers=workers,
                    fingerprint=fingerprint,
                )
                sink.write(json.dumps(chunk, ensure_ascii=False, sort_keys=True) + "\n")
                completed[key] = chunk
                print(
                    f"[{len(completed)}/{len(plan)}] method={method} "
                    f"chunk={key[1]} tasks={len(plan[key])} "
                    f"seconds={chunk['seconds_excluding_model_load']:.3f}",
                    flush=True,
                )

        del raw_proposal
        del raw_base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _paired_pass_at_k_comparison(
    standard: dict[str, Any],
    small: dict[str, Any],
    *,
    draws: int,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    standard_by_problem = {
        int(item["problem_index"]): int(item["correct_draws"])
        for item in standard["per_problem"]
    }
    small_by_problem = {
        int(item["problem_index"]): int(item["correct_draws"])
        for item in small["per_problem"]
    }
    if standard_by_problem.keys() != small_by_problem.keys():
        raise ValueError("paired IS comparison requires the same problem indices")
    problem_indices = tuple(standard_by_problem)
    rng = random.Random(seed)
    result: dict[str, Any] = {}
    for k_text in standard["estimated_pass_at_k"]:
        k = int(k_text)
        differences = [
            _estimated_pass_at_k(small_by_problem[index], draws, k)
            - _estimated_pass_at_k(standard_by_problem[index], draws, k)
            for index in problem_indices
        ]
        bootstrap = sorted(
            statistics.fmean(
                differences[rng.randrange(len(differences))] for _ in differences
            )
            for _ in range(replicates)
        )
        result[k_text] = {
            "small_proposal_minus_standard": statistics.fmean(differences),
            "paired_problem_bootstrap_95": [
                bootstrap[int(0.025 * replicates)],
                bootstrap[int(0.975 * replicates)],
            ],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_quick.toml"))
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--tag", default="is-passk")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--methods", default=",".join(IS_PASSK_METHODS))
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.draws <= 0 or args.limit <= 0 or args.workers <= 0:
        raise ValueError("draws, limit, and workers must be positive")

    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    unknown = sorted(set(methods) - set(IS_PASSK_METHODS))
    if unknown:
        raise ValueError(f"unsupported IS pass@k methods: {', '.join(unknown)}")
    if len(methods) != len(set(methods)):
        raise ValueError("IS pass@k methods must not contain duplicates")

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    config["run"]["sample_count"] = args.limit
    if str(config["runtime"]["device"]).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    problems = select_problems(
        load_gsm8k(args.data), args.limit, seed=int(config["run"]["subset_seed"])
    )
    problem_indices = [problem.index for problem in problems]
    problems_by_index = {problem.index: problem for problem in problems}
    profile = str(config["run"]["name"])
    output = args.output or Path(f"results/{profile}_{args.tag}.json")
    raw_path = args.raw_output or output.with_suffix(".chunks.jsonl")
    implementation_sha256 = {
        path: _file_sha256(Path(path)) for path in IS_PASSK_IMPLEMENTATION_FILES
    }
    input_weight_sha256 = _input_weight_hashes(config, methods)
    _, fingerprint, _ = _prepare_manifest(
        config=config,
        data_path=args.data,
        methods=methods,
        draws=args.draws,
        workers=args.workers,
        problem_indices=problem_indices,
        input_weight_sha256=input_weight_sha256,
        implementation_sha256=implementation_sha256,
        raw_path=raw_path,
    )
    plan = _chunk_plan(methods, args.draws, problem_indices, args.workers)
    completed = _validate_chunks(_load_jsonl(raw_path), fingerprint, plan)
    if not args.summarize_only:
        _run_pending_chunks(
            raw_path=raw_path,
            fingerprint=fingerprint,
            plan=plan,
            completed=completed,
            methods=methods,
            problems_by_index=problems_by_index,
            config=config,
            workers=args.workers,
        )

    chunks = _load_jsonl(raw_path)
    completed = _validate_chunks(chunks, fingerprint, plan)
    if len(completed) != len(plan):
        raise RuntimeError(f"IS pass@k grid is incomplete: {len(completed)}/{len(plan)} chunks")

    ks = tuple(k for k in (1, 2, 4, 8, 16, 32) if k <= args.draws)
    table: dict[str, dict[str, Any]] = {}
    for method in methods:
        method_chunks = [completed[key] for key in plan if key[0] == method]
        method_records = [
            record for chunk in method_chunks for record in chunk["records"]
        ]
        summary = _summarize_method(
            method_records,
            method_chunks,
            problem_indices,
            args.draws,
            ks,
            bootstrap_seed=SeedStream(int(config["run"]["seed"])).derive(method),
        )
        summary["compute_by_model"] = {
            "base": _summarize_model_compute(method_chunks, "base_backend_delta"),
            "proposal": _summarize_model_compute(
                method_chunks, "proposal_backend_delta"
            ),
        }
        summary["continuous_batching_by_model"] = {
            "base": _summarize_batching_by_model(method_chunks, "base"),
            "proposal": _summarize_batching_by_model(method_chunks, "proposal"),
        }
        table[method] = summary

    comparisons = None
    if set(IS_PASSK_METHODS).issubset(table):
        standard = table["conditional_is"]
        small = table["conditional_is_small_proposal"]
        comparisons = {
            "small_proposal_minus_standard_pass_at_k": _paired_pass_at_k_comparison(
                standard,
                small,
                draws=args.draws,
                seed=SeedStream(int(config["run"]["seed"])).derive("is-passk-paired"),
            ),
            "standard_over_small_proposal_wall_time": (
                float(standard["seconds_excluding_model_load"])
                / float(small["seconds_excluding_model_load"])
            ),
            "standard_over_small_proposal_flops": (
                int(standard["estimated_dense_forward_flops"])
                / int(small["estimated_dense_forward_flops"])
            ),
            "ratio_interpretation": (
                "Each ratio uses standard conditional IS as the numerator and the "
                "small-proposal off-policy variant as the denominator. A value above "
                "one means the small-proposal variant used less wall time or FLOPs."
            ),
        }

    report = {
        "schema_version": 1,
        "benchmark": "OpenAI GSM8K official test split",
        "profile": profile,
        "methods": table,
        "comparisons": comparisons,
        "problem_indices": problem_indices,
        "draws_per_problem": args.draws,
        "workers": args.workers,
        "manifest_fingerprint": fingerprint,
        "raw_chunks_sha256": _file_sha256(raw_path),
        "input_weight_sha256": input_weight_sha256,
        "implementation_sha256": implementation_sha256,
        "compute_definition": (
            "2 * each model's parameter count * that model's observed padded forward "
            "token slots, summed across base and proposal models; model loading and "
            "warmup are excluded from wall time"
        ),
        "independence_definition": (
            "Each draw has a separate deterministic random stream. Continuous batching "
            "merges physical model calls but does not share candidate or rollout samples "
            "between draws."
        ),
        "limitations": (
            "Eight draws estimate pass@k only through k=8. The standard and small-proposal "
            "methods use the same problem grid and candidate/rollout budgets, but finite "
            "importance-weight variance and clipped off-policy corrections can change "
            "their output distributions."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
