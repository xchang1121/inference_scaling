"""Run resumable, continuously batched GSM8K pass@k replicates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import statistics
import tomllib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

if __package__:
    from experiments.gsm8k_reproduction import (
        IMPLEMENTATION_FILES,
        _file_sha256,
        _load_backend,
        _prompt_tokens,
        _run_method,
        _sample_one,
        _snapshot_delta,
        _timed,
        _trim_eos,
    )
else:
    from gsm8k_reproduction import (
        IMPLEMENTATION_FILES,
        _file_sha256,
        _load_backend,
        _prompt_tokens,
        _run_method,
        _sample_one,
        _snapshot_delta,
        _timed,
        _trim_eos,
    )
from inference_scaling.algorithms import run_mh_chains_batched
from inference_scaling.backends import (
    BACKEND_CHOICES,
    AbsorbingEOSBackend,
    ContinuousBatchingBackend,
    ScoreCachingBackend,
    close_backend,
    set_backend_override,
)
from inference_scaling.config import MHConfig, SamplingConfig
from inference_scaling.evaluation import (
    extract_numeric_answer,
    load_gsm8k,
    select_problems,
)
from inference_scaling.rng import SeedStream

PASSK_METHODS = ("base", "mh", "rl_sample")
PASSK_IMPLEMENTATION_FILES = (
    *IMPLEMENTATION_FILES,
    "experiments/gsm8k_passk.py",
    "src/inference_scaling/backends/batching.py",
)


class _MethodBackend:
    """Expose model metadata while routing inference through the batch dispatcher."""

    def __init__(self, batching: ContinuousBatchingBackend, raw_backend: Any) -> None:
        self._batching = batching
        self._raw_backend = raw_backend
        self.tokenizer = raw_backend.tokenizer
        self.parameter_count = raw_backend.parameter_count

    @property
    def model_id(self) -> str:
        return self._batching.model_id

    def sample_batch(self, requests):
        return self._batching.sample_batch(requests)

    def score_batch(self, requests):
        return self._batching.score_batch(requests)

    def decode(self, tokens) -> str:
        return self._raw_backend.decode(tokens)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return records


def _estimated_pass_at_k(correct: int, draws: int, k: int) -> float:
    if draws <= 0 or not 1 <= k <= draws:
        raise ValueError("k must lie between one and the number of draws")
    if draws - correct < k:
        return 1.0
    return 1.0 - math.comb(draws - correct, k) / math.comb(draws, k)


def _chunk_plan(
    methods: Sequence[str],
    draws: int,
    problem_indices: Sequence[int],
    chunk_size: int,
) -> dict[tuple[str, int], tuple[tuple[int, int], ...]]:
    if draws <= 0 or chunk_size <= 0 or not problem_indices:
        raise ValueError("draws, chunk size, and problem indices must be non-empty")
    plan: dict[tuple[str, int], tuple[tuple[int, int], ...]] = {}
    for method in methods:
        chunk_index = 0
        for problem_index in problem_indices:
            tasks = [(draw, problem_index) for draw in range(draws)]
            for offset in range(0, len(tasks), chunk_size):
                plan[(method, chunk_index)] = tuple(
                    tasks[offset : offset + chunk_size]
                )
                chunk_index += 1
    return plan


def _prepare_manifest(
    *,
    config: dict[str, Any],
    data_path: Path,
    methods: Sequence[str],
    draws: int,
    workers: int,
    problem_indices: Sequence[int],
    input_weight_sha256: dict[str, str],
    implementation_sha256: dict[str, str],
    raw_path: Path,
) -> tuple[dict[str, Any], str, Path]:
    effective = {
        "config": config,
        "data_path": str(data_path),
        "data_sha256": _file_sha256(data_path),
        "methods": list(methods),
        "draws": draws,
        "workers": workers,
        "problem_indices": list(problem_indices),
        "input_weight_sha256": input_weight_sha256,
        "implementation_sha256": implementation_sha256,
    }
    fingerprint = _fingerprint(effective)
    manifest = {"schema_version": 1, "fingerprint": fingerprint, "effective": effective}
    manifest_path = raw_path.with_suffix(".manifest.json")
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError(
                f"{raw_path} belongs to a different pass@k grid; choose a new output"
            )
    elif raw_path.is_file():
        raise ValueError("pass@k chunks exist without their manifest")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return manifest, fingerprint, manifest_path


def _validate_chunks(
    chunks: Sequence[dict[str, Any]],
    fingerprint: str,
    plan: dict[tuple[str, int], tuple[tuple[int, int], ...]],
) -> dict[tuple[str, int], dict[str, Any]]:
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for chunk in chunks:
        if chunk.get("manifest_fingerprint") != fingerprint:
            raise ValueError("pass@k chunk has the wrong manifest fingerprint")
        key = (str(chunk.get("method")), int(chunk.get("chunk_index", -1)))
        if key not in plan:
            raise ValueError(f"unexpected pass@k chunk: {key}")
        if key in completed:
            raise ValueError(f"duplicate pass@k chunk: {key}")
        task_keys = tuple(
            (int(record["draw_index"]), int(record["problem_index"]))
            for record in chunk.get("records", [])
        )
        if task_keys != plan[key]:
            raise ValueError(f"pass@k chunk {key} contains the wrong task grid")
        completed[key] = chunk
    return completed


def _input_weight_hashes(config: dict[str, Any], methods: Sequence[str]) -> dict[str, str]:
    base_path = Path(str(config["models"]["base"])) / "model.safetensors"
    base_hash = _file_sha256(base_path)
    if base_hash != str(config["models"]["base_weight_sha256"]):
        raise ValueError("base model weight hash does not match the pinned configuration")
    hashes = {"base": base_hash}
    if "rl_sample" in methods:
        adapter_path = (
            Path(str(config["models"]["rl"])) / "adapter_model.safetensors"
        )
        hashes["rl_adapter"] = _file_sha256(adapter_path)
    return hashes


def _run_chunk(
    *,
    method: str,
    chunk_index: int,
    task_keys: Sequence[tuple[int, int]],
    problems_by_index: dict[int, Any],
    prompts_by_index: dict[int, tuple[int, ...]],
    raw_backend: Any,
    config: dict[str, Any],
    workers: int,
    fingerprint: str,
) -> dict[str, Any]:
    before = raw_backend.snapshot()
    with ContinuousBatchingBackend(
        raw_backend,
        max_batch_size=int(config["runtime"]["max_batch_size"]),
        max_batch_tokens=int(config["runtime"]["max_batch_tokens"]),
        batch_wait_seconds=0.01,
    ) as batching:
        backend = _MethodBackend(batching, raw_backend)

        def run_one(task_key: tuple[int, int]):
            draw, problem_index = task_key
            problem = problems_by_index[problem_index]
            seeds = SeedStream(
                SeedStream(int(config["run"]["seed"])).derive("draw", draw)
            )
            tokens, diagnostics = _run_method(
                method,
                backend,
                problem,
                prompts_by_index[problem_index],
                config,
                seeds,
                None,
            )
            return tokens, diagnostics

        def run_parallel():
            if method != "mh":
                with ThreadPoolExecutor(
                    max_workers=min(workers, len(task_keys))
                ) as executor:
                    return list(executor.map(run_one, task_keys))

            problem_indices = {problem_index for _, problem_index in task_keys}
            if len(problem_indices) != 1:
                raise ValueError("a vectorized MH chunk must contain one problem")
            problem_index = next(iter(problem_indices))
            prompt = prompts_by_index[problem_index]
            section = config["mh"]
            alpha = float(section["alpha"])
            absorbing = AbsorbingEOSBackend(
                ScoreCachingBackend(backend),
                raw_backend.tokenizer.eos_token_id,
                absorbing_after=len(prompt),
            )
            chain_seeds = tuple(
                SeedStream(
                    SeedStream(int(config["run"]["seed"])).derive("draw", draw)
                )
                for draw, _ in task_keys
            )
            chain_seeds = tuple(
                SeedStream(stream.derive("mh", problem_index))
                for stream in chain_seeds
            )
            results = run_mh_chains_batched(
                absorbing,
                prompt,
                MHConfig(
                    alpha=alpha,
                    total_length=int(config["generation"]["max_new_tokens"]),
                    block_size=int(section["block_size"]),
                    steps_per_block=int(section["steps_per_block"]),
                ),
                SamplingConfig(temperature=1.0 / alpha),
                chain_seeds,
            )
            return [
                (
                    _trim_eos(result.token_ids, raw_backend.tokenizer.eos_token_id),
                    {
                        "alpha": alpha,
                        "block_size": int(section["block_size"]),
                        "steps_per_block": int(section["steps_per_block"]),
                        "attempts": result.attempts,
                        "accepted": result.accepted,
                        "acceptance_rate": result.acceptance_rate,
                        "execution": "lockstep_vectorized_independent_chains",
                    },
                )
                for result in results
            ]

        outputs, elapsed = _timed(run_parallel)
        batching_snapshot = asdict(batching.snapshot())
    after = raw_backend.snapshot()

    records = []
    for (draw, problem_index), (tokens, diagnostics) in zip(
        task_keys, outputs, strict=True
    ):
        problem = problems_by_index[problem_index]
        text = raw_backend.decode(tokens)
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
        "backend_delta": _snapshot_delta(before, after),
        "continuous_batching": batching_snapshot,
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
        model_key = "rl" if method == "rl_sample" else "base"
        adapter_base = None
        if model_key == "rl" and config["models"].get("rl_kind") == "peft_adapter":
            adapter_base = str(config["models"]["rl_base"])
        raw_backend = _load_backend(
            str(config["models"][model_key]), config, adapter_base=adapter_base
        )
        prompts_by_index = {
            index: _prompt_tokens(raw_backend, problem)
            for index, problem in problems_by_index.items()
        }
        first_prompt = prompts_by_index[next(iter(problems_by_index))]
        _sample_one(
            raw_backend,
            first_prompt,
            max_new_tokens=2,
            temperature=1.0,
            seed=int(config["run"]["seed"]),
            request_id=f"passk-warmup:{method}",
        )
        with raw_path.open("a", encoding="utf-8", buffering=1) as sink:
            for key in pending:
                chunk = _run_chunk(
                    method=method,
                    chunk_index=key[1],
                    task_keys=plan[key],
                    problems_by_index=problems_by_index,
                    prompts_by_index=prompts_by_index,
                    raw_backend=raw_backend,
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
        close_backend(raw_backend)
        del raw_backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _bootstrap_pass_at_k(
    per_problem: Sequence[dict[str, Any]],
    k: int,
    draws: int,
    *,
    seed: int,
    replicates: int = 10_000,
) -> tuple[float, float]:
    rng = random.Random(seed)
    values = [
        _estimated_pass_at_k(int(item["correct_draws"]), draws, k)
        for item in per_problem
    ]
    estimates = sorted(
        statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        for _ in range(replicates)
    )
    return estimates[int(0.025 * replicates)], estimates[int(0.975 * replicates)]


def _summarize_method(
    records: Sequence[dict[str, Any]],
    chunks: Sequence[dict[str, Any]],
    problem_indices: Sequence[int],
    draws: int,
    ks: Sequence[int],
    *,
    bootstrap_seed: int,
) -> dict[str, Any]:
    by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        key = (int(record["draw_index"]), int(record["problem_index"]))
        if key in by_key:
            raise ValueError(f"duplicate pass@k sample: {key}")
        by_key[key] = record
    expected = {
        (draw, problem_index)
        for draw in range(draws)
        for problem_index in problem_indices
    }
    if set(by_key) != expected:
        raise ValueError("pass@k samples do not cover the requested draw/problem grid")

    per_problem = []
    for problem_index in problem_indices:
        problem_records = [by_key[(draw, problem_index)] for draw in range(draws)]
        parsed = {
            record["prediction"]
            for record in problem_records
            if record["prediction"] is not None
        }
        per_problem.append(
            {
                "problem_index": problem_index,
                "correct_draws": sum(bool(record["correct"]) for record in problem_records),
                "unique_parsed_answers": len(parsed),
                "unique_full_outputs": len(
                    {record["output_sha256"] for record in problem_records}
                ),
                "unparseable_draws": sum(
                    record["prediction"] is None for record in problem_records
                ),
            }
        )

    pass_at_k = {}
    pass_at_k_bootstrap = {}
    for k in ks:
        values = [
            _estimated_pass_at_k(int(item["correct_draws"]), draws, k)
            for item in per_problem
        ]
        pass_at_k[str(k)] = statistics.fmean(values)
        pass_at_k_bootstrap[str(k)] = _bootstrap_pass_at_k(
            per_problem,
            k,
            draws,
            seed=SeedStream(bootstrap_seed).derive("pass-at-k", k),
        )

    total_samples = len(problem_indices) * draws
    base_slots = sum(
        int(chunk["backend_delta"]["generation_forward_token_slots"])
        + int(chunk["backend_delta"]["score_forward_token_slots"])
        for chunk in chunks
    )
    total_flops = sum(
        int(chunk["backend_delta"]["estimated_dense_forward_flops"])
        for chunk in chunks
    )
    seconds = sum(float(chunk["seconds_excluding_model_load"]) for chunk in chunks)
    return {
        "examples": len(problem_indices),
        "draws_per_example": draws,
        "generated_answers": total_samples,
        "single_draw_accuracy": sum(
            bool(record["correct"]) for record in records
        )
        / total_samples,
        "estimated_pass_at_k": pass_at_k,
        "estimated_pass_at_k_problem_bootstrap_95": pass_at_k_bootstrap,
        "mean_unique_parsed_answers_across_all_draws": statistics.fmean(
            int(item["unique_parsed_answers"]) for item in per_problem
        ),
        "mean_unique_full_outputs_across_all_draws": statistics.fmean(
            int(item["unique_full_outputs"]) for item in per_problem
        ),
        "unparseable_fraction": sum(
            int(item["unparseable_draws"]) for item in per_problem
        )
        / total_samples,
        "total_forward_token_slots": base_slots,
        "estimated_dense_forward_flops": total_flops,
        "estimated_dense_forward_petaflops": total_flops / 1e15,
        "estimated_dense_flops_per_generated_answer": total_flops / total_samples,
        "seconds_excluding_model_load": seconds,
        "seconds_per_generated_answer": seconds / total_samples,
        "continuous_batching": {
            "sample_batches": sum(
                int(chunk["continuous_batching"]["sample_batches"])
                for chunk in chunks
            ),
            "score_batches": sum(
                int(chunk["continuous_batching"]["score_batches"])
                for chunk in chunks
            ),
            "sample_requests": sum(
                int(chunk["continuous_batching"]["sample_requests"])
                for chunk in chunks
            ),
            "score_sequences": sum(
                int(chunk["continuous_batching"]["score_sequences"])
                for chunk in chunks
            ),
            "maximum_sample_batch": max(
                int(chunk["continuous_batching"]["maximum_sample_batch"])
                for chunk in chunks
            ),
            "maximum_score_batch": max(
                int(chunk["continuous_batching"]["maximum_score_batch"])
                for chunk in chunks
            ),
        },
        "per_problem": per_problem,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gsm8k_quick.toml"))
    parser.add_argument("--backend", choices=BACKEND_CHOICES)
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument("--tag", default="passk")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--methods", default=",".join(PASSK_METHODS))
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--raw-output", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.draws <= 0 or args.limit <= 0 or args.workers <= 0:
        raise ValueError("draws, limit, and workers must be positive")

    methods = tuple(value.strip() for value in args.methods.split(",") if value.strip())
    unknown = sorted(set(methods) - set(PASSK_METHODS))
    if unknown:
        raise ValueError(f"unsupported pass@k methods: {', '.join(unknown)}")
    if len(methods) != len(set(methods)):
        raise ValueError("pass@k methods must not contain duplicates")

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    set_backend_override(config, args.backend)
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
        path: _file_sha256(Path(path)) for path in PASSK_IMPLEMENTATION_FILES
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
        raise RuntimeError(
            f"pass@k grid is incomplete: {len(completed)}/{len(plan)} chunks"
        )

    ks = tuple(k for k in (1, 2, 4, 8, 16, 32) if k <= args.draws)
    table = {}
    for method in methods:
        method_chunks = [
            completed[key] for key in plan if key[0] == method
        ]
        method_records = [
            record for chunk in method_chunks for record in chunk["records"]
        ]
        table[method] = _summarize_method(
            method_records,
            method_chunks,
            problem_indices,
            args.draws,
            ks,
            bootstrap_seed=SeedStream(int(config["run"]["seed"])).derive(method),
        )

    report = {
        "schema_version": 2,
        "benchmark": "OpenAI GSM8K official test split",
        "profile": profile,
        "methods": table,
        "problem_indices": problem_indices,
        "draws_per_problem": args.draws,
        "workers": args.workers,
        "manifest_fingerprint": fingerprint,
        "raw_chunks_sha256": _file_sha256(raw_path),
        "input_weight_sha256": input_weight_sha256,
        "implementation_sha256": implementation_sha256,
        "pass_at_k_definition": (
            "for n independent draws with c correct answers, each problem contributes "
            "1 - choose(n-c,k)/choose(n,k); the report averages this estimator over "
            "the same fixed public rows"
        ),
        "diversity_scope": (
            "pass@k is the primary diversity diagnostic; distinct parsed numeric "
            "answers and full-output hashes are supplemental and do not identify "
            "semantic reasoning-path diversity"
        ),
        "compute_definition": (
            "2 * model parameter count * observed padded forward token slots; each "
            "method uses the same continuous-batching worker count, and model loading "
            "and warmup are excluded from wall time"
        ),
        "limitations": (
            "Eight draws estimate pass@k only through k=8. Problem bootstrap intervals "
            "do not include model, prompt, hyperparameter, or finite-chain mixing "
            "uncertainty. Batched CUDA execution can differ numerically from sequential "
            "execution even with request-local seeds."
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
