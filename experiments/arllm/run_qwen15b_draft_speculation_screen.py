"""Resumable Qwen2.5-1.5B draft-model speculative-decoding screen."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import tomllib
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch

from experiments.arllm.runtime import source_hashes, validate_model_artifacts
from inference_scaling.arllm.acceleration import DraftModelSpeculationConfig
from inference_scaling.arllm.backends import (
    TransformersBackend,
    close_backend,
    load_backend_from_config,
)
from inference_scaling.experimental.arllm.draft_model_speculation import (
    DraftModelSpeculativeBackend,
)
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, SequenceSample, TokenSequence
from inference_scaling.shared.evaluation import (
    extract_numeric_answer,
    gsm8k_prompt,
    load_gsm8k,
    select_problems,
)
from inference_scaling.shared.rng import SeedStream


IMPLEMENTATION_FILES = (
    "experiments/arllm/run_qwen15b_draft_speculation_screen.py",
    "src/inference_scaling/experimental/arllm/draft_model_speculation.py",
    "src/inference_scaling/arllm/backends/transformers_backend.py",
    "src/inference_scaling/arllm/acceleration.py",
)


def _cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _numeric_snapshot(value: Any) -> dict[str, int | float]:
    raw = asdict(value) if is_dataclass(value) else dict(value)
    return {
        key: number
        for key, number in raw.items()
        if isinstance(number, (int, float)) and not isinstance(number, bool)
    }


def _delta(before: Any, after: Any) -> dict[str, int | float]:
    left = _numeric_snapshot(before)
    right = _numeric_snapshot(after)
    return {key: right[key] - left.get(key, 0) for key in right}


def _prompt_tokens(backend: TransformersBackend, question: str) -> TokenSequence:
    rendered = backend.tokenizer.apply_chat_template(
        [{"role": "user", "content": gsm8k_prompt(question)}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return backend.encode(str(rendered), add_special_tokens=False)


def _atomic_write(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_existing(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"invalid resumable result: {path}")
    return payload


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _arms(draft_tokens: Sequence[int]) -> tuple[str, ...]:
    return ("target_only", *(f"draft_model_k{count}" for count in draft_tokens))


def _record_key(record: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(record["arm"]), int(record["draw"]), int(record["problem_index"])


def _aggregate(records: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    selected = [record for record in records if record["arm"] == arm]
    if not selected:
        return {
            "arm": arm,
            "records": 0,
            "accuracy": None,
            "parseable_fraction": None,
            "wall_seconds": 0.0,
            "mean_wall_seconds": None,
            "output_tokens": 0,
            "output_tokens_per_second": 0.0,
            "main_model_forward_token_slots": 0,
            "draft_model_forward_token_slots": 0,
            "main_model_flops": 0,
            "draft_model_flops": 0,
            "total_flops": 0,
            "draft_tokens_proposed": 0,
            "draft_tokens_accepted": 0,
            "draft_acceptance_rate": 0.0,
            "verification_rounds": 0,
            "maximum_peak_allocated_mib": None,
        }
    wall = sum(float(record["wall_seconds"]) for record in selected)
    output_tokens = sum(int(record["output_tokens"]) for record in selected)
    proposed = sum(int(record["draft_tokens_proposed"]) for record in selected)
    accepted = sum(int(record["draft_tokens_accepted"]) for record in selected)
    main_flops = sum(int(record["main_model_flops"]) for record in selected)
    draft_flops = sum(int(record["draft_model_flops"]) for record in selected)
    return {
        "arm": arm,
        "records": len(selected),
        "accuracy": sum(bool(record["correct"]) for record in selected) / len(selected),
        "parseable_fraction": (
            sum(record["prediction"] is not None for record in selected) / len(selected)
        ),
        "wall_seconds": wall,
        "mean_wall_seconds": wall / len(selected),
        "output_tokens": output_tokens,
        "output_tokens_per_second": output_tokens / wall if wall else 0.0,
        "main_model_forward_token_slots": sum(
            int(record["main_model_forward_token_slots"]) for record in selected
        ),
        "draft_model_forward_token_slots": sum(
            int(record["draft_model_forward_token_slots"]) for record in selected
        ),
        "main_model_flops": main_flops,
        "draft_model_flops": draft_flops,
        "total_flops": main_flops + draft_flops,
        "draft_tokens_proposed": proposed,
        "draft_tokens_accepted": accepted,
        "draft_acceptance_rate": accepted / proposed if proposed else 0.0,
        "verification_rounds": sum(
            int(record["verification_rounds"]) for record in selected
        ),
        "maximum_peak_allocated_mib": max(
            float(record["peak_allocated_mib"]) for record in selected
        ),
    }


def summarize(
    records: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    *,
    expected_records_per_arm: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    aggregates = [_aggregate(records, arm) for arm in arms]
    baseline = aggregates[0]
    if int(baseline["records"]) == 0:
        return aggregates, {"complete": False, "selected_default": None}
    baseline_wall = float(baseline["wall_seconds"])
    baseline_main_flops = int(baseline["main_model_flops"])
    baseline_total_flops = int(baseline["total_flops"])
    for aggregate in aggregates:
        aggregate["wall_ratio_to_target_only"] = (
            float(aggregate["wall_seconds"]) / baseline_wall if baseline_wall else None
        )
        aggregate["main_model_flops_ratio_to_target_only"] = (
            int(aggregate["main_model_flops"]) / baseline_main_flops
            if baseline_main_flops
            else None
        )
        aggregate["total_flops_ratio_to_target_only"] = (
            int(aggregate["total_flops"]) / baseline_total_flops
            if baseline_total_flops
            else None
        )
    complete = all(
        int(aggregate["records"]) == expected_records_per_arm
        for aggregate in aggregates
    )
    passing = [
        aggregate
        for aggregate in aggregates[1:]
        if aggregate["records"] == expected_records_per_arm
        and float(aggregate["wall_ratio_to_target_only"]) <= 0.95
        and int(aggregate["draft_tokens_proposed"]) > 0
    ]
    selected = min(passing, key=lambda value: float(value["wall_seconds"])) if passing else None
    return aggregates, {
        "complete": complete,
        "criterion": (
            "The implementation and unit tests must preserve the target distribution; "
            "the Qwen screen then requires at least 5% aggregate wall-time reduction. "
            "Main-model, draft-model and total FLOPs are reported separately rather "
            "than treated as interchangeable costs."
        ),
        "selected_default": (
            str(selected["arm"]) if complete and selected is not None else "target_only"
        ),
        "status": (
            "accepted" if complete and selected is not None else "rejected" if complete else "running"
        ),
    }


def _measure_one(
    backend: TransformersBackend | DraftModelSpeculativeBackend,
    request: GenerationRequest,
) -> tuple[SequenceSample, float, float, dict[str, int | float]]:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    before = backend.snapshot()
    _cuda_sync()
    started = time.perf_counter()
    sample = backend.sample_batch([request])[0]
    _cuda_sync()
    wall = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
    after = backend.snapshot()
    return sample, wall, peak, _delta(before, after)


def _warmup(
    target: TransformersBackend,
    speculative: Sequence[DraftModelSpeculativeBackend],
    *,
    seed: int,
) -> None:
    prefix = target.encode(
        "Fixed kernel warmup text 91357.",
        add_special_tokens=True,
    )
    sampling = SamplingConfig(eos_token_id=target.tokenizer.eos_token_id)
    target.sample_batch(
        [GenerationRequest(prefix, 4, sampling, seed, "draft-spec:warmup:target")]
    )
    for index, backend in enumerate(speculative):
        backend.sample_batch(
            [
                GenerationRequest(
                    prefix,
                    backend.config.draft_tokens + 2,
                    sampling,
                    seed + index + 1,
                    f"draft-spec:warmup:speculative:{backend.config.draft_tokens}",
                )
            ]
        )
    _cuda_sync()


def _initial_payload(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    arms: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "study": "qwen15b_draft_model_speculative_decoding",
        "phase": args.phase,
        "scope": {
            "primary_model": "Qwen2.5-1.5B-Instruct",
            "auxiliary_model": "Qwen2.5-0.5B-Instruct",
            "auxiliary_role": "draft tokens only",
            "dataset": "pinned OpenAI GSM8K test split",
            "backend": "Transformers native speculative sampling",
            "hardware": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "dllm_experiments": False,
        },
        "protocol": {
            "questions": args.limit,
            "draws": args.draws,
            "maximum_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
            "sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            },
            "arms": list(arms),
            "execution_order": "cyclic rotation over arms for every question and draw",
            "model_residency": (
                "both models remain resident for every measured arm; model loading and "
                "warmup are excluded from online wall time"
            ),
            "distribution_guarantee": (
                "the 0.5B model proposes tokens; the 1.5B target verifies each block "
                "with speculative acceptance and residual correction"
            ),
        },
        "config": str(args.config),
        "model_artifacts": None,
        "implementation_sha256": source_hashes(IMPLEMENTATION_FILES),
        "environment": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "transformers": _package_version("transformers"),
            "cuda": torch.version.cuda,
            "git_revision": _git_revision(),
        },
        "load": None,
        "records": [],
        "summary": [],
        "decision": {"complete": False, "selected_default": None, "status": "running"},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gsm8k_quick.toml"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/gsm8k/test.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/arllm/qwen15b_optimization/draft_model_speculation_screen.json"
        ),
    )
    parser.add_argument("--phase", choices=("smoke", "screen"), default="screen")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--draws", type=int)
    parser.add_argument("--draft-tokens", type=int, nargs="+", default=(2, 4, 8))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is None:
        args.limit = 1 if args.phase == "smoke" else 8
    if args.draws is None:
        args.draws = 1 if args.phase == "smoke" else 2
    if min(args.limit, args.draws, args.max_new_tokens, *args.draft_tokens) <= 0:
        raise ValueError("experiment sizes and draft-token counts must be positive")
    if len(set(args.draft_tokens)) != len(args.draft_tokens):
        raise ValueError("draft-token counts must be unique")
    if args.phase == "screen" and (args.limit != 8 or args.draws != 2):
        raise ValueError("the registered screen requires 8 questions and 2 draws")
    with args.config.open("rb") as source:
        config = tomllib.load(source)
    config.setdefault("runtime", {})["backend"] = "transformers"
    config["runtime"]["dtype"] = args.dtype
    arms = _arms(args.draft_tokens)
    if args.dry_run:
        print(
            json.dumps(
                _initial_payload(args, config, arms)["protocol"],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.restart and args.output.exists():
        args.output.unlink()
    current_payload = _initial_payload(args, config, arms)
    payload = _load_existing(args.output) or current_payload
    expected_protocol = current_payload["protocol"]
    if payload["protocol"] != expected_protocol:
        raise ValueError("existing result uses a different experiment protocol")
    if payload.get("implementation_sha256") != current_payload["implementation_sha256"]:
        raise ValueError(
            "implementation changed since the partial result; rerun with --restart"
        )
    records: list[dict[str, Any]] = list(payload["records"])
    completed = {_record_key(record) for record in records}
    problems = select_problems(
        load_gsm8k(args.data),
        args.limit,
        seed=int(config["run"]["subset_seed"]),
    )
    expected_problem_indices = {problem.index for problem in problems}
    if any(key[2] not in expected_problem_indices for key in completed):
        raise ValueError("existing result contains a different GSM8K subset")
    expected = {
        (arm, draw, problem.index)
        for arm in arms
        for draw in range(args.draws)
        for problem in problems
    }
    if completed == expected:
        payload["summary"], payload["decision"] = summarize(
            records,
            arms,
            expected_records_per_arm=args.limit * args.draws,
        )
        _atomic_write(payload, args.output)
        print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))
        return

    payload["model_artifacts"] = validate_model_artifacts(config, ("base", "proposal"))
    target: TransformersBackend | None = None
    draft: TransformersBackend | None = None
    wrappers: dict[str, DraftModelSpeculativeBackend] = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    try:
        loaded_target = load_backend_from_config(str(config["models"]["base"]), config)
        loaded_draft = load_backend_from_config(str(config["models"]["proposal"]), config)
        if not isinstance(loaded_target, TransformersBackend) or not isinstance(
            loaded_draft, TransformersBackend
        ):
            raise TypeError("draft-model screen requires TransformersBackend")
        target, draft = loaded_target, loaded_draft
        if target.tokenizer.get_vocab() != draft.tokenizer.get_vocab():
            raise ValueError("base and proposal tokenizers must match")
        wrappers = {
            f"draft_model_k{count}": DraftModelSpeculativeBackend(
                target,
                draft,
                config=DraftModelSpeculationConfig(draft_tokens=count),
            )
            for count in args.draft_tokens
        }
        _cuda_sync()
        payload["load"] = {
            "wall_seconds": time.perf_counter() - load_started,
            "peak_allocated_mib": (
                torch.cuda.max_memory_allocated() / 2**20
                if torch.cuda.is_available()
                else 0.0
            ),
            "primary_parameter_count": target.parameter_count,
            "auxiliary_parameter_count": draft.parameter_count,
        }
        _warmup(
            target,
            tuple(wrappers.values()),
            seed=SeedStream(int(config["run"]["seed"])).derive("draft-spec", "warmup"),
        )
        prompts = {
            problem.index: _prompt_tokens(target, problem.question) for problem in problems
        }
        sampling = SamplingConfig(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_token_id=target.tokenizer.eos_token_id,
        )
        seed_stream = SeedStream(int(config["run"]["seed"]))
        for draw in range(args.draws):
            for problem_position, problem in enumerate(problems):
                shift = (draw * len(problems) + problem_position) % len(arms)
                ordered_arms = arms[shift:] + arms[:shift]
                shared_seed = seed_stream.derive(
                    "draft-model-speculation",
                    draw,
                    problem.index,
                )
                for arm in ordered_arms:
                    key = (arm, draw, problem.index)
                    if key in completed:
                        continue
                    backend: TransformersBackend | DraftModelSpeculativeBackend
                    backend = target if arm == "target_only" else wrappers[arm]
                    request = GenerationRequest(
                        prompts[problem.index],
                        args.max_new_tokens,
                        sampling,
                        shared_seed,
                        f"draft-spec:{arm}:draw:{draw}:problem:{problem.index}",
                    )
                    sample, wall, peak, statistics = _measure_one(backend, request)
                    prediction = extract_numeric_answer(target.decode(sample.token_ids))
                    if arm == "target_only":
                        main_slots = int(statistics["generation_forward_token_slots"])
                        main_flops = int(statistics["estimated_dense_forward_flops"])
                        draft_slots = 0
                        draft_flops = 0
                        proposed = 0
                        accepted = 0
                        rounds = 0
                    else:
                        main_slots = int(statistics["generation_forward_token_slots"])
                        main_flops = int(statistics["estimated_dense_forward_flops"])
                        draft_slots = int(statistics["draft_model_forward_token_slots"])
                        draft_flops = int(
                            statistics["draft_model_estimated_dense_forward_flops"]
                        )
                        proposed = int(statistics["draft_tokens_proposed"])
                        accepted = int(statistics["draft_tokens_accepted"])
                        rounds = int(statistics["verification_rounds"])
                    records.append(
                        {
                            "arm": arm,
                            "draw": draw,
                            "problem_index": problem.index,
                            "question_sha256": hashlib.sha256(
                                problem.question.encode("utf-8")
                            ).hexdigest(),
                            "request_seed": shared_seed,
                            "output_token_ids": list(sample.token_ids),
                            "output_tokens": len(sample.token_ids),
                            "prediction": (
                                None if prediction is None else str(prediction)
                            ),
                            "gold_answer": str(problem.gold_answer),
                            "correct": prediction == problem.gold_answer,
                            "finish_reason": sample.finish_reason,
                            "wall_seconds": wall,
                            "peak_allocated_mib": peak,
                            "main_model_forward_token_slots": main_slots,
                            "draft_model_forward_token_slots": draft_slots,
                            "main_model_flops": main_flops,
                            "draft_model_flops": draft_flops,
                            "total_flops": main_flops + draft_flops,
                            "draft_tokens_proposed": proposed,
                            "draft_tokens_accepted": accepted,
                            "verification_rounds": rounds,
                        }
                    )
                    completed.add(key)
                    records.sort(key=lambda value: _record_key(value))
                    payload["records"] = records
                    payload["summary"], payload["decision"] = summarize(
                        records,
                        arms,
                        expected_records_per_arm=args.limit * args.draws,
                    )
                    _atomic_write(payload, args.output)
                    print(
                        json.dumps(
                            {
                                "completed": len(completed),
                                "expected": len(expected),
                                "arm": arm,
                                "draw": draw,
                                "problem_index": problem.index,
                                "wall_seconds": wall,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    finally:
        wrappers.clear()
        close_backend(draft)
        close_backend(target)
        draft = None
        target = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload["summary"], payload["decision"] = summarize(
        records,
        arms,
        expected_records_per_arm=args.limit * args.draws,
    )
    _atomic_write(payload, args.output)
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
