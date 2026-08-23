"""Measure the Qwen2.5-1.5B multiscale-MH and frozen-replay stack."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys
import time
import tomllib
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.arllm.benchmark_is_mh_reuse import (
    _BackendFactory,
    _mh_replay_arm,
    _prompt_tokens,
)
from experiments.arllm.runtime import source_hashes, validate_model_artifacts
from inference_scaling.arllm.config import RewardMHConfig
from inference_scaling.shared.evaluation import load_gsm8k, select_problems
from inference_scaling.shared.rng import SeedStream


IMPLEMENTATION_FILES = (
    "experiments/arllm/run_qwen15b_mh_stack.py",
    "experiments/arllm/benchmark_is_mh_reuse.py",
    "src/inference_scaling/arllm/algorithms/mh.py",
    "src/inference_scaling/arllm/algorithms/mh_acceleration.py",
)
ARMS = (
    ("base_uniform", "uniform", False),
    ("base_multiscale", "multiscale", False),
    ("replay_uniform", "uniform", True),
    ("replay_multiscale", "multiscale", True),
)


def _atomic_write(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _online_wall(arm: Mapping[str, Any]) -> float:
    return float(arm["online"]["telemetry"]["wall_seconds"])


def _online_flops(arm: Mapping[str, Any]) -> float:
    return float(arm["online"]["main_model"]["estimated_dense_forward_flops"])


def _cache_wall(arm: Mapping[str, Any]) -> float:
    return float(arm["cache_build"]["telemetry"]["wall_seconds"])


def _cache_flops(arm: Mapping[str, Any]) -> float:
    return float(
        arm["cache_build"]["main_model"].get("estimated_dense_forward_flops", 0.0)
    )


def _mean_std(values: Sequence[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    mean = sum(numbers) / len(numbers)
    variance = (
        sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1)
        if len(numbers) > 1
        else 0.0
    )
    return {"mean": mean, "sample_std": math.sqrt(variance), "runs": len(numbers)}


def _ratio_by_seed(
    runs: Sequence[Mapping[str, Any]],
    numerator: str,
    denominator: str,
    metric,
) -> dict[str, float | int]:
    ratios = []
    for run in runs:
        arms = {arm["name"]: arm for arm in run["arms"]}
        ratios.append(metric(arms[numerator]) / metric(arms[denominator]))
    return _mean_std(ratios)


def _break_even_queries(
    baseline_online: float,
    optimized_online: float,
    cache_build: float,
) -> int | None:
    saving = baseline_online - optimized_online
    if saving <= 0:
        return None
    return max(1, math.floor(cache_build / saving) + 1)


def summarize(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {"complete": False, "runs": 0}
    arm_names = tuple(name for name, _schedule, _replay in ARMS)
    expected = set(arm_names)
    for run in runs:
        names = [str(arm["name"]) for arm in run["arms"]]
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("a stack run has missing or duplicate arms")
    aggregates: dict[str, Any] = {}
    for name in arm_names:
        selected = [
            next(arm for arm in run["arms"] if arm["name"] == name) for run in runs
        ]
        proposals = [
            arm["online"].get("proposal_sources", {"base": 0, "history": 0})
            for arm in selected
        ]
        aggregates[name] = {
            "online_wall_seconds": _mean_std([_online_wall(arm) for arm in selected]),
            "online_pflops": _mean_std([_online_flops(arm) / 1e15 for arm in selected]),
            "cache_build_seconds": _mean_std([_cache_wall(arm) for arm in selected]),
            "cache_build_pflops": _mean_std(
                [_cache_flops(arm) / 1e15 for arm in selected]
            ),
            "acceptance_rate": _mean_std(
                [float(arm["online"]["acceptance_rate"]) for arm in selected]
            ),
            "history_proposal_fraction": _mean_std(
                [
                    float(counts.get("history", 0)) / sum(counts.values())
                    if sum(counts.values())
                    else 0.0
                    for counts in proposals
                ]
            ),
        }
    comparisons = {
        "multiscale_over_uniform_without_replay": {
            "wall_factor": _ratio_by_seed(
                runs, "base_multiscale", "base_uniform", _online_wall
            ),
            "flops_factor": _ratio_by_seed(
                runs, "base_multiscale", "base_uniform", _online_flops
            ),
        },
        "replay_over_base_with_uniform_schedule": {
            "wall_factor": _ratio_by_seed(
                runs, "replay_uniform", "base_uniform", _online_wall
            ),
            "flops_factor": _ratio_by_seed(
                runs, "replay_uniform", "base_uniform", _online_flops
            ),
        },
        "stack_over_base_uniform": {
            "wall_factor": _ratio_by_seed(
                runs, "replay_multiscale", "base_uniform", _online_wall
            ),
            "flops_factor": _ratio_by_seed(
                runs, "replay_multiscale", "base_uniform", _online_flops
            ),
        },
        "stack_over_multiscale_without_replay": {
            "wall_factor": _ratio_by_seed(
                runs, "replay_multiscale", "base_multiscale", _online_wall
            ),
            "flops_factor": _ratio_by_seed(
                runs, "replay_multiscale", "base_multiscale", _online_flops
            ),
        },
        "stack_over_uniform_replay": {
            "wall_factor": _ratio_by_seed(
                runs, "replay_multiscale", "replay_uniform", _online_wall
            ),
            "flops_factor": _ratio_by_seed(
                runs, "replay_multiscale", "replay_uniform", _online_flops
            ),
        },
    }
    break_even = {"wall_queries": [], "flops_queries": []}
    for run in runs:
        arms = {arm["name"]: arm for arm in run["arms"]}
        baseline = arms["base_uniform"]
        stack = arms["replay_multiscale"]
        break_even["wall_queries"].append(
            _break_even_queries(
                _online_wall(baseline),
                _online_wall(stack),
                _cache_wall(stack),
            )
        )
        break_even["flops_queries"].append(
            _break_even_queries(
                _online_flops(baseline),
                _online_flops(stack),
                _cache_flops(stack),
            )
        )
    stack_wall = float(comparisons["stack_over_base_uniform"]["wall_factor"]["mean"])
    component_wall = max(
        float(
            comparisons["stack_over_multiscale_without_replay"]["wall_factor"]["mean"]
        ),
        float(comparisons["stack_over_uniform_replay"]["wall_factor"]["mean"]),
    )
    return {
        "complete": True,
        "runs": len(runs),
        "arms": aggregates,
        "comparisons": comparisons,
        "break_even_queries_by_seed": break_even,
        "decision": {
            "status": (
                "accepted"
                if stack_wall <= 0.95 and component_wall <= 1.05
                else "rejected"
            ),
            "criterion": (
                "the combined online wall factor versus uniform base MH must be at "
                "most 0.95, and adding either accepted component to the other may not "
                "increase mean wall time by more than 5%"
            ),
        },
    }


def _compact_arm(arm: dict[str, Any]) -> dict[str, Any]:
    compact = copy.deepcopy(arm)
    compact["online"].pop("token_ids", None)
    compact["online"].pop("traces", None)
    return compact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gsm8k_3090_aligned.toml"),
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
            "results/arllm/qwen15b_optimization/mh_replay_multiscale_stack.json"
        ),
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(20260812, 20260813, 20260814)
    )
    parser.add_argument("--total-length", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--history-rollouts", type=int, default=8)
    parser.add_argument("--reward-temperature", type=float, default=0.3)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if (
        min(
            args.total_length,
            args.steps,
            args.chains,
            args.history_rollouts,
            *args.seeds,
        )
        <= 0
    ):
        raise ValueError("budgets and seeds must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    with args.config.open("rb") as source:
        config = tomllib.load(source)
    config.setdefault("runtime", {})["backend"] = "transformers"
    config["runtime"]["dtype"] = args.dtype
    setting = {
        "model": "Qwen2.5-1.5B-Instruct",
        "dataset": "pinned OpenAI GSM8K test split",
        "problem_count": 1,
        "dtype": args.dtype,
        "seeds": list(args.seeds),
        "total_length": args.total_length,
        "steps_per_chain": args.steps,
        "chains": args.chains,
        "history_rollouts": args.history_rollouts,
        "reward_temperature": args.reward_temperature,
        "arms": [name for name, _schedule, _replay in ARMS],
        "dllm_experiments": False,
    }
    if args.dry_run:
        print(json.dumps(setting, ensure_ascii=False, indent=2))
        return
    if args.restart and args.output.exists():
        args.output.unlink()
    implementation = source_hashes(IMPLEMENTATION_FILES)
    payload: dict[str, Any]
    if args.output.is_file():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        if (
            payload["setting"] != setting
            or payload["implementation_sha256"] != implementation
        ):
            raise ValueError(
                "existing result has a different protocol or implementation"
            )
    else:
        payload = {
            "schema_version": 1,
            "study": "qwen15b_mh_multiscale_replay_stack",
            "setting": setting,
            "implementation_sha256": implementation,
            "model_artifacts": validate_model_artifacts(config, ("base",)),
            "runs": [],
            "summary": {"complete": False, "runs": 0},
        }
    completed = {int(run["seed"]) for run in payload["runs"]}
    problem = select_problems(
        load_gsm8k(args.data),
        1,
        seed=int(config["run"]["subset_seed"]),
    )[0]
    factory = _BackendFactory(config, "transformers", args.dtype)
    try:
        prompt = _prompt_tokens(factory.tokenizer, problem.question)
        for seed_position, seed in enumerate(args.seeds):
            if seed in completed:
                continue
            arm_results = []
            ordered = (
                ARMS[seed_position % len(ARMS) :] + ARMS[: seed_position % len(ARMS)]
            )
            for name, schedule, replay in ordered:
                config_for_arm = RewardMHConfig(
                    total_length=args.total_length,
                    block_size=args.total_length,
                    steps_per_block=args.steps,
                    reward_temperature=args.reward_temperature,
                    suffix_schedule=schedule,
                )
                arm = _mh_replay_arm(
                    factory,
                    name=name,
                    prompt=prompt,
                    config=config_for_arm,
                    history_rollouts=args.history_rollouts,
                    seed=SeedStream(seed).derive("mh-stack"),
                    replay=replay,
                    chains=args.chains,
                )
                arm_results.append(_compact_arm(arm))
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "arm": name,
                            "online_wall_seconds": _online_wall(arm),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            arm_results.sort(
                key=lambda arm: [value[0] for value in ARMS].index(arm["name"])
            )
            payload["runs"].append(
                {
                    "seed": seed,
                    "problem_index": problem.index,
                    "arms": arm_results,
                }
            )
            payload["runs"].sort(key=lambda run: int(run["seed"]))
            payload["summary"] = summarize(payload["runs"])
            _atomic_write(payload, args.output)
            completed.add(seed)
    finally:
        factory.close()
    payload["summary"] = summarize(payload["runs"])
    payload["completed_at_unix"] = time.time()
    _atomic_write(payload, args.output)
    print(json.dumps(payload["summary"]["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
