"""Measure the Qwen2.5-1.5B rollout-replay execution stack.

The five arms isolate three accepted or mechanically exact execution changes:
off-policy rollout replay, reuse of the candidate draws required to construct the
replay keys, and cross-request continuous batching.  Qwen2.5-0.5B is used only to
construct or score replay rollouts, and its compute is kept separate from the
Qwen2.5-1.5B account.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
from pathlib import Path
import sys
import time
import tomllib
from typing import Any, Callable, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPOSITORY_ROOT, REPOSITORY_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.arllm.gsm8k_reproduction import (
    _configured_verifier_reward,
    _load_backend,
    _prompt_tokens,
    _snapshot_delta,
    _timed,
)
from experiments.arllm.runtime import source_hashes, validate_model_artifacts
from inference_scaling.arllm.algorithms.base_replay import (
    _score_base,
    base_replay_step,
)
from inference_scaling.arllm.algorithms.conditional_is import _sample_candidates
from inference_scaling.arllm.backends import (
    ContinuousBatchingBackend,
    ScoreCachingBackend,
    close_backend,
)
from inference_scaling.arllm.config import BaseReplayConfig, SamplingConfig
from inference_scaling.arllm.replay import (
    BehaviorPolicy,
    BehaviorRegistry,
    InMemoryReplayStore,
    ReplayKey,
    ReplaySampleRequest,
    sample_replay_records,
    validate_record_probabilities,
)
from inference_scaling.arllm.types import (
    GenerationRequest,
    SequenceSample,
    TokenSequence,
)
from inference_scaling.shared.evaluation import (
    extract_numeric_answer,
    load_gsm8k,
    select_problems,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.verifier import replace_verifier_from_file


IMPLEMENTATION_FILES = (
    "experiments/arllm/run_qwen15b_is_stack.py",
    "experiments/arllm/gsm8k_replay_benchmark.py",
    "src/inference_scaling/arllm/algorithms/base_replay.py",
    "src/inference_scaling/arllm/algorithms/conditional_is.py",
    "src/inference_scaling/arllm/backends/batching.py",
    "src/inference_scaling/arllm/backends/cache.py",
    "src/inference_scaling/arllm/backends/transformers_backend.py",
    "src/inference_scaling/arllm/replay.py",
    "src/inference_scaling/shared/importance.py",
)
@dataclass(frozen=True, slots=True)
class ArmSpec:
    name: str
    replay: bool
    candidate_reuse: bool
    continuous_batching: bool


ARMS = (
    ArmSpec("fresh_sequential", False, False, False),
    ArmSpec("fresh_continuous", False, False, True),
    ArmSpec("replay_sequential", True, False, False),
    ArmSpec("replay_candidate_cache", True, True, False),
    ArmSpec("replay_candidate_cache_continuous", True, True, True),
)


@dataclass(slots=True)
class _QueryState:
    problem_index: int
    prompt: TokenSequence
    seeds: SeedStream
    config: BaseReplayConfig
    sampling: SamplingConfig
    reward: Callable[[TokenSequence, TokenSequence], float]
    reward_version: str
    registry: BehaviorRegistry = field(default_factory=BehaviorRegistry)
    store: InMemoryReplayStore = field(default_factory=InMemoryReplayStore)
    generated: list[int] = field(default_factory=list)
    steps: list[Any] = field(default_factory=list)
    built_candidates: tuple[SequenceSample, ...] | None = None
    candidate_draws_reused: int = 0
    candidates_reproduced: bool = True
    history_generated: int = 0

    @property
    def active(self) -> bool:
        return len(self.generated) < self.config.total_length


def _atomic_write(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _mean_std(values: Sequence[float]) -> dict[str, float | int]:
    numbers = [float(value) for value in values]
    mean = sum(numbers) / len(numbers)
    variance = (
        sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1)
        if len(numbers) > 1
        else 0.0
    )
    return {"mean": mean, "sample_std": math.sqrt(variance), "runs": len(numbers)}


def _empty_delta(backend: Any) -> dict[str, int | float]:
    snapshot = backend.snapshot()
    return _snapshot_delta(snapshot, snapshot)


def _add_delta(
    aggregate: dict[str, int | float],
    delta: Mapping[str, int | float],
) -> None:
    for key, value in delta.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        aggregate[key] = aggregate.get(key, 0) + value


def _phase_payload(
    wall_seconds: float,
    main_model: Mapping[str, int | float],
    auxiliary_model: Mapping[str, int | float],
) -> dict[str, Any]:
    main = dict(main_model)
    auxiliary = dict(auxiliary_model)
    main_flops = int(main.get("estimated_dense_forward_flops", 0))
    auxiliary_flops = int(auxiliary.get("estimated_dense_forward_flops", 0))
    main_slots = int(main.get("generation_forward_token_slots", 0)) + int(
        main.get("score_forward_token_slots", 0)
    )
    auxiliary_slots = int(auxiliary.get("generation_forward_token_slots", 0)) + int(
        auxiliary.get("score_forward_token_slots", 0)
    )
    return {
        "wall_seconds": wall_seconds,
        "main_model": main,
        "auxiliary_model": auxiliary,
        "main_model_forward_token_slots": main_slots,
        "auxiliary_model_forward_token_slots": auxiliary_slots,
        "main_model_estimated_dense_forward_flops": main_flops,
        "auxiliary_model_estimated_dense_forward_flops": auxiliary_flops,
        "total_estimated_dense_forward_flops": main_flops + auxiliary_flops,
    }


def _sum_phases(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "wall_seconds": float(left["wall_seconds"]) + float(right["wall_seconds"]),
        "main_model_forward_token_slots": int(left["main_model_forward_token_slots"])
        + int(right["main_model_forward_token_slots"]),
        "auxiliary_model_forward_token_slots": int(
            left["auxiliary_model_forward_token_slots"]
        )
        + int(right["auxiliary_model_forward_token_slots"]),
        "main_model_estimated_dense_forward_flops": int(
            left["main_model_estimated_dense_forward_flops"]
        )
        + int(right["main_model_estimated_dense_forward_flops"]),
        "auxiliary_model_estimated_dense_forward_flops": int(
            left["auxiliary_model_estimated_dense_forward_flops"]
        )
        + int(right["auxiliary_model_estimated_dense_forward_flops"]),
        "total_estimated_dense_forward_flops": int(
            left["total_estimated_dense_forward_flops"]
        )
        + int(right["total_estimated_dense_forward_flops"]),
    }


def _execute(
    states: Sequence[_QueryState],
    operation: Callable[[_QueryState], Any],
    *,
    concurrent: bool,
    workers: int,
) -> list[Any]:
    if not concurrent:
        return [operation(state) for state in states]
    with ThreadPoolExecutor(max_workers=min(workers, len(states))) as executor:
        futures = [executor.submit(operation, state) for state in states]
        return [future.result() for future in futures]


def _make_states(
    raw_backend: Any,
    problems: Sequence[Any],
    prompts: Sequence[TokenSequence],
    verifier_config: dict[str, Any],
    *,
    root_seed: int,
    candidate_count: int,
    block_size: int,
    total_length: int,
    reward_temperature: float,
    history_rollouts: int,
    fresh_rollouts: int,
    truncation: float,
    replay: bool,
    sampling: SamplingConfig,
) -> list[_QueryState]:
    algorithm = BaseReplayConfig(
        candidate_count=candidate_count,
        block_size=block_size,
        total_length=total_length,
        reward_temperature=reward_temperature,
        max_history_per_candidate=(history_rollouts if replay else 0),
        fresh_rollouts=(
            fresh_rollouts if replay else history_rollouts + fresh_rollouts
        ),
        truncation=truncation,
    )
    states: list[_QueryState] = []
    for problem, prompt in zip(problems, prompts, strict=True):
        reward = _configured_verifier_reward(raw_backend, problem, verifier_config)
        states.append(
            _QueryState(
                problem_index=problem.index,
                prompt=prompt,
                seeds=SeedStream(
                    SeedStream(root_seed).derive("qwen15b-is-stack", problem.index)
                ),
                config=algorithm,
                sampling=sampling,
                reward=reward,
                reward_version=reward.version,
            )
        )
    return states


def _build_history(
    state: _QueryState,
    base_backend: Any,
    history_policy: BehaviorPolicy,
) -> int:
    remaining = state.config.total_length - len(state.generated)
    block_length = min(state.config.block_size, remaining)
    rollout_length = remaining - block_length
    state.built_candidates = None
    if rollout_length <= 0:
        return 0

    step_index = len(state.steps)
    generated_prefix = tuple(state.generated)
    candidates = tuple(
        _sample_candidates(
            base_backend,
            state.prompt + generated_prefix,
            state.config.candidate_count,
            block_length,
            state.sampling,
            state.seeds,
            step_index,
        )
    )
    state.built_candidates = candidates
    requests: list[ReplaySampleRequest] = []
    eos = state.sampling.eos_token_id
    for candidate_index, candidate in enumerate(candidates):
        if eos is not None and candidate.token_ids[-1] == eos:
            continue
        key = ReplayKey(
            state.prompt,
            generated_prefix,
            candidate.token_ids,
            state.reward_version,
        )
        for history_index in range(state.config.max_history_per_candidate):
            requests.append(
                ReplaySampleRequest(
                    key=key,
                    max_new_tokens=rollout_length,
                    seed=state.seeds.derive(
                        "replay-history",
                        step_index,
                        candidate_index,
                        history_index,
                    ),
                    record_id=(
                        f"history:{state.problem_index}:{step_index}:"
                        f"{candidate_index}:{history_index}:"
                        f"{state.seeds.derive('history-id', step_index, candidate_index, history_index)}"
                    ),
                )
            )
    records = sample_replay_records(history_policy, requests, state.reward)
    validate_record_probabilities(records, state.registry)
    records_by_key: dict[ReplayKey, list[TokenSequence]] = {}
    for record in records:
        records_by_key.setdefault(record.key, []).append(record.completion)
    for key, completions in records_by_key.items():
        _score_base(base_backend, key, completions, state.sampling)
    for record in records:
        state.store.add_evaluation(record)
    state.history_generated += len(records)
    return len(records)


def _advance(
    state: _QueryState,
    base_backend: Any,
    *,
    candidate_reuse: bool,
) -> None:
    built = state.built_candidates
    step = base_replay_step(
        base_backend=base_backend,
        registry=state.registry,
        store=state.store,
        prompt=state.prompt,
        generated_prefix=tuple(state.generated),
        config=state.config,
        base_sampling=state.sampling,
        reward=state.reward,
        reward_version=state.reward_version,
        seeds=state.seeds,
        step_index=len(state.steps),
        candidate_samples=(built if candidate_reuse else None),
    )
    if built is not None:
        state.candidates_reproduced &= [item.token_ids for item in built] == [
            item.token_ids for item in step.candidates
        ]
        if candidate_reuse:
            state.candidate_draws_reused += len(built)
    state.built_candidates = None
    state.generated.extend(step.selected.token_ids)
    state.steps.append(step)
    eos = state.sampling.eos_token_id
    if eos is not None and eos in step.selected.token_ids:
        state.generated[:] = state.generated[: state.generated.index(eos) + 1]


def _run_arm(
    spec: ArmSpec,
    raw_backend: Any,
    raw_proposal: Any,
    problems: Sequence[Any],
    prompts: Sequence[TokenSequence],
    config: dict[str, Any],
    *,
    root_seed: int,
    workers: int,
    candidate_count: int,
    block_size: int,
    total_length: int,
    history_rollouts: int,
    fresh_rollouts: int,
) -> dict[str, Any]:
    section = config["conditional_is"]
    replay = config["replay"]
    sampling = SamplingConfig(
        temperature=float(config.get("sampling", {}).get("temperature", 1.0)),
        eos_token_id=raw_backend.tokenizer.eos_token_id,
    )
    proposal_sampling = SamplingConfig(
        temperature=sampling.temperature,
        eos_token_id=raw_proposal.tokenizer.eos_token_id,
    )
    states = _make_states(
        raw_backend,
        problems,
        prompts,
        config,
        root_seed=root_seed,
        candidate_count=candidate_count,
        block_size=block_size,
        total_length=total_length,
        reward_temperature=float(section["reward_temperature"]),
        history_rollouts=history_rollouts,
        fresh_rollouts=fresh_rollouts,
        truncation=float(replay["truncation"]),
        replay=spec.replay,
        sampling=sampling,
    )

    cache_wall = 0.0
    online_wall = 0.0
    cache_base_delta = _empty_delta(raw_backend)
    cache_proposal_delta = _empty_delta(raw_proposal)
    online_base_delta = _empty_delta(raw_backend)
    online_proposal_delta = _empty_delta(raw_proposal)
    batching_payload: dict[str, Any] | None = None

    with ExitStack() as stack:
        scheduled_base = (
            stack.enter_context(
                ContinuousBatchingBackend(
                    raw_backend,
                    max_batch_size=int(config["runtime"]["max_batch_size"]),
                    max_batch_tokens=int(config["runtime"]["max_batch_tokens"]),
                    batch_wait_seconds=0.01,
                )
            )
            if spec.continuous_batching
            else raw_backend
        )
        scheduled_proposal = (
            stack.enter_context(
                ContinuousBatchingBackend(
                    raw_proposal,
                    max_batch_size=int(config["runtime"]["max_batch_size"]),
                    max_batch_tokens=int(config["runtime"]["max_batch_tokens"]),
                    batch_wait_seconds=0.01,
                )
            )
            if spec.continuous_batching and spec.replay
            else raw_proposal
        )
        base_backend = ScoreCachingBackend(scheduled_base)
        proposal_backend = ScoreCachingBackend(scheduled_proposal)
        history_policy = BehaviorPolicy.for_backend(
            proposal_backend,
            proposal_sampling,
            label="small-model-history",
        )
        for state in states:
            if spec.replay:
                state.registry.register(history_policy)

        while any(state.active for state in states):
            active = [state for state in states if state.active]
            if spec.replay:
                base_before = raw_backend.snapshot()
                proposal_before = raw_proposal.snapshot()
                _, seconds = _timed(
                    lambda active=active: _execute(
                        active,
                        lambda query, base_backend=base_backend, history_policy=history_policy: (
                            _build_history(
                                query,
                                base_backend,
                                history_policy,
                            )
                        ),
                        concurrent=spec.continuous_batching,
                        workers=workers,
                    )
                )
                cache_wall += seconds
                _add_delta(
                    cache_base_delta,
                    _snapshot_delta(base_before, raw_backend.snapshot()),
                )
                _add_delta(
                    cache_proposal_delta,
                    _snapshot_delta(proposal_before, raw_proposal.snapshot()),
                )

            base_before = raw_backend.snapshot()
            proposal_before = raw_proposal.snapshot()
            _, seconds = _timed(
                lambda active=active: _execute(
                    active,
                    lambda query, base_backend=base_backend, candidate_reuse=spec.candidate_reuse: (
                        _advance(
                            query,
                            base_backend,
                            candidate_reuse=candidate_reuse,
                        )
                    ),
                    concurrent=spec.continuous_batching,
                    workers=workers,
                )
            )
            online_wall += seconds
            _add_delta(
                online_base_delta,
                _snapshot_delta(base_before, raw_backend.snapshot()),
            )
            _add_delta(
                online_proposal_delta,
                _snapshot_delta(proposal_before, raw_proposal.snapshot()),
            )

        if spec.continuous_batching:
            batching_payload = {
                "main_model": asdict(scheduled_base.snapshot()),
                "auxiliary_model": (
                    asdict(scheduled_proposal.snapshot()) if spec.replay else None
                ),
            }

    cache_phase = _phase_payload(
        cache_wall,
        cache_base_delta,
        cache_proposal_delta,
    )
    online_phase = _phase_payload(
        online_wall,
        online_base_delta,
        online_proposal_delta,
    )
    outputs = [tuple(state.generated) for state in states]
    decoded = [raw_backend.decode(tokens) for tokens in outputs]
    correct = [
        extract_numeric_answer(text) == problem.gold_answer
        for text, problem in zip(decoded, problems, strict=True)
    ]
    history_used = sum(
        candidate.estimate.history_count
        for state in states
        for step in state.steps
        for candidate in step.candidates
    )
    fresh_used = sum(
        candidate.estimate.fresh_count
        for state in states
        for step in state.steps
        for candidate in step.candidates
    )
    return {
        "name": spec.name,
        "replay": spec.replay,
        "candidate_reuse": spec.candidate_reuse,
        "continuous_batching": spec.continuous_batching,
        "cache_build": cache_phase,
        "online": online_phase,
        "cold_total": _sum_phases(cache_phase, online_phase),
        "accuracy": sum(correct) / len(correct),
        "correct": correct,
        "outputs": [list(tokens) for tokens in outputs],
        "output_sha256": [
            hashlib.sha256(
                ",".join(str(token) for token in tokens).encode("ascii")
            ).hexdigest()
            for tokens in outputs
        ],
        "steps": [len(state.steps) for state in states],
        "history_generated": sum(state.history_generated for state in states),
        "history_used": history_used,
        "fresh_used": fresh_used,
        "rollout_reuse_rate": (
            history_used / (history_used + fresh_used)
            if history_used + fresh_used
            else 0.0
        ),
        "candidate_draws_reused": sum(state.candidate_draws_reused for state in states),
        "candidates_reproduced": all(state.candidates_reproduced for state in states),
        "evaluation_records_remaining": sum(
            state.store.evaluation_count for state in states
        ),
        "design_records": sum(state.store.design_count for state in states),
        "score_cache": {
            "main_model": asdict(base_backend.snapshot()),
            "auxiliary_model": asdict(proposal_backend.snapshot()),
        },
        "continuous_batching_statistics": batching_payload,
    }


def _arm_value(arm: Mapping[str, Any], phase: str, metric: str) -> float:
    return float(arm[phase][metric])


def _paired_ratio(
    runs: Sequence[Mapping[str, Any]],
    numerator: str,
    denominator: str,
    phase: str,
    metric: str,
) -> dict[str, float | int]:
    ratios = []
    for run in runs:
        arms = {arm["name"]: arm for arm in run["arms"]}
        ratios.append(
            _arm_value(arms[numerator], phase, metric)
            / _arm_value(arms[denominator], phase, metric)
        )
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
    expected = {arm.name for arm in ARMS}
    for run in runs:
        names = [str(arm["name"]) for arm in run["arms"]]
        if len(names) != len(expected) or set(names) != expected:
            raise ValueError("an IS stack run has missing or duplicate arms")

    phase_metrics = (
        "wall_seconds",
        "main_model_estimated_dense_forward_flops",
        "auxiliary_model_estimated_dense_forward_flops",
        "total_estimated_dense_forward_flops",
    )
    aggregates: dict[str, Any] = {}
    for spec in ARMS:
        selected = [
            next(arm for arm in run["arms"] if arm["name"] == spec.name) for run in runs
        ]
        aggregates[spec.name] = {
            phase: {
                metric: _mean_std([_arm_value(arm, phase, metric) for arm in selected])
                for metric in phase_metrics
            }
            for phase in ("cache_build", "online", "cold_total")
        }
        aggregates[spec.name].update(
            {
                "accuracy": _mean_std([float(arm["accuracy"]) for arm in selected]),
                "rollout_reuse_rate": _mean_std(
                    [float(arm["rollout_reuse_rate"]) for arm in selected]
                ),
                "candidate_draws_reused": _mean_std(
                    [float(arm["candidate_draws_reused"]) for arm in selected]
                ),
            }
        )

    comparisons: dict[str, Any] = {}
    comparison_pairs = {
        "continuous_batching_on_fresh": (
            "fresh_continuous",
            "fresh_sequential",
        ),
        "candidate_cache_on_replay": (
            "replay_candidate_cache",
            "replay_sequential",
        ),
        "continuous_batching_on_cached_replay": (
            "replay_candidate_cache_continuous",
            "replay_candidate_cache",
        ),
        "full_stack_over_fresh_sequential": (
            "replay_candidate_cache_continuous",
            "fresh_sequential",
        ),
        "full_stack_over_fresh_continuous": (
            "replay_candidate_cache_continuous",
            "fresh_continuous",
        ),
        "full_stack_over_replay_sequential": (
            "replay_candidate_cache_continuous",
            "replay_sequential",
        ),
    }
    for label, (numerator, denominator) in comparison_pairs.items():
        comparisons[label] = {}
        for phase in ("online", "cold_total"):
            phase_comparisons = {}
            for metric in phase_metrics:
                denominators = []
                for run in runs:
                    arms = {arm["name"]: arm for arm in run["arms"]}
                    denominators.append(_arm_value(arms[denominator], phase, metric))
                if any(value == 0 for value in denominators):
                    continue
                phase_comparisons[metric] = _paired_ratio(
                    runs,
                    numerator,
                    denominator,
                    phase,
                    metric,
                )
            comparisons[label][phase] = phase_comparisons

    exact_pairs = (
        ("fresh_sequential", "fresh_continuous"),
        ("replay_sequential", "replay_candidate_cache"),
        ("replay_candidate_cache", "replay_candidate_cache_continuous"),
    )
    exactness: dict[str, Any] = {}
    all_exact = True
    for left, right in exact_pairs:
        per_run = []
        for run in runs:
            arms = {arm["name"]: arm for arm in run["arms"]}
            equal = arms[left]["outputs"] == arms[right]["outputs"]
            per_run.append(equal)
            all_exact &= equal
        exactness[f"{left}__{right}"] = {
            "all_runs_token_exact": all(per_run),
            "matching_runs": sum(per_run),
            "runs": len(per_run),
        }

    break_even = {
        "wall_queries": [],
        "main_model_flops_queries": [],
        "total_flops_queries": [],
    }
    for run in runs:
        arms = {arm["name"]: arm for arm in run["arms"]}
        baseline = arms["fresh_continuous"]
        stack = arms["replay_candidate_cache_continuous"]
        for output_key, metric in (
            ("wall_queries", "wall_seconds"),
            (
                "main_model_flops_queries",
                "main_model_estimated_dense_forward_flops",
            ),
            ("total_flops_queries", "total_estimated_dense_forward_flops"),
        ):
            break_even[output_key].append(
                _break_even_queries(
                    _arm_value(baseline, "online", metric),
                    _arm_value(stack, "online", metric),
                    _arm_value(stack, "cache_build", metric),
                )
            )

    candidate_wall = float(
        comparisons["candidate_cache_on_replay"]["online"]["wall_seconds"]["mean"]
    )
    candidate_flops = float(
        comparisons["candidate_cache_on_replay"]["online"][
            "main_model_estimated_dense_forward_flops"
        ]["mean"]
    )
    batching_wall = float(
        comparisons["continuous_batching_on_cached_replay"]["online"]["wall_seconds"][
            "mean"
        ]
    )
    batching_flops = float(
        comparisons["continuous_batching_on_cached_replay"]["online"][
            "total_estimated_dense_forward_flops"
        ]["mean"]
    )
    full_wall = float(
        comparisons["full_stack_over_fresh_continuous"]["online"]["wall_seconds"][
            "mean"
        ]
    )
    full_total_flops = float(
        comparisons["full_stack_over_fresh_continuous"]["online"][
            "total_estimated_dense_forward_flops"
        ]["mean"]
    )
    accepted = (
        all_exact
        and candidate_wall <= 1.05
        and candidate_flops <= 1.0
        and batching_wall <= 0.95
        and full_wall <= 0.95
        and full_total_flops <= 1.05
    )
    return {
        "complete": True,
        "runs": len(runs),
        "arms": aggregates,
        "comparisons": comparisons,
        "execution_exactness": exactness,
        "break_even_queries_by_seed": break_even,
        "break_even_scope": (
            "arithmetic comparison against fresh continuous batching only; the "
            "default exact replay lifecycle consumes each evaluation record once, so "
            "repeated use of one frozen history library is not enabled by this result"
        ),
        "decision": {
            "status": "accepted" if accepted else "rejected",
            "criterion": (
                "execution-only pairs must be token-exact; candidate reuse may add no "
                "main-model FLOPs and at most 5% wall time; continuous batching on "
                "cached replay must reduce mean online wall time by at least 5%; the "
                "full stack must reduce mean online wall time by at least 5% versus "
                "continuous-batched fresh-only execution and may add at most 5% total online "
                "FLOPs. The batching-only FLOP factor is reported as a trade-off "
                f"({batching_flops:.6g} in this summary) rather than hidden."
            ),
        },
    }


def _compact_arm(arm: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(arm))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/gsm8k_quick.toml"),
    )
    parser.add_argument("--data", type=Path, default=Path("data/gsm8k/test.jsonl"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/arllm/qwen15b_optimization/is_replay_batching_stack.json"
        ),
    )
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--questions", type=int, default=4)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(20260820, 20260821, 20260822)
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--total-length", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--history-rollouts", type=int, default=1)
    parser.add_argument("--fresh-rollouts", type=int, default=1)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verifier-config", type=Path)
    args = parser.parse_args()
    positive = (
        args.questions,
        args.workers,
        args.total_length,
        args.block_size,
        args.candidate_count,
        args.history_rollouts,
        args.fresh_rollouts,
        *args.seeds,
    )
    if min(positive) <= 0:
        raise ValueError("questions, budgets, workers, and seeds must be positive")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    if args.block_size > args.total_length:
        raise ValueError("block-size cannot exceed total-length")

    with args.config.open("rb") as source:
        config = tomllib.load(source)
    replace_verifier_from_file(config, args.verifier_config)
    config.setdefault("runtime", {})["backend"] = "transformers"
    config["runtime"]["dtype"] = args.dtype
    problems = select_problems(
        load_gsm8k(args.data),
        args.questions,
        seed=int(config["run"]["subset_seed"]),
    )
    setting = {
        "primary_model": "Qwen2.5-1.5B-Instruct",
        "auxiliary_model": "Qwen2.5-0.5B-Instruct",
        "auxiliary_model_role": "off-policy rollout proposal only",
        "dataset": "pinned OpenAI GSM8K test split",
        "problem_indices": [problem.index for problem in problems],
        "hardware": "NVIDIA GeForce RTX 3090 24 GiB",
        "backend": "Transformers",
        "dtype": args.dtype,
        "seeds": list(args.seeds),
        "workers": min(args.workers, args.questions),
        "total_length": args.total_length,
        "block_size": args.block_size,
        "candidate_count": args.candidate_count,
        "history_rollouts_per_candidate": args.history_rollouts,
        "fresh_rollouts_per_candidate": args.fresh_rollouts,
        "verifier": config["verifier"],
        "arms": [arm.name for arm in ARMS],
        "compute_definition": (
            "2 * model parameter count * observed forward token slots; Qwen 1.5B "
            "and Qwen 0.5B are reported separately"
        ),
        "dllm_experiments": False,
    }
    if args.dry_run:
        print(json.dumps(setting, ensure_ascii=False, indent=2))
        return
    if args.restart and args.output.exists():
        args.output.unlink()

    implementation = source_hashes(IMPLEMENTATION_FILES)
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
            "study": "qwen15b_is_replay_candidate_cache_batching_stack",
            "setting": setting,
            "implementation_sha256": implementation,
            "model_artifacts": validate_model_artifacts(config, ("base", "proposal")),
            "runs": [],
            "summary": {"complete": False, "runs": 0},
        }

    completed = {int(run["seed"]) for run in payload["runs"]}
    raw_backend = _load_backend(str(config["models"]["base"]), config)
    try:
        raw_proposal = _load_backend(str(config["models"]["proposal"]), config)
    except BaseException:
        close_backend(raw_backend)
        raise
    try:
        if raw_backend.tokenizer.get_vocab() != raw_proposal.tokenizer.get_vocab():
            raise ValueError(
                "base and auxiliary tokenizers must have identical vocabularies"
            )
        prompts = [_prompt_tokens(raw_backend, problem) for problem in problems]
        sampling = SamplingConfig(eos_token_id=raw_backend.tokenizer.eos_token_id)
        raw_backend.sample_batch(
            [GenerationRequest(prompts[0], 2, sampling, args.seeds[0], "warmup-main")]
        )
        raw_proposal.sample_batch(
            [
                GenerationRequest(
                    prompts[0], 2, sampling, args.seeds[0], "warmup-auxiliary"
                )
            ]
        )

        for seed_position, seed in enumerate(args.seeds):
            if seed in completed:
                continue
            ordered = (
                ARMS[seed_position % len(ARMS) :] + ARMS[: seed_position % len(ARMS)]
            )
            arm_results = []
            for arm in ordered:
                result, seconds = _timed(
                    lambda arm=arm, seed=seed: _run_arm(
                        arm,
                        raw_backend,
                        raw_proposal,
                        problems,
                        prompts,
                        config,
                        root_seed=seed,
                        workers=min(args.workers, args.questions),
                        candidate_count=args.candidate_count,
                        block_size=args.block_size,
                        total_length=args.total_length,
                        history_rollouts=args.history_rollouts,
                        fresh_rollouts=args.fresh_rollouts,
                    )
                )
                result["measured_end_to_end_seconds"] = seconds
                arm_results.append(_compact_arm(result))
                print(
                    json.dumps(
                        {
                            "seed": seed,
                            "arm": arm.name,
                            "online_seconds": result["online"]["wall_seconds"],
                            "cache_seconds": result["cache_build"]["wall_seconds"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            arm_results.sort(
                key=lambda item: [arm.name for arm in ARMS].index(item["name"])
            )
            payload["runs"].append({"seed": seed, "arms": arm_results})
            payload["runs"].sort(key=lambda run: int(run["seed"]))
            payload["summary"] = summarize(payload["runs"])
            _atomic_write(payload, args.output)
            completed.add(seed)
    finally:
        close_backend(raw_proposal)
        close_backend(raw_backend)

    payload["summary"] = summarize(payload["runs"])
    payload["completed_at_unix"] = time.time()
    _atomic_write(payload, args.output)
    print(json.dumps(payload["summary"]["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
