"""Converged inference-scaling pipeline shared by CLI experiments and HTTP serving.

This module is the single source of truth on the algorithm side for:
- translating model/reward/IS settings (plus optional external M, K, B overrides)
  into validated runtime configs;
- composing per-request backend wrappers (request metering, thinking budget,
  sampling scope, token-statistic recording);
- executing Conditional IS and final-content generation with optional decoupled
  ``finish_sampling``, streaming callbacks, and preemption-safe cost attribution.

Neither CLI datasets nor HTTP protocols are imported here, so both ``app`` and
``serve`` remain thin adapters over the same engine-agnostic scaling pipeline.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import time
from typing import Any

from inference_scaling.app.records import Meter, RequestMeter, importance_trace
from inference_scaling.app.rewards import Reward, memoized
from inference_scaling.arllm.algorithms.conditional_is import run_conditional_is
from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.backends.capabilities import EngineCapabilities, backend_capabilities
from inference_scaling.arllm.backends.statistics import StatisticRecorder
from inference_scaling.arllm.backends.thinking_budget import budgeted_backend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.output import output_settings_from_config, thinking_format_from_backend
from inference_scaling.arllm.rewards.intrinsic import ConsilienceReward, SelfCertaintyReward, TokenStatisticReward
from inference_scaling.arllm.scope import SamplingScope
from inference_scaling.arllm.types import TokenStatistic
from inference_scaling.shared.model.generation import generation_budget
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence

CONFIDENCE_REWARDS = frozenset({"self_certainty", "consilience"})


@dataclass(frozen=True, slots=True)
class ScalingOverrides:
    """External runtime overrides for the three primary IS scaling dimensions (M, K, B)."""

    candidate_count: int | None = None
    rollout_count: int | None = None
    block_size: int | None = None

    def __post_init__(self) -> None:
        for name in ("candidate_count", "rollout_count", "block_size"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ScalingOverrides | None":
        if not data:
            return None
        allowed = {"candidate_count", "rollout_count", "block_size"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError("unknown scaling overrides: " + ", ".join(unknown))
        return cls(
            candidate_count=data.get("candidate_count"),
            rollout_count=data.get("rollout_count"),
            block_size=data.get("block_size"),
        )


def sampling_policy_from_config(sampling: Mapping[str, Any], eos_token_id: int | None) -> SamplingConfig:
    """Construct a validated :class:`SamplingConfig` from a settings mapping."""
    return SamplingConfig(
        temperature=float(sampling["temperature"]),
        top_p=float(sampling["top_p"]),
        top_k=sampling["top_k"],
        eos_token_id=eos_token_id,
    )


def finish_sampling_from_config(ar: Mapping[str, Any], base_policy: SamplingConfig) -> SamplingConfig:
    """Resolve optional decoupled final-content sampling, defaulting to the global policy."""
    finish = ar.get("output", {}).get("finish_sampling")
    if finish is None:
        return base_policy
    return sampling_policy_from_config(finish, base_policy.eos_token_id)


def confidence_statistic(kind: str, settings: Mapping[str, Any]) -> TokenStatistic:
    """The token statistic that the ``kind`` confidence reward reads."""

    return TokenStatistic(
        SamplingConfig(temperature=float(settings["score_temperature"])),
        int(settings["top_k"]) if kind == "consilience" else None,
    )


def confidence_reward(kind: str, settings: Mapping[str, Any], backend: Any, thinking_format: Any) -> Reward:
    """Self-Certainty or Consilience of the statistic that ``backend`` (a ``StatisticRecorder``) records."""

    statistic = backend.statistic
    span: dict[str, Any] = {
        "thinking_format": thinking_format if settings["scope"] == "thinking" else None,
        "scope": settings["scope"],
        "incomplete_penalty": float(settings.get("incomplete_penalty", 0.0)),
    }
    if kind == "self_certainty":
        model: TokenStatisticReward = SelfCertaintyReward(
            backend,
            statistic,
            aggregation=settings.get("aggregation", "mean"),
            tail_fraction=float(settings.get("tail_fraction", 0.25)),
            window_fraction=float(settings.get("window_fraction", 0.2)),
            final_weight=float(settings.get("final_weight", 0.5)),
            skip_fraction=float(settings.get("skip_fraction", 0.05)),
            **span,
        )
    else:
        model = ConsilienceReward(
            backend,
            statistic,
            window_fraction=float(settings["window_fraction"]),
            window_tokens=settings["window_tokens"],
            skip_fraction=float(settings["skip_fraction"]),
            initial_penalty=float(settings["initial_penalty"]),
            **span,
        )
    return Reward(float(settings["temperature"]), memoized(model.batch), model.describe(), model)


def conditional_is_config(
    config: Mapping[str, Any],
    reward: Reward,
    *,
    total_length: int,
    backend: Any | None = None,
    overrides: ScalingOverrides | None = None,
) -> ConditionalISConfig:
    """Build :class:`ConditionalISConfig` from settings and optional external (M, K, B) overrides."""

    schedule = str(config.get("exploration_schedule", "fixed"))
    scheduler = getattr(backend, "scheduler_config", None)
    if schedule not in {"two_stage", "adaptive"} or not getattr(backend, "supports_streaming_generation", False):
        scheduler = None
    candidate_count = int(overrides.candidate_count if overrides and overrides.candidate_count is not None
                          else config["candidate_count"])
    rollout_count = int(overrides.rollout_count if overrides and overrides.rollout_count is not None
                        else config["rollout_count"])
    raw_block_size = int(overrides.block_size if overrides and overrides.block_size is not None
                         else config["block_size"])
    return ConditionalISConfig(
        candidate_count=candidate_count,
        rollout_count=rollout_count,
        block_size=min(raw_block_size, total_length),
        total_length=total_length,
        reward_temperature=reward.temperature,
        block_first=bool(config["block_first"]),
        exploration_schedule=schedule,
        final_pool_limit=scheduler.max_in_flight if scheduler is not None else None,
        selection_ready=bool(config.get("selection_ready", False)),
        selection=str(config.get("selection", "sir")),
        pilot_gate=bool(config.get("pilot_gate", True)),
        selective_lookahead=bool(config.get("selective_lookahead", True)),
        straggler_pruning=bool(config.get("straggler_pruning", True)),
        adaptive_temperature=bool(config.get("adaptive_temperature", True)),
        pilot_tokens=config.get("pilot_tokens"),
    )


@dataclass(frozen=True)
class ScalingRequest:
    """Engine- and protocol-agnostic request to run Conditional IS + answer finish."""

    prompt: TokenSequence
    budget: Mapping[str, Any]
    maximum: int
    seed: int
    ar: Mapping[str, Any]
    reward_kind: str
    reward_settings: Mapping[str, Any]
    deadline: float | None = None
    overrides: ScalingOverrides | None = None
    on_commit: Callable[[TokenSequence], None] | None = None
    on_search_complete: Callable[[TokenSequence], None] | None = None
    on_content: Callable[[TokenSequence], None] | None = None
    # Optional custom seed for the final content step; defaults to SeedStream(seed).derive("final-content").
    finish_seed: int | None = None


@dataclass(frozen=True)
class ScalingResult:
    """Complete output of one Conditional IS scaling execution."""

    tokens: TokenSequence
    search_tokens: TokenSequence
    info: dict[str, Any]
    reward_value: float
    steps: tuple[Any, ...]
    stopping_reason: str
    config: ConditionalISConfig
    cost: dict[str, Any] | None
    generated_tokens: int | None
    fallbacks: list[str]
    trace: dict[str, Any]
    rollout_scheduler: str
    thought_ready_seconds: float
    total_seconds: float


class ScalingPipeline:
    """Converged algorithm execution layer over any :class:`AutoregressiveBackend`."""

    def __init__(self, raw_backend: Any, execution_backend: Any | None = None) -> None:
        self.raw = raw_backend
        self.backend = execution_backend if execution_backend is not None else raw_backend

    def capabilities(self) -> EngineCapabilities:
        return backend_capabilities(self.raw)

    def prepare_budget(
        self,
        prompt: TokenSequence,
        ar: Mapping[str, Any],
        thinking: Mapping[str, Any],
        *,
        client_max_tokens: int | None = None,
        validate_budget_backend: bool = False,
    ) -> tuple[dict[str, Any], int]:
        configured = ar["max_new_tokens"]
        limits = [value for value in (configured, client_max_tokens) if value is not None]
        requested = min(limits) if limits else None
        budget = generation_budget(
            requested,
            len(prompt),
            [self.raw],
            context_window=ar["engine"]["context_window"],
            thinking=thinking,
        )
        if client_max_tokens is not None or validate_budget_backend:
            budget.update(configured_max_new_tokens=configured, client_max_tokens=client_max_tokens)
        maximum = int(budget["effective_max_new_tokens"])
        if validate_budget_backend:
            thinking_format = thinking_format_from_backend(self.raw, output_settings_from_config(ar))
            budgeted_backend(self.raw, prompt, thinking_format, budget)
        return budget, maximum

    def build_scope(
        self,
        prompt: TokenSequence,
        ar: Mapping[str, Any],
        *,
        active: bool = True,
        reward_kind: str | None = None,
        reward_settings: Mapping[str, Any] | None = None,
    ) -> SamplingScope:
        scope = SamplingScope.from_config(self.raw, ar, active=active).for_prompt(prompt)
        if scope.scope == "thinking" and (
            reward_kind == "verifier"
            or (reward_kind in CONFIDENCE_REWARDS and reward_settings is not None and reward_settings["scope"] == "full")
        ):
            scope = scope.full_fallback("reward_uses_full_sequence")
        return scope

    def run_is(
        self,
        request: ScalingRequest,
        *,
        is_runner: Callable[..., Any] = run_conditional_is,
        track_preemptions: bool = True,
        meter: Meter | None = None,
    ) -> ScalingResult:
        if request.reward_kind not in CONFIDENCE_REWARDS:
            raise ValueError("ScalingPipeline.run_is requires self_certainty or consilience reward")
        raw, ar = self.raw, request.ar
        eos = getattr(raw.tokenizer, "eos_token_id", None)
        policy = sampling_policy_from_config(ar["sampling"], eos)
        finish_policy = finish_sampling_from_config(ar, policy)
        thinking_format = thinking_format_from_backend(raw, output_settings_from_config(ar))

        active_meter = meter if meter is not None else Meter({})
        measured = RequestMeter(self.backend, active_meter, raw.parameter_count)
        bounded = budgeted_backend(measured, request.prompt, thinking_format, dict(request.budget))
        scope = self.build_scope(
            request.prompt,
            ar,
            active=True,
            reward_kind=request.reward_kind,
            reward_settings=request.reward_settings,
        )
        stat_backend = StatisticRecorder(
            scope.wrap(bounded, request.prompt),
            confidence_statistic(request.reward_kind, request.reward_settings),
        )
        reward = confidence_reward(request.reward_kind, request.reward_settings, stat_backend, thinking_format)
        config = conditional_is_config(
            ar["algorithms"]["is"],
            reward,
            total_length=request.maximum,
            backend=raw,
            overrides=request.overrides,
        )

        clock_start = time.perf_counter()
        snapshot = getattr(raw, "snapshot", lambda: None)
        preemptions = getattr(snapshot(), "preemptions", 0) if track_preemptions else 0

        with active_meter.phase("search"):
            result = is_runner(
                stat_backend,
                request.prompt,
                config,
                reward.batch,
                SeedStream(request.seed),
                sampling=policy,
                deadline=request.deadline,
                prefix_reward=reward.model.cached_prefix,
                on_commit=request.on_commit,
            )
        thought_ready = time.perf_counter() - clock_start
        if request.on_search_complete is not None:
            request.on_search_complete(result.token_ids)

        finish_seed = (
            request.finish_seed
            if request.finish_seed is not None
            else SeedStream(request.seed).derive("final-content")
        )
        with active_meter.phase("finish"):
            tokens, info = scope.finish(
                bounded,
                request.prompt,
                result.token_ids,
                max_new_tokens=request.maximum,
                sampling=finish_policy,
                seed=finish_seed,
                on_content=request.on_content,
            )

        if track_preemptions and getattr(snapshot(), "preemptions", 0) != preemptions:
            active_meter.attribute(None)
        cost = active_meter.cost(
            lambda delta: delta.get("generation_forward_token_slots", 0) + delta.get("score_forward_token_slots", 0),
            lambda _role, delta: delta.get("estimated_dense_forward_flops", 0),
        )
        generated_tokens = (
            sum(phase.get("base", {}).get("generated_tokens", 0) for phase in active_meter.phases.values())
            if cost is not None
            else None
        )
        fallbacks = [info["sampling_fallback_reason"]] if info["sampling_fallback_reason"] else []
        confidence = getattr(reward, "model", None)
        if isinstance(confidence, TokenStatisticReward) and (reason := confidence.fallback(request.prompt, tokens)):
            fallbacks.append(f"{request.reward_kind}:{reason}")

        rollout_scheduler = "streaming" if getattr(stat_backend, "supports_streaming_generation", False) else "batch"
        trace = importance_trace(list(result.steps))
        trace["generation_budget"] = dict(request.budget)
        trace["reward"] = dict(reward.description)
        trace["rollout_scheduler"] = rollout_scheduler
        trace["exploration_schedule"] = config.exploration_schedule
        trace["final_pool_limit"] = config.final_pool_limit
        trace["selection_ready"] = config.selection_ready
        trace["selection"] = config.selection
        kept_reward = float(result.steps[-1].selected.rollouts[result.steps[-1].completion_index].reward)
        return ScalingResult(
            tokens=tokens,
            search_tokens=result.token_ids,
            info=info,
            reward_value=kept_reward,
            steps=result.steps,
            stopping_reason=result.stopping_reason,
            config=config,
            cost=cost,
            generated_tokens=generated_tokens,
            fallbacks=fallbacks,
            trace=trace,
            rollout_scheduler=rollout_scheduler,
            thought_ready_seconds=thought_ready,
            total_seconds=time.perf_counter() - clock_start,
        )


__all__ = [
    "CONFIDENCE_REWARDS",
    "ScalingOverrides",
    "ScalingPipeline",
    "ScalingRequest",
    "ScalingResult",
    "conditional_is_config",
    "confidence_reward",
    "confidence_statistic",
    "finish_sampling_from_config",
    "sampling_policy_from_config",
]
