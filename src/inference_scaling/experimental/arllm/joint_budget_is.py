"""Joint M/K/block scheduling over the existing on-policy conditional IS kernel.

Budget units are reserved forward-token slots, including an explicit reward
forward-pass allowance. They are not measured GPU FLOPs or elapsed time.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from inference_scaling.arllm.algorithms.conditional_is import (
    ConditionalISStep,
    RewardFunction,
    _validate_base_sampling,
    conditional_is_step,
)
from inference_scaling.arllm.config import ConditionalISConfig, SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, TokenSequence
from inference_scaling.shared.generation import DEFAULT_MAX_NEW_TOKENS
from inference_scaling.shared.joint_budget import (
    BlockBudgetEstimate,
    JointBudgetPlan,
    WeightMoments,
    choose_joint_budget,
    estimate_weight_moments,
    positive_integer,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.experimental.arllm.adaptive_budget import AdaptiveBudgetController


@dataclass(frozen=True, slots=True)
class JointBudgetISConfig:
    forward_token_budget: int
    total_length: int = DEFAULT_MAX_NEW_TOKENS
    block_sizes: tuple[int, ...] = (64, 128, 256)
    candidate_counts: tuple[int, ...] = (2, 4, 8, 16)
    rollout_counts: tuple[int, ...] = (1, 2, 4, 8)
    pilot_candidates: int = 2
    pilot_rollouts: int = 2
    pilot_fraction: float = 0.15
    reward_temperature: float = 1.0
    reward_forward_passes: int = 1
    relative_variance_floor: float = 1e-4
    planning_mode: str = "full_horizon"
    initial_block_size: int | None = None
    initial_candidate_count: int | None = None
    initial_rollout_count: int | None = None
    adjustment_min_improvement: float = 0.1

    def __post_init__(self) -> None:
        if self.planning_mode not in {"full_horizon", "chunk_adaptive"}:
            raise ValueError("planning_mode must be full_horizon or chunk_adaptive")
        for name in ("forward_token_budget", "total_length"):
            positive_integer(name, getattr(self, name))
        for name in ("block_sizes", "candidate_counts", "rollout_counts"):
            values = getattr(self, name)
            if not values:
                raise ValueError(f"{name} cannot be empty")
            for value in values:
                positive_integer(
                    name, value, minimum=2 if name == "candidate_counts" else 1
                )
        positive_integer("pilot_candidates", self.pilot_candidates, minimum=2)
        positive_integer("pilot_rollouts", self.pilot_rollouts, minimum=2)
        positive_integer("reward_forward_passes", self.reward_forward_passes, minimum=0)
        if not isfinite(self.pilot_fraction) or not 0 <= self.pilot_fraction < 1:
            raise ValueError("pilot_fraction must be in [0, 1)")
        if not isfinite(self.reward_temperature) or self.reward_temperature <= 0:
            raise ValueError("reward_temperature must be finite and positive")
        if (
            not isfinite(self.relative_variance_floor)
            or self.relative_variance_floor <= 0
        ):
            raise ValueError("relative_variance_floor must be finite and positive")
        for name, grid, minimum in (
            ("initial_block_size", self.block_sizes, 1),
            ("initial_candidate_count", self.candidate_counts, 2),
            ("initial_rollout_count", self.rollout_counts, 1),
        ):
            value = getattr(self, name)
            if self.planning_mode == "chunk_adaptive":
                positive_integer(name, value, minimum=minimum)
                if value not in grid:
                    raise ValueError(f"{name} must belong to its configured grid")
            elif value is not None:
                raise ValueError(f"{name} requires chunk_adaptive")
        if not isfinite(self.adjustment_min_improvement) or not 0 < self.adjustment_min_improvement < 1:
            raise ValueError("adjustment_min_improvement must be in (0, 1)")
        if self.planning_mode != "chunk_adaptive" and self.adjustment_min_improvement != 0.1:
            raise ValueError("adjustment_min_improvement requires chunk_adaptive")


@dataclass(frozen=True, slots=True)
class JointBudgetStep:
    plan: JointBudgetPlan
    estimates: tuple[BlockBudgetEstimate, ...]
    pilot_reserved_cost: int
    evaluation: ConditionalISStep
    remaining_budget: int
    adjustment: dict[str, object] | None = None


@dataclass(frozen=True, slots=True)
class JointBudgetISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[JointBudgetStep, ...]
    reserved_forward_tokens: int
    pilot_reserved_forward_tokens: int
    stopping_reason: str


def block_costs(
    *,
    prompt_length: int,
    generated_length: int,
    total_length: int,
    block_size: int,
    reward_forward_passes: int,
) -> tuple[int, int]:
    """Cold prefix + decode + full-sequence reward allowance, without refunds.

    All samples use the same model. Reward callbacks must fit the declared number
    of full-sequence scoring passes; auxiliary models require a separate cost model.
    """
    full = max(1, prompt_length + total_length)
    candidate = max(1, prompt_length + generated_length + block_size)
    scoring = reward_forward_passes * full
    if generated_length + block_size == total_length:
        return candidate + scoring, 0
    # Early-EOS candidates are scored once rather than K times; K >= 1 covers them.
    return candidate, full + scoring


def run_joint_budget_is(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: JointBudgetISConfig,
    reward: RewardFunction,
    seeds: SeedStream,
    *,
    sampling: SamplingConfig | None = None,
) -> JointBudgetISResult:
    """Replan at each prefix; pilots never supply production candidates/weights.

    Full-horizon planning includes a full-remaining-length completion option.
    Adaptive chunks use completion only at the output boundary or when the
    incumbent chunk plus completion reserve no longer fits the budget.
    Insufficient initial budget raises before any backend/reward call.
    Only fixed pointwise rewards and full-support on-policy sampling are supported.
    """
    sampling = sampling or SamplingConfig()
    _validate_base_sampling(sampling)
    finish_cost = (
        min(config.candidate_counts)
        * max(1, len(prompt) + config.total_length)
        * (1 + config.reward_forward_passes)
    )
    if config.forward_token_budget < finish_cost:
        raise ValueError(
            f"budget must reserve at least {finish_cost} forward-token slots"
        )
    generated: TokenSequence = ()
    steps: list[JointBudgetStep] = []
    remaining_budget = config.forward_token_budget
    pilot_spent = 0
    controller = (
        AdaptiveBudgetController(config, finish_cost)
        if config.planning_mode == "chunk_adaptive" else None
    )

    def evaluate(
        block: int, candidates: int, rollouts: int, phase: str
    ) -> ConditionalISStep:
        return conditional_is_step(
            base_backend=backend,
            rollout_backend=backend,
            prompt=prompt,
            generated_prefix=generated,
            config=ConditionalISConfig(
                candidate_count=candidates,
                rollout_count=max(1, rollouts),
                block_size=block,
                total_length=config.total_length,
                reward_temperature=config.reward_temperature,
            ),
            base_sampling=sampling,
            rollout_sampling=sampling,
            reward=reward,
            seeds=SeedStream(seeds.derive("joint_budget_is", len(steps), phase, block)),
            step_index=len(steps),
        )

    def estimate_block(block: int) -> BlockBudgetEstimate:
        candidate_cost, rollout_cost = block_costs(
            prompt_length=len(prompt), generated_length=len(generated),
            total_length=config.total_length, block_size=block,
            reward_forward_passes=config.reward_forward_passes,
        )
        return BlockBudgetEstimate(
            block, WeightMoments(1.0, 0.0 if rollout_cost == 0 else 1.0),
            candidate_cost, rollout_cost,
        )

    def measure(estimate: BlockBudgetEstimate) -> WeightMoments | None:
        pilot = evaluate(estimate.block_size, config.pilot_candidates, config.pilot_rollouts, "pilot")
        weights = [[rollout.log_weight for rollout in candidate.rollouts] for candidate in pilot.candidates]
        if any(not isfinite(value) for group in weights for value in group):
            return None
        return estimate_weight_moments(weights, deterministic=[
            estimate.rollout_cost == 0 or (
                sampling.eos_token_id is not None and candidate.token_ids[-1] == sampling.eos_token_id
            ) for candidate in pilot.candidates
        ])

    def full_horizon_plan(remaining: int):
        blocks = sorted(
            {min(value, remaining) for value in config.block_sizes} | {remaining}
        )
        estimates: list[BlockBudgetEstimate] = []
        pilot_limit = min(
            int(config.pilot_fraction * remaining_budget),
            remaining_budget - finish_cost,
        )
        step_pilot = 0
        for block in blocks:
            candidate_cost, rollout_cost = block_costs(
                prompt_length=len(prompt),
                generated_length=len(generated),
                total_length=config.total_length,
                block_size=block,
                reward_forward_passes=config.reward_forward_passes,
            )
            terminal = block == remaining
            cost = config.pilot_candidates * (
                candidate_cost
                + (0 if terminal else config.pilot_rollouts * rollout_cost)
            )
            moments = WeightMoments(1.0, 0.0 if terminal else 1.0)
            if step_pilot + cost <= pilot_limit:
                pilot = evaluate(
                    block, config.pilot_candidates, config.pilot_rollouts, "pilot"
                )
                moments = estimate_weight_moments(
                    [
                        [rollout.log_weight for rollout in candidate.rollouts]
                        for candidate in pilot.candidates
                    ],
                    deterministic=[
                        terminal
                        or (
                            sampling.eos_token_id is not None
                            and candidate.token_ids[-1] == sampling.eos_token_id
                        )
                        for candidate in pilot.candidates
                    ],
                )
                step_pilot += cost
            estimates.append(
                BlockBudgetEstimate(block, moments, candidate_cost, rollout_cost)
            )
        plan = choose_joint_budget(
            estimates,
            remaining_length=remaining,
            remaining_budget=remaining_budget - step_pilot,
            candidate_counts=config.candidate_counts,
            rollout_counts=config.rollout_counts,
            finish_reserve=finish_cost,
            relative_variance_floor=config.relative_variance_floor,
        )
        return plan, estimates, step_pilot

    while len(generated) < config.total_length:
        remaining = config.total_length - len(generated)
        adjustment = None
        if controller is not None:
            selection = controller.select(remaining, remaining_budget, estimate_block, measure)
            plan, estimates = selection.plan, selection.estimates
            step_pilot = selection.pilot_reserved_cost
            adjustment = selection.adjustment
        else:
            plan, estimates, step_pilot = full_horizon_plan(remaining)
        remaining_budget -= step_pilot
        pilot_spent += step_pilot
        if plan is None:
            raise RuntimeError("completion reservation invariant violated")
        evaluation = evaluate(
            plan.block_size, plan.candidate_count, plan.rollout_count, "evaluation"
        )
        remaining_budget -= int(plan.reserved_cost)
        generated += evaluation.selected.token_ids
        steps.append(
            JointBudgetStep(
                plan, tuple(estimates), step_pilot, evaluation, remaining_budget, adjustment
            )
        )
        if sampling.eos_token_id is not None and sampling.eos_token_id in generated:
            break
    return JointBudgetISResult(
        prompt,
        generated,
        tuple(steps),
        config.forward_token_budget - remaining_budget,
        pilot_spent,
        "eos"
        if sampling.eos_token_id is not None and sampling.eos_token_id in generated
        else "length",
    )
