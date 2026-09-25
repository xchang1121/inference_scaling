"""Joint M/K/block scheduling over the conditional IS kernel.

This module only samples: it runs pilots and production IS steps and keeps the
budget ledger. Each production step is a conditional IS step on the kept
complete sequence, so the carried block and its reused completion cost nothing
again; pilots evaluate fresh candidates at the current cut and are discarded. Plans come from :mod:`inference_scaling.shared.budget.planners`
and planned costs from :mod:`inference_scaling.shared.budget.costs`.

Budget units are forward-token slots, including an explicit reward
forward-pass allowance; they are not measured GPU FLOPs or elapsed time.
Rollouts and completions run until they stop, so their cost is planned from the
expected remaining length observed in earlier completions, and the ledger
charges the tokens each step actually used. The output limit ``total_length``
only caps generation: chunk sizes and counts do not depend on it unless
outputs reach it. The budget is a planning target, and a step can spend more
than planned when its completions run longer than expected.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite

from inference_scaling.arllm.algorithms.conditional_is import (
    ConditionalCandidate,
    ConditionalISStep,
    RetainedSequence,
    RewardFunction,
    conditional_is_step,
    estimate_conditional_weights,
)
from inference_scaling.arllm.algorithms.candidates import sample_candidates, validate_base_sampling
from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, GenerationRequest, TokenSequence
from inference_scaling.shared.budget.costs import block_costs, completion_reserve
from inference_scaling.shared.budget.joint import (
    BlockBudgetEstimate,
    JointBudgetPlan,
    WeightMoments,
    estimate_weight_moments,
    positive_integer,
)
from inference_scaling.shared.budget.planners import (
    AdaptiveBudgetController,
    FullHorizonPlanner,
    PlanningState,
)
from inference_scaling.shared.model.generation import DEFAULT_MAX_NEW_TOKENS
from inference_scaling.shared.rng import SeedStream


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
    # Initial expected output length; None measures one plain completion.
    expected_output_tokens: int | None = None

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
        if self.expected_output_tokens is not None:
            positive_integer("expected_output_tokens", self.expected_output_tokens)


@dataclass(frozen=True, slots=True)
class JointBudgetStep:
    plan: JointBudgetPlan
    estimates: tuple[BlockBudgetEstimate, ...]
    pilot_reserved_cost: int
    evaluation: ConditionalISStep
    remaining_budget: int
    adjustment: dict[str, object] | None = None
    pilot_actual_cost: int = 0
    actual_cost: int = 0
    expected_remaining: int = 0


@dataclass(frozen=True, slots=True)
class JointBudgetISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[JointBudgetStep, ...]
    reserved_forward_tokens: int
    pilot_reserved_forward_tokens: int
    stopping_reason: str
    actual_forward_tokens: int = 0
    pilot_actual_forward_tokens: int = 0
    length_probe_forward_tokens: int = 0


def realized_cost(
    candidates: Sequence[ConditionalCandidate],
    *,
    prefix_length: int,
    reward_forward_passes: int,
    carried: bool = False,
) -> int:
    """Forward-token slots a step used: cold prefixes, decoded tokens and scoring.

    With ``carried``, candidate 0 and its first completion come from the kept
    sequence, so neither is charged again.
    """
    total = 0
    for index, candidate in enumerate(candidates):
        block = prefix_length + len(candidate.token_ids)
        reused = carried and index == 0
        if not reused:
            total += block
        for rollout in candidate.rollouts[int(reused):]:
            sequence = block + len(rollout.token_ids)
            total += (sequence if rollout.token_ids else 0) + reward_forward_passes * sequence
    return total


def run_joint_budget_is(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: JointBudgetISConfig,
    reward: RewardFunction,
    seeds: SeedStream,
    *,
    sampling: SamplingConfig | None = None,
) -> JointBudgetISResult:
    """Replan at each cut of the kept sequence; pilots never supply production candidates/weights.

    Full-horizon planning includes a complete-to-EOS option. Adaptive chunks
    complete only at the output limit or when the incumbent chunk plus the
    completion reserve no longer fits the remaining budget; completing the
    sequence is never refused. Without ``expected_output_tokens`` one plain
    completion from the prompt measures the initial length estimate; each later
    estimate is the mean length of the new rollouts of the previous step. A
    budget that cannot cover the completion reserve raises, before any backend
    or reward call when the estimate is given. Only fixed pointwise rewards and
    full-support on-policy sampling are supported.
    """
    sampling = sampling or SamplingConfig()
    validate_base_sampling(sampling)
    prompt_length = len(prompt)

    def reserve(generated_length: int, expected: int) -> int:
        return completion_reserve(
            prompt_length=prompt_length,
            generated_length=generated_length,
            total_length=config.total_length,
            expected_remaining=expected,
            candidates=min(config.candidate_counts),
            reward_forward_passes=config.reward_forward_passes,
        )

    def require_budget(spent: int, expected: int) -> None:
        needed = spent + reserve(0, expected)
        if config.forward_token_budget < needed:
            raise ValueError(f"budget must cover at least {needed} forward-token slots")

    probe_cost = 0
    if config.expected_output_tokens is not None:
        expected = config.expected_output_tokens
        require_budget(0, expected)
    else:
        # Reject budgets that cannot finish even one token per candidate before probing.
        require_budget(0, 1)
        probe = backend.sample_batch([
            GenerationRequest(
                prompt,
                config.total_length,
                sampling,
                seeds.derive("joint_budget_is", "length_probe"),
                "joint-budget-is:length-probe",
            )
        ])[0]
        expected = max(1, len(probe.token_ids))
        probe_cost = prompt_length + len(probe.token_ids)
        require_budget(probe_cost, expected)
    state = RetainedSequence()
    steps: list[JointBudgetStep] = []
    remaining_budget = config.forward_token_budget - probe_cost
    pilot_costs: list[int] = []
    planner = (
        AdaptiveBudgetController(config)
        if config.planning_mode == "chunk_adaptive"
        else FullHorizonPlanner(config)
    )

    def step_seeds(block: int, phase: str) -> SeedStream:
        # A block reaching the output limit completes the sequence; its seed key
        # omits the limit so completions do not depend on it.
        key = block if state.fixed + block < config.total_length else "rest"
        return SeedStream(seeds.derive("joint_budget_is", len(steps), phase, key))

    def estimate_block(block: int) -> BlockBudgetEstimate:
        candidate_cost, rollout_cost = block_costs(
            prompt_length=prompt_length, generated_length=state.fixed,
            total_length=config.total_length, block_size=block,
            reward_forward_passes=config.reward_forward_passes,
            expected_remaining=expected,
        )
        return BlockBudgetEstimate(
            block, WeightMoments(1.0, 0.0 if rollout_cost == 0 else 1.0),
            candidate_cost, rollout_cost,
        )

    def measure(estimate: BlockBudgetEstimate) -> WeightMoments | None:
        # Pilots evaluate fresh candidates at the current cut and are then discarded.
        pilot_seeds = step_seeds(estimate.block_size, "pilot")
        prefix = state.token_ids[: state.fixed]
        block = min(estimate.block_size, config.total_length - state.fixed)
        pilot = estimate_conditional_weights(
            backend=backend,
            prompt=prompt,
            generated_prefix=prefix,
            candidates=sample_candidates(
                backend, prompt + prefix, config.pilot_candidates, block, sampling, pilot_seeds, len(steps),
            ),
            rollout_length=config.total_length - state.fixed - block,
            rollout_count=config.pilot_rollouts,
            sampling=sampling,
            reward_temperature=config.reward_temperature,
            reward=reward,
            seeds=pilot_seeds,
            step_index=len(steps),
        )
        pilot_costs.append(realized_cost(
            pilot, prefix_length=prompt_length + state.fixed,
            reward_forward_passes=config.reward_forward_passes,
        ))
        weights = [[rollout.log_weight for rollout in candidate.rollouts] for candidate in pilot]
        if any(not isfinite(value) for group in weights for value in group):
            return None
        # A candidate that ends the sequence has one empty completion and a known weight.
        return estimate_weight_moments(weights, deterministic=[
            not candidate.rollouts[0].token_ids for candidate in pilot
        ])

    while not state.token_ids or state.fixed < len(state.token_ids):
        remaining = config.total_length - state.fixed
        finish_reserve = reserve(state.fixed, expected)
        # Earlier overruns never block the completion: the planner always sees
        # at least the completion reserve.
        planning = PlanningState(
            remaining=remaining,
            budget=max(remaining_budget, finish_reserve),
            finish_reserve=finish_reserve,
            expected_remaining=min(expected, remaining),
        )
        pilot_costs.clear()
        selection = planner.select(planning, estimate_block, measure)
        plan = selection.plan
        evaluation, kept = conditional_is_step(
            backend=backend,
            prompt=prompt,
            state=state,
            config=ConditionalISConfig(
                candidate_count=plan.candidate_count,
                rollout_count=max(1, plan.rollout_count),
                block_size=plan.block_size,
                total_length=config.total_length,
                reward_temperature=config.reward_temperature,
            ),
            sampling=sampling,
            reward=reward,
            seeds=step_seeds(plan.block_size, "evaluation"),
            step_index=len(steps),
        )
        pilot_actual = sum(pilot_costs)
        actual = realized_cost(
            evaluation.candidates, prefix_length=prompt_length + state.fixed,
            reward_forward_passes=config.reward_forward_passes, carried=evaluation.retained_candidate,
        )
        remaining_budget -= pilot_actual + actual
        steps.append(
            JointBudgetStep(
                plan, selection.estimates, selection.pilot_reserved_cost, evaluation,
                remaining_budget, selection.adjustment, pilot_actual, actual,
                planning.expected_remaining,
            )
        )
        rollout_lengths = [
            len(rollout.token_ids)
            for index, candidate in enumerate(evaluation.candidates)
            for rollout in candidate.rollouts[int(evaluation.retained_candidate and index == 0):]
            if rollout.token_ids
        ]
        if rollout_lengths:
            expected = max(1, ceil(sum(rollout_lengths) / len(rollout_lengths)))
        state = kept
    pilot_reserved = sum(step.pilot_reserved_cost for step in steps)
    pilot_actual_total = sum(step.pilot_actual_cost for step in steps)
    return JointBudgetISResult(
        prompt,
        state.token_ids,
        tuple(steps),
        pilot_reserved + sum(int(step.plan.reserved_cost) for step in steps),
        pilot_reserved,
        "eos" if sampling.eos_token_id is not None and state.token_ids[-1] == sampling.eos_token_id
        else "length" if len(state.token_ids) >= config.total_length else "stop",
        actual_forward_tokens=probe_cost + pilot_actual_total + sum(step.actual_cost for step in steps),
        pilot_actual_forward_tokens=pilot_actual_total,
        length_probe_forward_tokens=probe_cost,
    )
