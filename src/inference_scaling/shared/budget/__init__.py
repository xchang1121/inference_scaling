"""Model-independent budget allocation.

- ``allocation``: variance--cost split between history and fresh rollouts
- ``joint``: pilot weight moments and the integer (B, M, K) choice
- ``planners``: next-block planners built on ``joint``
- ``costs``: reserved forward-token cost model used by the planners

Budget code never samples: callers pass cost estimates and pilot moments.
"""

from inference_scaling.shared.budget.allocation import (
    BudgetAllocation,
    VarianceCostEstimate,
    allocate_fresh_rollout_budget,
    allocate_variance_cost_budget,
)
from inference_scaling.shared.budget.costs import block_costs, completion_reserve
from inference_scaling.shared.budget.joint import (
    BlockBudgetEstimate,
    JointBudgetPlan,
    WeightMoments,
    choose_joint_budget,
    estimate_weight_moments,
)
from inference_scaling.shared.budget.planners import (
    AdaptiveBudgetController,
    FullHorizonPlanner,
    JointPlannerSettings,
    PlanSelection,
)

__all__ = [
    "AdaptiveBudgetController",
    "BlockBudgetEstimate",
    "BudgetAllocation",
    "FullHorizonPlanner",
    "JointBudgetPlan",
    "JointPlannerSettings",
    "PlanSelection",
    "VarianceCostEstimate",
    "WeightMoments",
    "allocate_fresh_rollout_budget",
    "allocate_variance_cost_budget",
    "block_costs",
    "choose_joint_budget",
    "completion_reserve",
    "estimate_weight_moments",
]
