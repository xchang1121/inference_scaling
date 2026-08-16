"""Infrastructure shared by autoregressive and diffusion language models."""

from inference_scaling.shared.budget import (
    BudgetAllocation,
    VarianceCostEstimate,
    allocate_fresh_rollout_budget,
    allocate_variance_cost_budget,
)
from inference_scaling.shared.config import RuntimeConfig, SMCForestConfig
from inference_scaling.shared.importance import (
    MonteCarloWeightEstimate,
    MonteCarloRolloutWeightProvider,
    ProbabilityObservation,
    ReplayWeightEstimate,
    RolloutObservation,
    TruncatedReplayRolloutWeightProvider,
    WeightedRollout,
    corrected_replay_log_weight,
    logmeanexp,
)
from inference_scaling.shared.mh import (
    MetropolisHastingsDecision,
    MetropolisHastingsProposal,
    MetropolisHastingsTransition,
    apply_metropolis_hastings,
    decide_metropolis_hastings,
    metropolis_hastings_log_acceptance,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.smc import (
    normalize_smc_log_weights,
    partition_resampled_reservoirs,
    systematic_resample,
)
from inference_scaling.shared.stepwise import (
    StepwiseCandidate,
    StepwiseGenerationBackend,
    StepwiseGenerationResult,
    StepwiseSelection,
    normalize_log_weights,
    run_stepwise_generation,
    select_stepwise_candidate,
    stepwise_generation_step,
)
from inference_scaling.shared.types import TokenSequence

__all__ = [
    "BudgetAllocation",
    "VarianceCostEstimate",
    "SMCForestConfig",
    "MetropolisHastingsDecision",
    "MetropolisHastingsProposal",
    "MetropolisHastingsTransition",
    "MonteCarloWeightEstimate",
    "MonteCarloRolloutWeightProvider",
    "ProbabilityObservation",
    "ReplayWeightEstimate",
    "RolloutObservation",
    "StepwiseCandidate",
    "StepwiseGenerationBackend",
    "StepwiseGenerationResult",
    "StepwiseSelection",
    "TruncatedReplayRolloutWeightProvider",
    "WeightedRollout",
    "RuntimeConfig",
    "SeedStream",
    "TokenSequence",
    "apply_metropolis_hastings",
    "allocate_fresh_rollout_budget",
    "allocate_variance_cost_budget",
    "corrected_replay_log_weight",
    "logmeanexp",
    "metropolis_hastings_log_acceptance",
    "normalize_log_weights",
    "normalize_smc_log_weights",
    "partition_resampled_reservoirs",
    "decide_metropolis_hastings",
    "run_stepwise_generation",
    "select_stepwise_candidate",
    "stepwise_generation_step",
    "systematic_resample",
]
