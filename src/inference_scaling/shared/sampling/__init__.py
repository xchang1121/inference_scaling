"""Sampling-algorithm kernels shared by autoregressive and diffusion models.

The kernels contain only target, weight and selection arithmetic. Model
families supply proposals, rollouts and probability scores through adapters.

- ``importance``: on/off-policy rollout weights and truncated replay correction
- ``mh``: Metropolis--Hastings acceptance kernel
- ``stepwise``: finite-candidate SIR driver used by conditional IS
- ``smc``: particle-weight normalization and resampling
"""

from inference_scaling.shared.sampling.importance import (
    MonteCarloRolloutWeightProvider,
    MonteCarloWeightEstimate,
    ProbabilityObservation,
    ReplayWeightEstimate,
    RolloutObservation,
    TruncatedReplayRolloutWeightProvider,
    WeightedRollout,
    corrected_replay_log_weight,
    logmeanexp,
)
from inference_scaling.shared.sampling.mh import (
    MetropolisHastingsDecision,
    MetropolisHastingsProposal,
    MetropolisHastingsTransition,
    apply_metropolis_hastings,
    decide_metropolis_hastings,
    metropolis_hastings_log_acceptance,
)
from inference_scaling.shared.sampling.smc import (
    normalize_smc_log_weights,
    partition_resampled_reservoirs,
    systematic_resample,
)
from inference_scaling.shared.sampling.stepwise import (
    StepwiseCandidate,
    StepwiseGenerationBackend,
    StepwiseGenerationResult,
    StepwiseSelection,
    categorical_index_from_uniform,
    normalize_log_weights,
    run_stepwise_generation,
    select_stepwise_candidate,
    stepwise_generation_step,
)

__all__ = [
    "MetropolisHastingsDecision",
    "MetropolisHastingsProposal",
    "MetropolisHastingsTransition",
    "MonteCarloRolloutWeightProvider",
    "MonteCarloWeightEstimate",
    "ProbabilityObservation",
    "ReplayWeightEstimate",
    "RolloutObservation",
    "StepwiseCandidate",
    "StepwiseGenerationBackend",
    "StepwiseGenerationResult",
    "StepwiseSelection",
    "TruncatedReplayRolloutWeightProvider",
    "WeightedRollout",
    "apply_metropolis_hastings",
    "categorical_index_from_uniform",
    "corrected_replay_log_weight",
    "decide_metropolis_hastings",
    "logmeanexp",
    "metropolis_hastings_log_acceptance",
    "normalize_log_weights",
    "normalize_smc_log_weights",
    "partition_resampled_reservoirs",
    "run_stepwise_generation",
    "select_stepwise_candidate",
    "stepwise_generation_step",
    "systematic_resample",
]
