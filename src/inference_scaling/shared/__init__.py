"""Infrastructure shared by autoregressive and diffusion language models."""

from inference_scaling.shared.importance import (
    MonteCarloEnergyEstimate,
    MonteCarloRolloutWeightProvider,
    ProbabilityObservation,
    ReplayEnergyEstimate,
    RolloutObservation,
    TruncatedReplayRolloutWeightProvider,
    WeightedRollout,
    corrected_replay_log_energy,
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
from inference_scaling.shared.config import RuntimeConfig
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.stepwise import (
    StepwiseCandidate,
    StepwiseGenerationBackend,
    StepwiseGenerationResult,
    StepwiseSelection,
    normalize_log_energies,
    run_stepwise_generation,
    select_stepwise_candidate,
    stepwise_generation_step,
)
from inference_scaling.shared.types import TokenSequence

__all__ = [
    "MetropolisHastingsDecision",
    "MetropolisHastingsProposal",
    "MetropolisHastingsTransition",
    "MonteCarloEnergyEstimate",
    "MonteCarloRolloutWeightProvider",
    "ProbabilityObservation",
    "ReplayEnergyEstimate",
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
    "corrected_replay_log_energy",
    "logmeanexp",
    "metropolis_hastings_log_acceptance",
    "normalize_log_energies",
    "decide_metropolis_hastings",
    "run_stepwise_generation",
    "select_stepwise_candidate",
    "stepwise_generation_step",
]
