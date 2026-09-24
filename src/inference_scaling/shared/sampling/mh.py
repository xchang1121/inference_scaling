"""Model-independent Metropolis--Hastings transition kernel.

Generation backends only need to construct the current/proposed unnormalized
target log densities and the forward/reverse proposal log probabilities.  The
accept/reject calculation is identical for autoregressive suffix proposals,
diffusion block proposals, and whole-continuation independence proposals.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, isfinite, log
from sys import float_info


@dataclass(frozen=True, slots=True)
class MetropolisHastingsDecision:
    """Result of one model-independent accept/reject decision."""

    log_acceptance: float
    acceptance_probability: float
    accepted: bool


def metropolis_hastings_log_acceptance(
    *,
    current_target_log_density: float,
    proposed_target_log_density: float,
    forward_proposal_log_probability: float = 0.0,
    reverse_proposal_log_probability: float = 0.0,
) -> float:
    """Return ``log(min(1, target ratio * reverse/forward proposal ratio))``."""

    terms = (
        current_target_log_density,
        proposed_target_log_density,
        forward_proposal_log_probability,
        reverse_proposal_log_probability,
    )
    if any(not isfinite(float(value)) for value in terms):
        raise ValueError("MH target and proposal log terms must be finite")
    return min(
        0.0,
        float(proposed_target_log_density)
        - float(current_target_log_density)
        + float(reverse_proposal_log_probability)
        - float(forward_proposal_log_probability),
    )


def decide_metropolis_hastings(
    *,
    current_target_log_density: float,
    proposed_target_log_density: float,
    uniform: float,
    forward_proposal_log_probability: float = 0.0,
    reverse_proposal_log_probability: float = 0.0,
) -> MetropolisHastingsDecision:
    """Apply one MH accept/reject draw using a supplied uniform variate."""

    uniform = float(uniform)
    if not isfinite(uniform) or not 0.0 <= uniform < 1.0:
        raise ValueError("MH uniform variate must lie in [0, 1)")
    log_acceptance = metropolis_hastings_log_acceptance(
        current_target_log_density=current_target_log_density,
        proposed_target_log_density=proposed_target_log_density,
        forward_proposal_log_probability=forward_proposal_log_probability,
        reverse_proposal_log_probability=reverse_proposal_log_probability,
    )
    accepted = log(max(uniform, float_info.min)) <= log_acceptance
    return MetropolisHastingsDecision(
        log_acceptance=log_acceptance,
        acceptance_probability=exp(log_acceptance),
        accepted=accepted,
    )


__all__ = ["MetropolisHastingsDecision", "decide_metropolis_hastings", "metropolis_hastings_log_acceptance"]
