import numpy as np
import pytest

from inference_scaling.shared.sampling.mh import decide_metropolis_hastings, metropolis_hastings_log_acceptance


def test_hastings_ratio_includes_forward_and_reverse_proposals() -> None:
    value = metropolis_hastings_log_acceptance(
        current_target_log_density=-3.0,
        proposed_target_log_density=-2.0,
        forward_proposal_log_probability=-0.5,
        reverse_proposal_log_probability=-2.0,
    )
    assert value == pytest.approx(-0.5)


def test_invalid_uniform_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\)"):
        decide_metropolis_hastings(
            current_target_log_density=0.0,
            proposed_target_log_density=0.0,
            uniform=1.0,
        )


def test_common_kernel_converges_to_two_state_target() -> None:
    rng = np.random.default_rng(2026)
    state = 0
    counts = [0, 0]
    target_log_density = (0.0, np.log(3.0))
    for update in range(30_000):
        proposal = 1 - state
        decision = decide_metropolis_hastings(
            current_target_log_density=target_log_density[state],
            proposed_target_log_density=target_log_density[proposal],
            uniform=float(rng.random()),
        )
        if decision.accepted:
            state = proposal
        if update >= 2_000:
            counts[state] += 1
    assert counts[1] / sum(counts) == pytest.approx(0.75, abs=0.015)
