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
