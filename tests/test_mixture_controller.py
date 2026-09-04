from __future__ import annotations

import pytest

from online_speculation.mixture_controller import (
    EmaMixtureConfig,
    VerifierEmaMixtureController,
)


def test_ema_controller_activates_and_deactivates_only_after_observations() -> None:
    controller = VerifierEmaMixtureController(
        EmaMixtureConfig(
            max_candidate_weight=0.25,
            evaluation_interval=2,
            warmup_observations=2,
            ema_decay=0.0,
            activation_margin=0.01,
            deactivation_margin=0.01,
        )
    )
    assert controller.weight == 0.0
    first = controller.observe(cycle=2, advantage=0.02)
    assert first.action == "warmup"
    assert controller.weight == 0.0

    second = controller.observe(cycle=4, advantage=0.02)
    assert second.action == "activate"
    assert controller.weight == 0.25

    third = controller.observe(cycle=6, advantage=-0.02)
    assert third.action == "deactivate"
    assert controller.weight == 0.0


def test_ema_controller_rejects_off_schedule_or_nonfinite_feedback() -> None:
    controller = VerifierEmaMixtureController(EmaMixtureConfig())
    with pytest.raises(ValueError, match="schedule"):
        controller.observe(cycle=1, advantage=0.1)
    with pytest.raises(ValueError, match="finite"):
        controller.observe(cycle=4, advantage=float("nan"))


@pytest.mark.parametrize(
    "config",
    (
        EmaMixtureConfig(activation_margin=float("nan")),
        EmaMixtureConfig(evaluation_interval=0),
        EmaMixtureConfig(warmup_observations=1.5),  # type: ignore[arg-type]
    ),
)
def test_ema_controller_rejects_invalid_configuration(
    config: EmaMixtureConfig,
) -> None:
    with pytest.raises(ValueError):
        VerifierEmaMixtureController(config)
