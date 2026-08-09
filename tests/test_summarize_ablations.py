import pytest

from experiments.summarize_gsm8k_ablations import (
    RUNNER_PATH,
    SERIALIZATION_ONLY_RUNNER_TRANSITION,
    _implementation_provenance,
)


def _variant(runner: str, algorithm: str = "algorithm") -> dict[str, str]:
    return {RUNNER_PATH: runner, "src/inference_scaling/algorithm.py": algorithm}


def test_known_diagnostic_serialization_transition_keeps_algorithm_provenance() -> None:
    runners = sorted(SERIALIZATION_ONLY_RUNNER_TRANSITION)

    algorithm, observed_runners, note = _implementation_provenance(
        [_variant(runners[0]), _variant(runners[1])]
    )

    assert algorithm == {"src/inference_scaling/algorithm.py": "algorithm"}
    assert observed_runners == runners
    assert "diagnostic keys" in note


def test_unknown_runner_or_algorithm_transition_is_rejected() -> None:
    with pytest.raises(ValueError, match="incompatible runners"):
        _implementation_provenance([_variant("first"), _variant("second")])
    runner = next(iter(SERIALIZATION_ONLY_RUNNER_TRANSITION))
    with pytest.raises(ValueError, match="different algorithms"):
        _implementation_provenance(
            [_variant(runner, "first"), _variant(runner, "second")]
        )
