import numpy as np
import pytest

from inference_scaling.shared.rqmc import scrambled_sobol_uniforms


def test_scrambled_sobol_points_are_seeded_and_bounded() -> None:
    first = scrambled_sobol_uniforms(8, 5, seed=17)
    repeated = scrambled_sobol_uniforms(8, 5, seed=17)
    different = scrambled_sobol_uniforms(8, 5, seed=18)

    assert first == repeated
    assert first != different
    assert len(first) == 8
    assert all(len(point) == 5 for point in first)
    assert all(0.0 <= value < 1.0 for point in first for value in point)


@pytest.mark.parametrize("count,dimension,seed", [(0, 1, 0), (1, 0, 0), (1, 1, -1)])
def test_scrambled_sobol_rejects_invalid_shapes(count, dimension, seed) -> None:
    with pytest.raises(ValueError):
        scrambled_sobol_uniforms(count, dimension, seed=seed)


def test_scrambled_sobol_reduces_randomization_variance_for_linear_integrand() -> None:
    count = 8
    repetitions = 128
    sobol_means = np.asarray(
        [
            np.mean(scrambled_sobol_uniforms(count, 1, seed=seed))
            for seed in range(repetitions)
        ]
    )
    iid_means = np.asarray(
        [
            np.random.default_rng(seed).random(count).mean()
            for seed in range(repetitions)
        ]
    )

    assert abs(float(sobol_means.mean()) - 0.5) < 0.01
    assert np.var(sobol_means) < 0.1 * np.var(iid_means)
