import numpy as np
import pytest

from inference_scaling.shared.rng import SeedStream, uniform_stream


def test_request_stream_does_not_depend_on_call_order() -> None:
    stream = SeedStream(123)
    first = stream.derive("prompt-7", "block-2", "candidate-1", "rollout-3")
    stream.derive("unrelated", 99)
    second = stream.derive("prompt-7", "block-2", "candidate-1", "rollout-3")
    assert first == second
    assert first != stream.derive("prompt-7", "block-2", "candidate-1", "rollout-4")


def test_seed_paths_are_type_safe_and_root_range_is_validated() -> None:
    stream = SeedStream(1)
    assert stream.derive(1) != stream.derive("1")
    assert stream.derive(True) != stream.derive(1)
    with pytest.raises(ValueError):
        SeedStream(2**128)


def test_uniform_stream_continues_at_any_offset() -> None:
    assert np.array_equal(np.concatenate([uniform_stream(5, 0, 3), uniform_stream(5, 3, 4)]), uniform_stream(5, 0, 7))
