from inference_scaling.rng import SeedStream


def test_request_stream_does_not_depend_on_call_order() -> None:
    stream = SeedStream(123)
    first = stream.derive("prompt-7", "block-2", "candidate-1", "rollout-3")
    stream.derive("unrelated", 99)
    second = stream.derive("prompt-7", "block-2", "candidate-1", "rollout-3")
    assert first == second
    assert first != stream.derive("prompt-7", "block-2", "candidate-1", "rollout-4")

