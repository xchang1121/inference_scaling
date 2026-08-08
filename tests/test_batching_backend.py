import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from inference_scaling.backends import (
    ContinuousBatchingBackend,
    TabularAutoregressiveBackend,
)
from inference_scaling.config import SamplingConfig
from inference_scaling.types import GenerationRequest, ScoreRequest


class RecordingBackend:
    def __init__(self) -> None:
        self.backend = TabularAutoregressiveBackend({}, fallback=[0.65, 0.35])
        self.lock = threading.Lock()
        self.sample_batch_sizes: list[int] = []
        self.score_batch_sizes: list[int] = []

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    def sample_batch(self, requests):
        with self.lock:
            self.sample_batch_sizes.append(len(requests))
        return self.backend.sample_batch(requests)

    def score_batch(self, requests):
        with self.lock:
            self.score_batch_sizes.append(
                sum(len(request.continuations) for request in requests)
            )
        return self.backend.score_batch(requests)


def test_concurrent_sampling_is_coalesced_and_seed_stable() -> None:
    recording = RecordingBackend()
    requests = [
        GenerationRequest((), 3, SamplingConfig(), 100 + index, f"request-{index}")
        for index in range(12)
    ]
    expected = recording.backend.sample_batch(requests)
    with ContinuousBatchingBackend(
        recording,
        max_batch_size=12,
        max_batch_tokens=100,
        batch_wait_seconds=0.02,
    ) as batched:
        barrier = threading.Barrier(len(requests))

        def run(request):
            barrier.wait()
            return batched.sample_batch([request])[0]

        with ThreadPoolExecutor(max_workers=len(requests)) as executor:
            actual = list(executor.map(run, requests))
        snapshot = batched.snapshot()

    assert actual == expected
    assert max(recording.sample_batch_sizes) > 1
    assert snapshot.sample_requests == len(requests)
    assert snapshot.maximum_sample_batch > 1


def test_score_requests_are_flattened_and_split_without_reordering() -> None:
    recording = RecordingBackend()
    requests = [
        ScoreRequest((index % 2,), ((0,), (1, 0)), SamplingConfig())
        for index in range(6)
    ]
    expected = recording.backend.score_batch(requests)
    with ContinuousBatchingBackend(
        recording,
        max_batch_size=12,
        max_batch_tokens=100,
        batch_wait_seconds=0.01,
    ) as batched:
        actual = batched.score_batch(requests)
        snapshot = batched.snapshot()

    assert actual == expected
    assert snapshot.score_sequences == 12
    assert snapshot.maximum_score_batch == 12


def test_closed_batching_backend_rejects_new_work() -> None:
    batched = ContinuousBatchingBackend(RecordingBackend())
    batched.close()
    with pytest.raises(RuntimeError, match="closed"):
        batched.sample_batch(
            [GenerationRequest((), 1, SamplingConfig(), 1, "after-close")]
        )
