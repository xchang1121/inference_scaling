from inference_scaling.backends import ScoreCachingBackend, TabularAutoregressiveBackend
from inference_scaling.config import SamplingConfig
from inference_scaling.types import ScoreRequest


class CountingBackend:
    def __init__(self) -> None:
        self.backend = TabularAutoregressiveBackend({}, fallback=[0.7, 0.3])
        self.score_calls = 0
        self.scored_sequences = 0

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    def sample_batch(self, requests):
        return self.backend.sample_batch(requests)

    def score_batch(self, requests):
        self.score_calls += 1
        self.scored_sequences += sum(len(request.continuations) for request in requests)
        return self.backend.score_batch(requests)


def test_score_cache_deduplicates_and_preserves_flattened_order() -> None:
    counting = CountingBackend()
    cached = ScoreCachingBackend(counting)
    requests = [
        ScoreRequest((), ((0, 1), (1,))),
        ScoreRequest((), ((0, 1),)),
    ]
    expected = counting.backend.score_batch(requests)
    first = cached.score_batch(requests)
    second = cached.score_batch(requests)
    snapshot = cached.snapshot()

    assert first == expected
    assert second == expected
    assert counting.score_calls == 1
    assert counting.scored_sequences == 2
    assert snapshot.entries == 2
    assert snapshot.hits == 3
    assert snapshot.misses == 3


def test_actual_sampling_policy_is_part_of_the_cache_key() -> None:
    counting = CountingBackend()
    cached = ScoreCachingBackend(counting)
    unmodified = cached.score_batch([ScoreRequest((), ((0,),), SamplingConfig())])
    tempered = cached.score_batch(
        [ScoreRequest((), ((0,),), SamplingConfig(temperature=0.5))]
    )
    repeated = cached.score_batch([ScoreRequest((), ((0,),), SamplingConfig())])

    assert unmodified == repeated
    assert unmodified != tempered
    assert counting.scored_sequences == 2
    assert cached.snapshot().entries == 2


def test_lru_capacity_evicts_old_scores() -> None:
    counting = CountingBackend()
    cached = ScoreCachingBackend(counting, maximum_entries=2)
    cached.score_batch([ScoreRequest((), ((0,), (1,), (0, 0)))])
    snapshot = cached.snapshot()
    assert snapshot.entries == 2
    assert snapshot.evictions == 1

    cached.score_batch([ScoreRequest((), ((0,),))])
    assert counting.scored_sequences == 4
