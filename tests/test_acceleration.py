from __future__ import annotations

from inference_scaling.acceleration import (
    ActiveBatchSpeculationConfig,
    LowPriorityRunAheadBackend,
    RolloutTokenTree,
    SpeculationTier,
)
from inference_scaling.backends import TabularAutoregressiveBackend
from inference_scaling.config import SamplingConfig
from inference_scaling.types import GenerationRequest


def test_rollout_tree_uses_longest_suffix_and_tracks_verification() -> None:
    tree = RolloutTokenTree(
        max_context_tokens=4,
        max_contexts=100,
        min_context_tokens=1,
        min_token_probability=0.5,
    )
    tree.observe((9, 1, 2, 3, 4))
    tree.observe((8, 1, 2, 3, 5))
    tree.observe((7, 2, 3, 4, 6))

    proposal = tree.draft((0, 1, 2, 3), 3)

    assert proposal.token_ids[:1] == (4,)
    assert proposal.matched_context_tokens == 3
    tree.record_verification(proposed=len(proposal.token_ids), accepted=1)
    snapshot = tree.snapshot()
    assert snapshot.hits == 1
    assert snapshot.proposed_tokens == len(proposal.token_ids)
    assert snapshot.accepted_tokens == 1


def test_active_batch_schedule_is_shared_with_vllm_schema() -> None:
    config = ActiveBatchSpeculationConfig(
        tiers=(SpeculationTier(2, 8), SpeculationTier(5, 3), SpeculationTier(12, 0))
    )
    assert config.draft_tokens(1) == 8
    assert config.draft_tokens(4) == 3
    assert config.draft_tokens(8) == 0
    native = config.vllm_suffix_config(dynamic=True)
    assert native["method"] == "suffix"
    assert native["num_speculative_tokens_per_batch_size"] == [
        [1, 2, 8],
        [3, 5, 3],
        [6, 12, 0],
    ]


def test_low_priority_run_ahead_populates_only_the_draft_tree() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.8, 0.2], model_id="base")
    tree = RolloutTokenTree(
        max_context_tokens=4,
        max_contexts=100,
        min_context_tokens=1,
    )
    wrapper = LowPriorityRunAheadBackend(backend, tree, chunk_tokens=2)
    try:
        accepted = wrapper.submit_run_ahead(
            [GenerationRequest((1,), 4, SamplingConfig(), 7, "idle")]
        )
        wrapper.wait_for_run_ahead()
        snapshot = wrapper.snapshot()
        assert accepted == 1
        assert snapshot.completed_requests == 1
        assert snapshot.completed_tokens == 4
        assert tree.snapshot().observed_sequences >= 2

        foreground = wrapper.sample_batch(
            [GenerationRequest((1,), 1, SamplingConfig(), 9, "foreground")]
        )
        assert len(foreground) == 1
        assert wrapper.snapshot().critical_calls == 1
    finally:
        wrapper.close()

