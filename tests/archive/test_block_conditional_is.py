"""Archived block conditional IS keeps the behavior behind its reported results."""

from collections import Counter
from math import exp

import pytest

from inference_scaling.archive.arllm.block_conditional_is import (
    BlockConditionalISConfig,
    block_conditional_is_step,
    run_block_conditional_is,
)
from inference_scaling.arllm.backends import TabularAutoregressiveBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import ScoreRequest
from inference_scaling.shared.metrics import total_variation
from inference_scaling.shared.rng import SeedStream


def _backend(**kwargs) -> TabularAutoregressiveBackend:
    return TabularAutoregressiveBackend(
        {(): [0.7, 0.3], (0,): [0.9, 0.1], (1,): [0.2, 0.8]}, fallback=[0.5, 0.5], **kwargs
    )


def _reward(_prompt, generated) -> float:
    return 1.0 if tuple(generated) == (1, 1) else 0.0


def _step(config, seed, *, base=None, rollout_sampling=SamplingConfig(), **kwargs):
    return block_conditional_is_step(
        base_backend=base or _backend(),
        prompt=(),
        generated_prefix=(),
        config=config,
        reward=_reward,
        seeds=SeedStream(seed),
        step_index=0,
        rollout_sampling=rollout_sampling,
        **kwargs,
    )


def _rollouts(step):
    return [rollout for candidate in step.candidates for rollout in candidate.rollouts]


def test_off_policy_first_block_approaches_the_exact_target() -> None:
    completion = ((0.9, 0.1), (0.2, 0.8))
    weights = [
        (0.7, 0.3)[first] * sum(completion[first][token] * exp(_reward((), (first, token))) for token in (0, 1))
        for first in (0, 1)
    ]
    target = {first: weight / sum(weights) for first, weight in enumerate(weights)}
    config = BlockConditionalISConfig(candidate_count=12, rollout_count=8, block_size=1, total_length=2)
    counts: Counter[int] = Counter()
    for trial in range(500):
        step = _step(config, 10_000 + trial, rollout_sampling=SamplingConfig(temperature=0.55))
        counts[step.selected.token_ids[0]] += 1
    assert total_variation({token: count / 500 for token, count in counts.items()}, target) < 0.08


def test_off_policy_ratio_scores_only_the_completion_suffix() -> None:
    base_sampling = SamplingConfig(temperature=0.8)
    step = _step(
        BlockConditionalISConfig(candidate_count=2, rollout_count=2, block_size=1, total_length=2),
        192,
        base_sampling=base_sampling,
        rollout_sampling=SamplingConfig(temperature=0.5),
    )
    for candidate in step.candidates:
        for rollout in candidate.rollouts:
            assert len(rollout.token_ids) == 1
            expected = _backend().score_batch([ScoreRequest(candidate.token_ids, (rollout.token_ids,), base_sampling)])[0]
            assert rollout.base_logprob == pytest.approx(sum(expected))
            assert rollout.log_weight == pytest.approx(
                rollout.reward + rollout.base_logprob - rollout.proposal_logprob
            )


def test_log_ratio_clipping_is_recorded_per_rollout() -> None:
    step = _step(
        BlockConditionalISConfig(
            candidate_count=4, rollout_count=4, block_size=1, total_length=2, importance_log_ratio_clip=0.05,
        ),
        293,
        rollout_sampling=SamplingConfig(temperature=0.25),
    )
    rollouts = _rollouts(step)
    assert all(abs(item.applied_log_importance_ratio) <= 0.05 for item in rollouts)
    assert any(item.raw_log_importance_ratio != item.applied_log_importance_ratio for item in rollouts)
    assert all(item.log_weight == pytest.approx(item.reward + item.applied_log_importance_ratio) for item in rollouts)


def test_uncorrected_proposal_rollouts_skip_base_rescoring() -> None:
    class NoScoreBackend(TabularAutoregressiveBackend):
        def score_batch(self, requests):
            raise AssertionError("uncorrected proposal rollouts must not be rescored")

    step = block_conditional_is_step(
        base_backend=NoScoreBackend({}, fallback=[0.6, 0.4], model_id="base"),
        prompt=(),
        generated_prefix=(),
        config=BlockConditionalISConfig(
            candidate_count=3, rollout_count=2, block_size=1, total_length=2, apply_importance_correction=False,
        ),
        reward=_reward,
        seeds=SeedStream(394),
        step_index=0,
        rollout_backend=TabularAutoregressiveBackend({}, fallback=[0.2, 0.8], model_id="proposal"),
    )
    rollouts = _rollouts(step)
    assert all(item.base_logprob is None and item.applied_log_importance_ratio is None for item in rollouts)
    assert all(item.log_weight == pytest.approx(item.reward) for item in rollouts)


def test_block_run_commits_only_the_selected_blocks() -> None:
    result = run_block_conditional_is(
        TabularAutoregressiveBackend({}, fallback=[0.5, 0.5]),
        (),
        BlockConditionalISConfig(candidate_count=2, rollout_count=2, block_size=2, total_length=5),
        lambda _prompt, generated: float(sum(generated)),
        SeedStream(17),
    )
    assert len(result.token_ids) == 5
    assert result.token_ids == sum((step.selected.token_ids for step in result.steps), ())
    assert [len(step.selected.token_ids) for step in result.steps] == [2, 2, 1]


@pytest.mark.parametrize(
    "options",
    [
        {"importance_log_ratio_clip": 1.0, "apply_importance_correction": False},
        {"block_size": 8, "total_length": 4},
        {"reward_temperature": 0.0},
    ],
)
def test_block_config_rejects_inconsistent_options(options) -> None:
    with pytest.raises(ValueError):
        BlockConditionalISConfig(**options)
