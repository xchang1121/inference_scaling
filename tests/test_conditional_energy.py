from collections import Counter
from math import exp

import pytest

from inference_scaling.algorithms.conditional_energy import conditional_is_step, run_conditional_is
from inference_scaling.backends import TabularAutoregressiveBackend
from inference_scaling.config import ConditionalEnergyConfig, SamplingConfig
from inference_scaling.metrics import total_variation
from inference_scaling.rng import SeedStream


def _backend() -> TabularAutoregressiveBackend:
    return TabularAutoregressiveBackend(
        {
            (): [0.7, 0.3],
            (0,): [0.9, 0.1],
            (1,): [0.2, 0.8],
        },
        fallback=[0.5, 0.5],
    )


def _reward(_prompt, generated) -> float:
    return 1.0 if tuple(generated) == (1, 1) else 0.0


def _exact_first_token_target() -> dict[int, float]:
    base_first = (0.7, 0.3)
    completion = ((0.9, 0.1), (0.2, 0.8))
    energies = []
    for candidate in (0, 1):
        energy = sum(
            completion[candidate][token]
            * exp(_reward((), (candidate, token)))
            for token in (0, 1)
        )
        energies.append(energy)
    weights = [base_first[index] * energies[index] for index in (0, 1)]
    total = sum(weights)
    return {index: weights[index] / total for index in (0, 1)}


@pytest.mark.parametrize("off_policy", [False, True])
def test_conditional_is_first_candidate_approaches_exact_conditional_energy_target(off_policy) -> None:
    backend = _backend()
    config = ConditionalEnergyConfig(
        candidate_count=12,
        rollout_count=8,
        block_size=1,
        total_length=2,
        reward_temperature=1.0,
    )
    counts: Counter[int] = Counter()
    trials = 500
    for trial in range(trials):
        step = conditional_is_step(
            base_backend=backend,
            rollout_backend=backend,
            prompt=(),
            generated_prefix=(),
            config=config,
            base_sampling=SamplingConfig(),
            rollout_sampling=SamplingConfig(temperature=0.55 if off_policy else 1.0),
            reward=_reward,
            seeds=SeedStream(10_000 + trial),
            step_index=0,
        )
        counts[step.selected.token_ids[0]] += 1
    empirical = {token: count / trials for token, count in counts.items()}
    assert total_variation(empirical, _exact_first_token_target()) < 0.08


def test_off_policy_ratio_scores_only_rollout_suffix() -> None:
    backend = _backend()
    step = conditional_is_step(
        base_backend=backend,
        rollout_backend=backend,
        prompt=(),
        generated_prefix=(),
        config=ConditionalEnergyConfig(candidate_count=2, rollout_count=2, block_size=1, total_length=2),
        base_sampling=SamplingConfig(),
        rollout_sampling=SamplingConfig(temperature=0.5),
        reward=_reward,
        seeds=SeedStream(91),
        step_index=0,
    )
    for candidate in step.candidates:
        for rollout in candidate.rollouts:
            assert len(rollout.token_ids) == 1
            assert rollout.log_weight == pytest.approx(
                rollout.reward + rollout.base_logprob - rollout.proposal_logprob
            )


def test_rollout_budget_subtracts_candidate_block() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.5, 0.5])
    step = conditional_is_step(
        base_backend=backend,
        rollout_backend=backend,
        prompt=(),
        generated_prefix=(),
        config=ConditionalEnergyConfig(candidate_count=3, rollout_count=2, block_size=2, total_length=5),
        base_sampling=SamplingConfig(),
        rollout_sampling=SamplingConfig(),
        reward=lambda _prompt, generated: float(sum(generated)),
        seeds=SeedStream(4),
        step_index=0,
    )
    assert all(len(candidate.token_ids) == 2 for candidate in step.candidates)
    assert all(
        len(rollout.token_ids) == 3
        for candidate in step.candidates
        for rollout in candidate.rollouts
    )


def test_run_conditional_is_never_exceeds_total_length() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=[0.5, 0.5])
    result = run_conditional_is(
        backend,
        (),
        ConditionalEnergyConfig(candidate_count=2, rollout_count=2, block_size=2, total_length=5),
        lambda _prompt, generated: float(sum(generated)),
        SeedStream(17),
    )
    assert len(result.token_ids) == 5
    assert [len(step.selected.token_ids) for step in result.steps] == [2, 2, 1]


@pytest.mark.parametrize(
    "candidate_sampling,rollout_sampling",
    [
        (SamplingConfig(temperature=0.8), SamplingConfig()),
        (SamplingConfig(), SamplingConfig(top_p=0.9)),
        (SamplingConfig(), SamplingConfig(top_k=1)),
    ],
)
def test_conditional_is_rejects_sampling_policies_that_break_the_weight_formula(
    candidate_sampling, rollout_sampling
) -> None:
    with pytest.raises(ValueError):
        run_conditional_is(
            _backend(),
            (),
            ConditionalEnergyConfig(candidate_count=2, rollout_count=2, block_size=1, total_length=2),
            _reward,
            SeedStream(1),
            base_sampling=candidate_sampling,
            rollout_sampling=rollout_sampling,
        )
