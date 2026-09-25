"""Whole-continuation independence MH for dLLM reward targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from inference_scaling.dllm.algorithms.config import DiffusionMHConfig
from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.types import DiffusionBackend, DiffusionGenerationRequest, DiffusionSample
from inference_scaling.shared.sampling.mh import decide_metropolis_hastings
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenBatchReward, TokenReward, TokenSequence

DiffusionRewardFunction = TokenReward
DiffusionRewardBatchFunction = TokenBatchReward


def _mh_requests(
    prompt: TokenSequence,
    config: DiffusionMHConfig,
    sampling: DiffusionSamplingConfig,
    seeds: SeedStream,
) -> list[DiffusionGenerationRequest]:
    return [
        DiffusionGenerationRequest(
            prefix=prompt,
            generation_length=config.total_length,
            sampling=sampling,
            seed=seeds.derive("dllm-mh", draw),
            request_id=f"dllm-mh:draw:{draw}",
            stop_at_eos=True,
        )
        for draw in range(config.updates + 1)
    ]


def _evaluate_mh_rewards(
    prompt: TokenSequence,
    samples: Sequence[DiffusionSample],
    reward: DiffusionRewardFunction | None,
    reward_batch: DiffusionRewardBatchFunction | None,
) -> list[float]:
    continuations = [sample.token_ids for sample in samples]
    if reward_batch is not None:
        values = [float(value) for value in reward_batch(prompt, continuations)]
    else:
        assert reward is not None
        values = [float(reward(prompt, continuation)) for continuation in continuations]
    if len(values) != len(samples):
        raise RuntimeError("reward evaluator returned an invalid number of values")
    if any(not isfinite(value) for value in values):
        raise ValueError("reward values must be finite")
    return values


@dataclass(frozen=True, slots=True)
class DiffusionMHStep:
    update: int
    accepted: bool


@dataclass(frozen=True, slots=True)
class DiffusionMHResult:
    steps: tuple[DiffusionMHStep, ...]
    final: DiffusionSample
    final_reward: float

    @property
    def acceptance_rate(self) -> float:
        if not self.steps:
            return 0.0
        return sum(step.accepted for step in self.steps) / len(self.steps)


def run_diffusion_reward_mh(
    *,
    backend: DiffusionBackend,
    prompt: TokenSequence,
    config: DiffusionMHConfig,
    sampling: DiffusionSamplingConfig,
    reward: DiffusionRewardFunction | None = None,
    seed: int = 0,
    reward_batch: DiffusionRewardBatchFunction | None = None,
) -> DiffusionMHResult:
    """Run independence MH with proposals drawn from the base dLLM sampler.

    For the target proportional to ``base_trajectory * exp(reward / tau)``, the
    base trajectory density cancels the independence-proposal density.  The
    acceptance probability therefore needs rewards but no dLLM likelihood.
    """

    if (reward is None) == (reward_batch is None):
        raise ValueError("provide exactly one of reward or reward_batch")
    sampling.validate_generation_length(config.total_length)
    seeds = SeedStream(seed)
    # Proposals do not depend on the chain state, so all of them are drawn in one batch.
    requests = _mh_requests(prompt, config, sampling, seeds)
    samples = backend.sample_batch(requests)
    if len(samples) != len(requests):
        raise RuntimeError("backend returned an invalid number of MH proposals")
    reward_values = _evaluate_mh_rewards(prompt, samples, reward, reward_batch)

    current = samples[0]
    current_reward = reward_values[0]
    steps: list[DiffusionMHStep] = []
    for update, (proposal, proposal_reward) in enumerate(
        zip(samples[1:], reward_values[1:], strict=True), start=1
    ):
        uniform = float(seeds.generator("dllm-mh", "accept", update).random())
        decision = decide_metropolis_hastings(
            current_target_log_density=current_reward / config.reward_temperature,
            proposed_target_log_density=proposal_reward / config.reward_temperature,
            uniform=uniform,
        )
        if decision.accepted:
            current = proposal
            current_reward = proposal_reward
        steps.append(DiffusionMHStep(update, decision.accepted))
    return DiffusionMHResult(tuple(steps), current, current_reward)


__all__ = ["DiffusionMHResult", "DiffusionMHStep", "run_diffusion_reward_mh"]
