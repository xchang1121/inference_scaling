"""Whole-continuation independence MH for dLLM reward targets."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import exp, log

from inference_scaling.dllm.config import DiffusionMHConfig, DiffusionSamplingConfig
from inference_scaling.dllm.types import DiffusionBackend, DiffusionGenerationRequest, DiffusionSample
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenSequence

DiffusionRewardFunction = Callable[[TokenSequence, TokenSequence], float]
DiffusionRewardBatchFunction = Callable[
    [TokenSequence, Sequence[TokenSequence]], Sequence[float]
]


@dataclass(frozen=True, slots=True)
class DiffusionMHStep:
    update: int
    proposal: DiffusionSample
    proposal_reward: float
    previous_reward: float
    acceptance_probability: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class DiffusionMHResult:
    prompt: TokenSequence
    initial: DiffusionSample
    initial_reward: float
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
    requests = [
        DiffusionGenerationRequest(
            prefix=prompt,
            generation_length=config.total_length,
            sampling=sampling,
            seed=seeds.derive("dllm-mh", draw),
            request_id=f"dllm-mh:draw:{draw}",
        )
        for draw in range(config.updates + 1)
    ]
    samples = backend.sample_batch(requests)
    if len(samples) != len(requests):
        raise RuntimeError("backend returned an invalid number of MH proposals")
    continuations = [sample.token_ids for sample in samples]
    if reward_batch is not None:
        reward_values = [float(value) for value in reward_batch(prompt, continuations)]
    else:
        assert reward is not None
        reward_values = [float(reward(prompt, continuation)) for continuation in continuations]
    if len(reward_values) != len(samples):
        raise RuntimeError("reward evaluator returned an invalid number of values")

    current = samples[0]
    current_reward = reward_values[0]
    initial = current
    initial_reward = current_reward
    steps: list[DiffusionMHStep] = []
    for update, (proposal, proposal_reward) in enumerate(
        zip(samples[1:], reward_values[1:], strict=True), start=1
    ):
        log_acceptance = min(0.0, (proposal_reward - current_reward) / config.reward_temperature)
        acceptance_probability = exp(log_acceptance)
        uniform = float(seeds.generator("dllm-mh", "accept", update).random())
        accepted = log(max(uniform, float.fromhex("0x1.0p-1022"))) <= log_acceptance
        previous_reward = current_reward
        if accepted:
            current = proposal
            current_reward = proposal_reward
        steps.append(
            DiffusionMHStep(
                update=update,
                proposal=proposal,
                proposal_reward=proposal_reward,
                previous_reward=previous_reward,
                acceptance_probability=acceptance_probability,
                accepted=accepted,
            )
        )
    return DiffusionMHResult(
        prompt=prompt,
        initial=initial,
        initial_reward=initial_reward,
        steps=tuple(steps),
        final=current,
        final_reward=current_reward,
    )


__all__ = ["DiffusionMHResult", "DiffusionMHStep", "run_diffusion_reward_mh"]
