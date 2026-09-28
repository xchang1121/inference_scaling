"""Whole-continuation independence MH for dLLM reward targets.

Every proposal is a complete output drawn independently of the chain state, so
all of them are drawn in one batch. From the base sampler alone the base
trajectory density cancels the proposal density, and the acceptance needs
rewards but no dLLM likelihood. With ``history_probability > 0`` the proposal is
a frozen defensive mixture of the base sampler and a cache of earlier exact-policy
trajectories; its forward and reverse probabilities then enter the Hastings ratio,
which needs exact trajectory probabilities.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, log
from typing import Hashable

import numpy as np

from inference_scaling.dllm.algorithms.config import DiffusionMHConfig
from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.types import DiffusionBackend, DiffusionGenerationRequest, DiffusionSample
from inference_scaling.shared.sampling.mh import decide_metropolis_hastings
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import TokenBatchReward, TokenSequence


def _trajectory_key(sample: DiffusionSample) -> Hashable:
    return sample.token_ids, tuple((step.block_index, step.step_index, step.positions, step.token_ids)
                                   for step in sample.trace)


@dataclass(frozen=True, slots=True)
class DiffusionMHStep:
    update: int
    proposal_source: str
    accepted: bool


@dataclass(frozen=True, slots=True)
class DiffusionMHResult:
    steps: tuple[DiffusionMHStep, ...]
    final: DiffusionSample
    final_reward: float

    @property
    def acceptance_rate(self) -> float:
        return sum(step.accepted for step in self.steps) / len(self.steps) if self.steps else 0.0

    @property
    def history_draws(self) -> int:
        return sum(step.proposal_source == "history" for step in self.steps)


def run_diffusion_reward_mh(
    *,
    backend: DiffusionBackend,
    prompt: TokenSequence,
    config: DiffusionMHConfig,
    sampling: DiffusionSamplingConfig,
    reward: TokenBatchReward,
    seed: int,
    history: Sequence[DiffusionSample] = (),
    history_probability: float = 0.0,
) -> DiffusionMHResult:
    """Independence MH toward ``base * exp(reward / tau)``, optionally mixing a frozen trajectory cache."""

    sampling.validate_generation_length(config.total_length)
    if not 0 <= history_probability < 1:
        raise ValueError("history_probability must lie in [0, 1)")
    if history_probability > 0:
        if not history:
            raise ValueError("positive history probability requires a frozen cache")
        if not sampling.has_exact_trajectory_density:
            raise ValueError("a history mixture requires exact trajectory probabilities")
    for sample in history:
        if (sample.prefix != prompt or sample.trajectory_logprob is None or sample.policy_id != sampling.policy_id
                or sample.model_id != backend.model_id):
            raise ValueError("cached trajectories must match the prompt and exact policy")
    seeds = SeedStream(seed)
    cached = [False] + [bool(value) for value in
                        seeds.generator("dllm-mh", "sources").random(config.updates) < history_probability]
    # Proposals do not depend on the chain state, so all base draws go in one batch.
    requests = [DiffusionGenerationRequest(prefix=prompt, generation_length=config.total_length, sampling=sampling,
                                           seed=seeds.derive("dllm-mh", draw), request_id=f"dllm-mh:draw:{draw}",
                                           stop_at_eos=True)
                for draw, from_history in enumerate(cached) if not from_history]
    outputs = backend.sample_batch(requests)
    if len(outputs) != len(requests):
        raise RuntimeError("backend returned an invalid number of MH proposals")
    drawn = iter(outputs)
    samples = [history[int(seeds.generator("dllm-mh", "history", draw).integers(len(history)))] if from_history
               else next(drawn) for draw, from_history in enumerate(cached)]
    rewards = [float(value) for value in reward(prompt, [sample.token_ids for sample in samples])]
    if len(rewards) != len(samples) or any(not isfinite(value) for value in rewards):
        raise ValueError("reward must return one finite value per proposal")
    counts = Counter(_trajectory_key(sample) for sample in history)

    def excess(sample: DiffusionSample) -> float:
        """``log p - log q`` of a proposal: zero from the base sampler alone, whose density cancels."""

        if not history_probability:
            return 0.0
        if sample.trajectory_logprob is None:
            raise ValueError("an MH proposal omitted its trajectory probability")
        count = counts.get(_trajectory_key(sample), 0)
        mixture = np.logaddexp(log(1 - history_probability) + sample.trajectory_logprob,
                               log(history_probability) + log(count / len(history)) if count else float("-inf"))
        return float(sample.trajectory_logprob - mixture)

    current, current_reward = samples[0], rewards[0]
    steps: list[DiffusionMHStep] = []
    for update in range(1, len(samples)):
        decision = decide_metropolis_hastings(
            current_target_log_density=current_reward / config.reward_temperature + excess(current),
            proposed_target_log_density=rewards[update] / config.reward_temperature + excess(samples[update]),
            uniform=float(seeds.generator("dllm-mh", "accept", update).random()),
        )
        if decision.accepted:
            current, current_reward = samples[update], rewards[update]
        steps.append(DiffusionMHStep(update, "history" if cached[update] else "base", decision.accepted))
    return DiffusionMHResult(tuple(steps), current, current_reward)


__all__ = ["DiffusionMHResult", "DiffusionMHStep", "run_diffusion_reward_mh"]
