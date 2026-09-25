"""Blockwise reward-weighted importance sampling for masked dLLMs.

Each step draws ``candidate_count`` next blocks from the base policy, completes
every candidate ``rollout_count`` times with the same policy and keeps one
candidate in proportion to the mean ``exp(reward / temperature)`` of its
completions; the completions are then discarded. With finite counts this is a
blockwise SIR approximation of ``p(y) exp(r(y) / temperature)``. Generation stops
after a block of EOS; a candidate that ends there completes the sequence.
"""

from __future__ import annotations

from dataclasses import dataclass

from inference_scaling.dllm.algorithms.config import DiffusionISConfig
from inference_scaling.dllm.config import DiffusionSamplingConfig, diffusion_decision_stage_lengths
from inference_scaling.dllm.types import DiffusionBackend, DiffusionGenerationRequest
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.importance import (
    categorical_index_from_uniform,
    logmeanexp,
    normalize_log_weights,
)
from inference_scaling.shared.types import TokenBatchReward, TokenReward, TokenSequence


@dataclass(frozen=True, slots=True)
class DiffusionRolloutEvaluation:
    token_ids: TokenSequence
    reward: float
    log_weight: float


@dataclass(frozen=True, slots=True)
class DiffusionConditionalCandidate:
    token_ids: TokenSequence
    rollouts: tuple[DiffusionRolloutEvaluation, ...]
    log_weight: float


@dataclass(frozen=True, slots=True)
class DiffusionConditionalISStep:
    generated_length_before: int
    candidates: tuple[DiffusionConditionalCandidate, ...]
    probabilities: tuple[float, ...]
    selected_index: int

    @property
    def selected(self) -> DiffusionConditionalCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class DiffusionConditionalISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[DiffusionConditionalISStep, ...]


def run_conditional_diffusion_is(
    *,
    backend: DiffusionBackend,
    prompt: TokenSequence,
    config: DiffusionISConfig,
    sampling: DiffusionSamplingConfig,
    seed: int,
    reward: TokenReward | None = None,
    reward_batch: TokenBatchReward | None = None,
) -> DiffusionConditionalISResult:
    """Blockwise IS; the last block's candidates are complete and have one empty completion."""

    if (reward is None) == (reward_batch is None):
        raise ValueError("provide exactly one of reward or reward_batch")
    seeds = SeedStream(seed)
    stages = diffusion_decision_stage_lengths(
        total_length=config.total_length,
        decision_block_size=config.block_size, sampling=sampling,
    )
    state: TokenSequence = ()
    steps: list[DiffusionConditionalISStep] = []
    for step_index, length in enumerate(stages):
        candidates = backend.sample_batch([
            DiffusionGenerationRequest(
                prefix=prompt + state, generation_length=length, sampling=sampling,
                seed=seeds.derive("dllm-is", step_index, "candidate", index),
                request_id=f"dllm-is:step:{step_index}:candidate:{index}", stop_at_eos=True,
            )
            for index in range(config.candidate_count)
        ])
        if len(candidates) != config.candidate_count:
            raise RuntimeError("backend returned an invalid number of dLLM candidates")
        remaining = config.total_length - len(state) - length
        terminal = [not remaining or candidate.finish_reason == "eos" for candidate in candidates]
        # (owner, index of its rollout request); a terminal candidate has one empty completion.
        slots: list[tuple[int, int | None]] = []
        requests: list[DiffusionGenerationRequest] = []
        for owner, candidate in enumerate(candidates):
            if terminal[owner]:
                slots.append((owner, None))
                continue
            for rollout in range(config.rollout_count):
                slots.append((owner, len(requests)))
                requests.append(DiffusionGenerationRequest(
                    prefix=prompt + state + candidate.token_ids, generation_length=remaining, sampling=sampling,
                    seed=seeds.derive("dllm-is", step_index, "rollout", owner, rollout),
                    request_id=f"dllm-is:step:{step_index}:candidate:{owner}:rollout:{rollout}", stop_at_eos=True,
                ))
        samples = backend.sample_batch(requests) if requests else []
        if len(samples) != len(requests):
            raise RuntimeError("backend returned an invalid number of dLLM rollouts")
        owners = [owner for owner, _ in slots]
        completions: list[TokenSequence] = [() if index is None else samples[index].token_ids for _, index in slots]
        sequences = [state + candidates[owner].token_ids + tokens for owner, tokens in zip(owners, completions, strict=True)]
        if reward_batch is not None:
            rewards = [float(value) for value in reward_batch(prompt, sequences)]
        else:
            assert reward is not None
            rewards = [float(reward(prompt, sequence)) for sequence in sequences]
        if len(rewards) != len(sequences):
            raise RuntimeError("reward evaluator returned an invalid number of values")
        grouped: list[list[DiffusionRolloutEvaluation]] = [[] for _ in candidates]
        for owner, tokens, value in zip(owners, completions, rewards, strict=True):
            grouped[owner].append(DiffusionRolloutEvaluation(tokens, value, value / config.reward_temperature))
        evaluated = tuple(
            DiffusionConditionalCandidate(candidate.token_ids, tuple(group), logmeanexp([item.log_weight for item in group]))
            for candidate, group in zip(candidates, grouped, strict=True)
        )
        probabilities = normalize_log_weights([candidate.log_weight for candidate in evaluated])
        selected = categorical_index_from_uniform(
            probabilities, float(seeds.generator("dllm-is", step_index, "select").random()),
        )
        steps.append(DiffusionConditionalISStep(len(state), evaluated, probabilities, selected))
        state += evaluated[selected].token_ids
        if terminal[selected]:
            break
    return DiffusionConditionalISResult(prompt, state, tuple(steps))


__all__ = [
    "DiffusionConditionalCandidate",
    "DiffusionConditionalISResult",
    "DiffusionConditionalISStep",
    "DiffusionRolloutEvaluation",
    "run_conditional_diffusion_is",
]
