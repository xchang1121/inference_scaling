"""Archived block conditional importance sampling.

Superseded by :mod:`inference_scaling.arllm.algorithms.conditional_is`, which
keeps a complete sequence between steps.  Here each step commits only the
selected block and discards the completions that weighted it.  Kept to
reproduce results reported with it, including off-policy (small-proposal)
completions with the ``p_base / q`` correction, optional symmetric clipping of
that log-ratio and the uncorrected ablation.  Candidate blocks always come from
the base model; weights come from the mainline estimator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from inference_scaling.arllm.algorithms.candidates import (
    sample_candidates,
    validate_base_sampling,
    validate_rollout_sampling,
)
from inference_scaling.arllm.algorithms.conditional_is import (
    ConditionalCandidate,
    RewardBatchFunction,
    RewardFunction,
    estimate_conditional_weights,
)
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, SequenceSample, TokenSequence
from inference_scaling.shared.config import require_positive
from inference_scaling.shared.model.generation import DEFAULT_MAX_NEW_TOKENS
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.stepwise import (
    StepwiseCandidate,
    run_stepwise_generation,
)


@dataclass(frozen=True, slots=True)
class BlockConditionalISConfig:
    candidate_count: int = 4
    rollout_count: int = 4
    block_size: int = 16
    total_length: int = DEFAULT_MAX_NEW_TOKENS
    reward_temperature: float = 1.0
    importance_log_ratio_clip: float | None = None
    apply_importance_correction: bool = True

    def __post_init__(self) -> None:
        for name in ("candidate_count", "rollout_count", "block_size", "total_length"):
            require_positive(name, getattr(self, name))
        require_positive("reward_temperature", self.reward_temperature)
        if self.importance_log_ratio_clip is not None:
            require_positive("importance_log_ratio_clip", self.importance_log_ratio_clip)
            if not self.apply_importance_correction:
                raise ValueError(
                    "importance_log_ratio_clip requires apply_importance_correction=True"
                )
        if self.block_size > self.total_length:
            raise ValueError("block_size cannot exceed total_length")


@dataclass(frozen=True, slots=True)
class BlockConditionalISStep:
    generated_length_before: int
    candidates: tuple[ConditionalCandidate, ...]
    selected_index: int

    @property
    def selected(self) -> ConditionalCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class BlockConditionalISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[BlockConditionalISStep, ...]


class BlockConditionalISAdapter:
    """Commit the selected block; its completions only weight the candidates."""

    def __init__(
        self,
        *,
        base_backend: AutoregressiveBackend,
        rollout_backend: AutoregressiveBackend,
        prompt: TokenSequence,
        config: BlockConditionalISConfig,
        base_sampling: SamplingConfig,
        rollout_sampling: SamplingConfig,
        reward: RewardFunction | None,
        reward_batch: RewardBatchFunction | None = None,
    ) -> None:
        self.base_backend = base_backend
        self.rollout_backend = rollout_backend
        self.prompt = prompt
        self.config = config
        self.base_sampling = base_sampling
        self.rollout_sampling = rollout_sampling
        self.reward = reward
        self.reward_batch = reward_batch

    @property
    def initial_state(self) -> TokenSequence:
        return ()

    def is_terminal(self, state: TokenSequence) -> bool:
        eos = self.base_sampling.eos_token_id
        return len(state) >= self.config.total_length or (eos is not None and eos in state)

    def propose(
        self, state: TokenSequence, step_index: int, seeds: SeedStream
    ) -> Sequence[SequenceSample]:
        validate_base_sampling(self.base_sampling)
        return sample_candidates(
            self.base_backend,
            self.prompt + state,
            self.config.candidate_count,
            min(self.config.block_size, self.config.total_length - len(state)),
            self.base_sampling,
            seeds,
            step_index,
        )

    def evaluate(
        self,
        state: TokenSequence,
        proposals: Sequence[SequenceSample],
        step_index: int,
        seeds: SeedStream,
    ) -> Sequence[StepwiseCandidate[ConditionalCandidate]]:
        remaining = self.config.total_length - len(state)
        evaluated = estimate_conditional_weights(
            base_backend=self.base_backend,
            rollout_backend=self.rollout_backend,
            prompt=self.prompt,
            generated_prefix=state,
            candidates=proposals,
            # Non-terminal candidates all have this length; EOS-terminated ones are shorter.
            rollout_length=max(0, remaining - min(self.config.block_size, remaining)),
            rollout_count=self.config.rollout_count,
            base_sampling=self.base_sampling,
            rollout_sampling=self.rollout_sampling,
            reward_temperature=self.config.reward_temperature,
            importance_log_ratio_clip=self.config.importance_log_ratio_clip,
            apply_importance_correction=self.config.apply_importance_correction,
            reward=self.reward,
            seeds=seeds,
            step_index=step_index,
            reward_batch=self.reward_batch,
        )
        return tuple(StepwiseCandidate(candidate, candidate.log_weight) for candidate in evaluated)

    def advance(
        self, state: TokenSequence, selected: ConditionalCandidate, step_index: int
    ) -> TokenSequence:
        del step_index
        generated = state + selected.token_ids
        eos = self.base_sampling.eos_token_id
        if eos is not None and eos in generated:
            generated = generated[: generated.index(eos) + 1]
        return generated


def _adapter(
    base_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: BlockConditionalISConfig,
    reward: RewardFunction | None,
    base_sampling: SamplingConfig | None,
    rollout_backend: AutoregressiveBackend | None,
    rollout_sampling: SamplingConfig | None,
    reward_batch: RewardBatchFunction | None,
) -> BlockConditionalISAdapter:
    base_sampling = base_sampling or SamplingConfig()
    rollout_sampling = rollout_sampling or base_sampling
    validate_base_sampling(base_sampling)
    validate_rollout_sampling(rollout_sampling)
    if base_sampling.eos_token_id != rollout_sampling.eos_token_id:
        raise ValueError("candidate and rollout policies must agree on eos_token_id")
    return BlockConditionalISAdapter(
        base_backend=base_backend,
        rollout_backend=rollout_backend or base_backend,
        prompt=prompt,
        config=config,
        base_sampling=base_sampling,
        rollout_sampling=rollout_sampling,
        reward=reward,
        reward_batch=reward_batch,
    )


def run_block_conditional_is(
    base_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: BlockConditionalISConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    *,
    base_sampling: SamplingConfig | None = None,
    rollout_backend: AutoregressiveBackend | None = None,
    rollout_sampling: SamplingConfig | None = None,
    reward_batch: RewardBatchFunction | None = None,
) -> BlockConditionalISResult:
    """Generate a sequence by repeatedly committing the block chosen by conditional IS."""

    adapter = _adapter(
        base_backend, prompt, config, reward, base_sampling, rollout_backend, rollout_sampling, reward_batch,
    )
    generic = run_stepwise_generation(adapter, seeds, selection_namespace=("conditional_is",))
    return BlockConditionalISResult(
        prompt,
        generic.final_state,
        tuple(
            BlockConditionalISStep(
                len(step.state_before),
                tuple(candidate.value for candidate in step.candidates),
                step.selected_index,
            )
            for step in generic.steps
        ),
    )
