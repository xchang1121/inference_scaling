"""Conditional importance sampling.

Candidate blocks are always sampled from the base model in this module.  A
completion may be sampled on-policy or from a full-support off-policy proposal.
Only the completion suffix receives the ``p_base / q`` correction.  This is the
finite-candidate, finite-rollout sampling-importance-resampling algorithm used as
the foundation for the replay extensions.  Optional symmetric clipping of the
sequence log-ratio is recorded explicitly; it is a biased variance-control
setting, while the default ``None`` retains the exact importance ratio.  An
explicit uncorrected ablation skips target-model rescoring and instead estimates
each candidate's future reward weighting under the rollout proposal itself.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import isfinite

from inference_scaling.arllm.config import ConditionalISConfig, SamplingConfig
from inference_scaling.shared.importance import (
    MonteCarloRolloutWeightProvider,
    RolloutObservation,
    logmeanexp,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.stepwise import (
    StepwiseCandidate,
    run_stepwise_generation,
    stepwise_generation_step,
)
from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
    TokenSequence,
)

RewardFunction = Callable[[TokenSequence, TokenSequence], float]
RewardBatchFunction = Callable[
    [TokenSequence, Sequence[TokenSequence]], Sequence[float]
]


@dataclass(frozen=True, slots=True)
class RolloutEvaluation:
    token_ids: TokenSequence
    reward: float
    base_logprob: float | None
    proposal_logprob: float
    raw_log_importance_ratio: float | None
    applied_log_importance_ratio: float | None
    log_weight: float
    proposal_model_id: str
    proposal_policy_id: str


@dataclass(frozen=True, slots=True)
class ConditionalCandidate:
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    rollouts: tuple[RolloutEvaluation, ...]
    log_weight: float


@dataclass(frozen=True, slots=True)
class ConditionalISStep:
    generated_length_before: int
    candidates: tuple[ConditionalCandidate, ...]
    selected_index: int

    @property
    def selected(self) -> ConditionalCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class ConditionalISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[ConditionalISStep, ...]


def _validate_base_sampling(sampling: SamplingConfig) -> None:
    if sampling.top_p < 1 or sampling.top_k is not None:
        raise ValueError(
            "base candidates must use a full-support autoregressive policy: "
            "top_p=1 and top_k=None"
        )


def _validate_rollout_sampling(sampling: SamplingConfig) -> None:
    if sampling.top_p < 1 or sampling.top_k is not None:
        raise ValueError(
            "off-policy IS requires proposal support wherever the base weighted target is positive; "
            "hard top-k/top-p truncation is not accepted"
        )


def _score_samples(
    base_backend: AutoregressiveBackend,
    prefixes: Sequence[TokenSequence],
    samples: Sequence[SequenceSample],
    base_sampling: SamplingConfig,
) -> list[float]:
    requests = [
        ScoreRequest(prefix, (sample.token_ids,), base_sampling)
        for prefix, sample in zip(prefixes, samples, strict=True)
    ]
    token_scores = base_backend.score_batch(requests)
    if len(token_scores) != len(samples):
        raise RuntimeError("backend returned an invalid number of base scores")
    totals: list[float] = []
    for sample, scores in zip(samples, token_scores, strict=True):
        if len(scores) != len(sample.token_ids):
            raise RuntimeError("backend returned an invalid base token score shape")
        total = float(sum(scores))
        if not isfinite(total):
            raise ValueError("rollout proposal generated a completion outside base-model support")
        totals.append(total)
    return totals


def _sample_candidates(
    base_backend: AutoregressiveBackend,
    prefix: TokenSequence,
    count: int,
    block_length: int,
    sampling: SamplingConfig,
    seeds: SeedStream,
    step_index: int,
) -> list[SequenceSample]:
    requests = [
        GenerationRequest(
            prefix=prefix,
            max_new_tokens=block_length,
            sampling=sampling,
            seed=seeds.derive("conditional_is", step_index, "candidate", candidate_index),
            request_id=f"conditional-is:step:{step_index}:candidate:{candidate_index}",
        )
        for candidate_index in range(count)
    ]
    candidates = base_backend.sample_batch(requests)
    if len(candidates) != count:
        raise RuntimeError("backend returned an invalid number of candidates")
    for candidate in candidates:
        if not candidate.token_ids:
            raise RuntimeError("a candidate block must contain at least one token")
        if candidate.model_id != base_backend.model_id or candidate.policy_id != sampling.policy_id:
            raise RuntimeError("candidate was not sampled and scored by the requested base policy")
    return candidates


def estimate_conditional_weights(
    *,
    base_backend: AutoregressiveBackend,
    rollout_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    candidates: Sequence[SequenceSample],
    rollout_length: int,
    rollout_count: int,
    base_sampling: SamplingConfig,
    rollout_sampling: SamplingConfig,
    reward_temperature: float,
    importance_log_ratio_clip: float | None,
    apply_importance_correction: bool,
    reward: RewardFunction | None,
    seeds: SeedStream,
    step_index: int,
    reward_batch: RewardBatchFunction | None = None,
) -> tuple[ConditionalCandidate, ...]:
    """Estimate each candidate's conditional weight with on/off-policy rollouts."""

    _validate_rollout_sampling(rollout_sampling)
    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")
    if reward_temperature <= 0:
        raise ValueError("reward_temperature must be positive")
    if (reward is None) == (reward_batch is None):
        raise ValueError("provide exactly one of reward or reward_batch")

    requests: list[GenerationRequest] = []
    request_candidates: list[int] = []
    rollout_prefixes: list[TokenSequence] = []
    terminal_candidates: set[int] = set()
    eos = rollout_sampling.eos_token_id

    for candidate_index, candidate in enumerate(candidates):
        full_generated_candidate = generated_prefix + candidate.token_ids
        terminal = rollout_length == 0 or (
            eos is not None and candidate.token_ids[-1] == eos
        )
        if terminal:
            terminal_candidates.add(candidate_index)
            continue
        rollout_prefix = prompt + full_generated_candidate
        for rollout_index in range(rollout_count):
            requests.append(
                GenerationRequest(
                    prefix=rollout_prefix,
                    max_new_tokens=rollout_length,
                    sampling=rollout_sampling,
                    seed=seeds.derive(
                        "conditional_is",
                        step_index,
                        "candidate",
                        candidate_index,
                        "rollout",
                        rollout_index,
                    ),
                    request_id=(
                        "conditional-is:"
                        f"step:{step_index}:candidate:{candidate_index}:rollout:{rollout_index}"
                    ),
                )
            )
            request_candidates.append(candidate_index)
            rollout_prefixes.append(rollout_prefix)

    samples = rollout_backend.sample_batch(requests) if requests else []
    if len(samples) != len(requests):
        raise RuntimeError("backend returned an invalid number of rollouts")
    if rollout_backend is not base_backend:
        observe = getattr(base_backend, "observe_draft_samples", None)
        if callable(observe):
            observe(samples)
    rollout_is_base_policy = (
        rollout_backend.model_id == base_backend.model_id
        and rollout_sampling == base_sampling
    )
    if rollout_is_base_policy:
        base_totals: list[float | None] = [sample.logprob for sample in samples]
    elif apply_importance_correction:
        base_totals = (
            _score_samples(
                base_backend,
                rollout_prefixes,
                samples,
                base_sampling,
            )
            if samples
            else []
        )
    else:
        # This is a deliberate biased ablation, not an IS estimate of the base
        # continuation distribution.  Keep the score absent so diagnostics and
        # backend accounting cannot mistake it for an evaluated zero log-ratio.
        base_totals = [None for _ in samples]

    pending_by_candidate: list[
        list[tuple[TokenSequence, float, float, str, str, TokenSequence]]
    ] = [[] for _ in candidates]
    for candidate_index in terminal_candidates:
        generated = generated_prefix + candidates[candidate_index].token_ids
        pending_by_candidate[candidate_index].append(
            (
                (),
                0.0,
                0.0,
                rollout_backend.model_id,
                rollout_sampling.policy_id,
                generated,
            )
        )
    for candidate_index, sample, base_logprob in zip(
        request_candidates, samples, base_totals, strict=True
    ):
        generated = generated_prefix + candidates[candidate_index].token_ids + sample.token_ids
        proposal_logprob = sample.logprob
        pending_by_candidate[candidate_index].append(
            (
                sample.token_ids,
                base_logprob,
                proposal_logprob,
                sample.model_id,
                sample.policy_id,
                generated,
            )
        )

    pending = [item for group in pending_by_candidate for item in group]
    generated_sequences = [item[-1] for item in pending]
    if reward_batch is not None:
        rewards = tuple(float(value) for value in reward_batch(prompt, generated_sequences))
        if len(rewards) != len(pending):
            raise ValueError("reward_batch returned an invalid number of rewards")
    else:
        assert reward is not None
        rewards = tuple(float(reward(prompt, generated)) for generated in generated_sequences)
    if any(not isfinite(value) for value in rewards):
        raise ValueError("reward must be finite")

    importance_weights = MonteCarloRolloutWeightProvider[
        tuple[TokenSequence, str, str]
    ](
        reward_temperature=reward_temperature,
        correction="importance",
        log_ratio_clip=importance_log_ratio_clip,
    )
    reward_only_weights = MonteCarloRolloutWeightProvider[
        tuple[TokenSequence, str, str]
    ](
        reward_temperature=reward_temperature,
        correction="none",
    )
    by_candidate: list[list[RolloutEvaluation]] = [[] for _ in candidates]
    reward_index = 0
    for candidate_index, group in enumerate(pending_by_candidate):
        for token_ids, base_logprob, proposal_logprob, model_id, policy_id, _ in group:
            reward_value = rewards[reward_index]
            reward_index += 1
            observation = RolloutObservation(
                reward=reward_value,
                target_logprob=base_logprob,
                proposal_logprob=proposal_logprob,
                payload=(token_ids, model_id, policy_id),
            )
            weighted = (
                importance_weights.weight(observation)
                if base_logprob is not None
                else reward_only_weights.weight(observation)
            )
            by_candidate[candidate_index].append(
                RolloutEvaluation(
                    token_ids=token_ids,
                    reward=reward_value,
                    base_logprob=base_logprob,
                    proposal_logprob=proposal_logprob,
                    raw_log_importance_ratio=weighted.raw_log_importance_ratio,
                    applied_log_importance_ratio=weighted.applied_log_importance_ratio,
                    log_weight=weighted.log_weight,
                    proposal_model_id=model_id,
                    proposal_policy_id=policy_id,
                )
            )

    evaluated: list[ConditionalCandidate] = []
    for candidate_index, candidate in enumerate(candidates):
        evaluations = by_candidate[candidate_index]
        if not evaluations:
            raise RuntimeError("each candidate must have at least one weight contribution")
        evaluated.append(
            ConditionalCandidate(
                token_ids=candidate.token_ids,
                base_token_logprobs=candidate.token_logprobs,
                rollouts=tuple(evaluations),
                log_weight=logmeanexp([item.log_weight for item in evaluations]),
            )
        )
    return tuple(evaluated)


class AutoregressiveStepwiseAdapter:
    """Expose conditional AR generation through the common stepwise protocol."""

    def __init__(
        self,
        *,
        base_backend: AutoregressiveBackend,
        rollout_backend: AutoregressiveBackend,
        prompt: TokenSequence,
        config: ConditionalISConfig,
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
        return len(state) >= self.config.total_length or (
            eos is not None and eos in state
        )

    def propose(
        self,
        state: TokenSequence,
        step_index: int,
        seeds: SeedStream,
    ) -> Sequence[SequenceSample]:
        _validate_base_sampling(self.base_sampling)
        remaining = self.config.total_length - len(state)
        if remaining <= 0:
            raise ValueError("generated prefix has already reached total_length")
        return _sample_candidates(
            self.base_backend,
            self.prompt + state,
            self.config.candidate_count,
            min(self.config.block_size, remaining),
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
        candidate_length = len(proposals[0].token_ids)
        evaluated = estimate_conditional_weights(
            base_backend=self.base_backend,
            rollout_backend=self.rollout_backend,
            prompt=self.prompt,
            generated_prefix=state,
            candidates=proposals,
            rollout_length=max(0, remaining - candidate_length),
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
        return tuple(
            StepwiseCandidate(candidate, candidate.log_weight)
            for candidate in evaluated
        )

    def advance(
        self,
        state: TokenSequence,
        selected: ConditionalCandidate,
        step_index: int,
    ) -> TokenSequence:
        del step_index
        generated = state + selected.token_ids
        eos = self.base_sampling.eos_token_id
        if eos is not None and eos in generated:
            generated = generated[: generated.index(eos) + 1]
        return generated


def conditional_is_step(
    *,
    base_backend: AutoregressiveBackend,
    rollout_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    config: ConditionalISConfig,
    base_sampling: SamplingConfig,
    rollout_sampling: SamplingConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    step_index: int,
    reward_batch: RewardBatchFunction | None = None,
) -> ConditionalISStep:
    adapter = AutoregressiveStepwiseAdapter(
        base_backend=base_backend,
        rollout_backend=rollout_backend,
        prompt=prompt,
        config=config,
        base_sampling=base_sampling,
        rollout_sampling=rollout_sampling,
        reward=reward,
        reward_batch=reward_batch,
    )
    selection = stepwise_generation_step(
        adapter,
        generated_prefix,
        step_index,
        seeds,
        selection_namespace=("conditional_is",),
    )
    return ConditionalISStep(
        generated_length_before=len(generated_prefix),
        candidates=tuple(candidate.value for candidate in selection.candidates),
        selected_index=selection.selected_index,
    )


def run_conditional_is(
    base_backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: ConditionalISConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    *,
    base_sampling: SamplingConfig | None = None,
    rollout_backend: AutoregressiveBackend | None = None,
    rollout_sampling: SamplingConfig | None = None,
    reward_batch: RewardBatchFunction | None = None,
) -> ConditionalISResult:
    """Generate a sequence by repeatedly applying finite conditional-IS steps."""

    base_sampling = base_sampling or SamplingConfig()
    rollout_backend = rollout_backend or base_backend
    rollout_sampling = rollout_sampling or base_sampling
    _validate_base_sampling(base_sampling)
    _validate_rollout_sampling(rollout_sampling)
    if base_sampling.eos_token_id != rollout_sampling.eos_token_id:
        raise ValueError("candidate and rollout policies must agree on eos_token_id")

    adapter = AutoregressiveStepwiseAdapter(
        base_backend=base_backend,
        rollout_backend=rollout_backend,
        prompt=prompt,
        config=config,
        base_sampling=base_sampling,
        rollout_sampling=rollout_sampling,
        reward=reward,
        reward_batch=reward_batch,
    )
    generic = run_stepwise_generation(
        adapter,
        seeds,
        selection_namespace=("conditional_is",),
    )
    steps = tuple(
        ConditionalISStep(
            generated_length_before=len(step.state_before),
            candidates=tuple(candidate.value for candidate in step.candidates),
            selected_index=step.selected_index,
        )
        for step in generic.steps
    )
    return ConditionalISResult(prompt=prompt, token_ids=generic.final_state, steps=steps)
