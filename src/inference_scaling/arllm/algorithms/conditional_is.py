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

With ``retain_sequence`` a step keeps a complete sequence rather than only the
selected block: one completion of the selected candidate is kept with
probability proportional to its weight.  The next step's candidate 0 is that
sequence's next block, and its remaining completion counts as one of the
candidate's rollouts.  Each step is then a conditional SIR move that leaves the
target invariant, and every step ends with a complete sequence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, isfinite, log

from inference_scaling.arllm.algorithms.candidates import (
    sample_candidates,
    score_samples,
    validate_base_sampling,
    validate_rollout_sampling,
)
from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.shared.sampling.importance import (
    MonteCarloRolloutWeightProvider,
    RolloutObservation,
    logmeanexp,
)
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.stepwise import (
    StepwiseCandidate,
    categorical_index_from_uniform,
    normalize_log_weights,
    run_stepwise_generation,
    stepwise_generation_step,
)
from inference_scaling.shared.rewards.verifier import TokenBatchReward, TokenReward
from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    SequenceSample,
    TokenSequence,
)

RewardFunction = TokenReward
RewardBatchFunction = TokenBatchReward


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
    # Actual rollout-policy log-probabilities of token_ids.
    token_logprobs: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class ConditionalCandidate:
    token_ids: TokenSequence
    base_token_logprobs: tuple[float, ...]
    rollouts: tuple[RolloutEvaluation, ...]
    log_weight: float
    planned_rollout_count: int = 0
    log_weight_lower_bound: float | None = None
    log_weight_upper_bound: float | None = None


@dataclass(frozen=True, slots=True)
class ConditionalISStep:
    generated_length_before: int
    candidates: tuple[ConditionalCandidate, ...]
    selected_index: int
    rollout_evaluations_planned: int = 0
    rollout_evaluations_performed: int = 0
    rollout_evaluations_skipped: int = 0
    rollout_evaluation_batches: int = 0
    exact_early_stop: bool = False
    selection_invariant_verified: bool = False
    # Retained-sequence mode: whether candidate 0 continues the kept sequence,
    # which completion of the selected candidate is kept, and the sweep.
    retained_candidate: bool = False
    completion_index: int | None = None
    sweep: int = 0

    @property
    def selected(self) -> ConditionalCandidate:
        return self.candidates[self.selected_index]


@dataclass(frozen=True, slots=True)
class ConditionalISResult:
    prompt: TokenSequence
    token_ids: TokenSequence
    steps: tuple[ConditionalISStep, ...]


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
    rollout_design: str = "iid",
    rollout_index_offset: int = 0,
    retained: RolloutEvaluation | None = None,
) -> tuple[ConditionalCandidate, ...]:
    """Estimate each candidate's conditional weight with on/off-policy rollouts.

    ``retained`` is an already evaluated completion of candidate 0.  It counts
    as one of that candidate's ``rollout_count`` rollouts and is neither
    regenerated nor re-scored.
    """

    validate_rollout_sampling(rollout_sampling)
    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")
    if reward_temperature <= 0:
        raise ValueError("reward_temperature must be positive")
    if (reward is None) == (reward_batch is None):
        raise ValueError("provide exactly one of reward or reward_batch")
    if rollout_design not in {
        "iid",
        "scrambled_sobol",
        "arithmetic_lattice",
    }:
        raise ValueError("unknown rollout_design")
    if rollout_index_offset < 0:
        raise ValueError("rollout_index_offset must be non-negative")
    if (rollout_index_offset or retained is not None) and rollout_design != "iid":
        raise ValueError("staged or retained rollouts currently require iid rollouts")
    if rollout_design != "iid" and reward_batch is not None:
        raise ValueError(
            "randomized QMC rollouts require a fixed pointwise reward; "
            "batch-coupled rewards change when rollout dependence changes"
        )

    requests: list[GenerationRequest] = []
    request_candidates: list[int] = []
    rollout_prefixes: list[TokenSequence] = []
    terminal_candidates: set[int] = set()
    eos = rollout_sampling.eos_token_id
    retained_tokens = None if retained is None else retained.token_ids

    for candidate_index, candidate in enumerate(candidates):
        full_generated_candidate = generated_prefix + candidate.token_ids
        terminal = rollout_length == 0 or (
            eos is not None and candidate.token_ids[-1] == eos
        )
        kept = retained_tokens is not None and candidate_index == 0
        if kept and terminal == bool(retained_tokens):
            raise ValueError("a retained completion must be empty exactly after a terminal block")
        if terminal:
            if not kept:
                terminal_candidates.add(candidate_index)
            continue
        rollout_prefix = prompt + full_generated_candidate
        if rollout_design == "scrambled_sobol":
            from inference_scaling.experimental.shared.rqmc import (
                scrambled_sobol_uniforms,
            )

            token_uniforms = scrambled_sobol_uniforms(
                rollout_count,
                rollout_length,
                seed=seeds.derive(
                    "conditional_is",
                    step_index,
                    "candidate",
                    candidate_index,
                    "scrambled_sobol",
                ),
            )
        else:
            token_uniforms = (None,) * rollout_count
        if rollout_design == "arithmetic_lattice":
            from inference_scaling.experimental.shared.rqmc import (
                randomized_lattice_uniforms,
            )

            arithmetic_uniforms = randomized_lattice_uniforms(
                rollout_count,
                seed=seeds.derive(
                    "conditional_is",
                    step_index,
                    "candidate",
                    candidate_index,
                    "arithmetic_lattice",
                ),
            )
        else:
            arithmetic_uniforms = (None,) * rollout_count
        for rollout_index in range(int(kept), rollout_count):
            global_rollout_index = rollout_index_offset + rollout_index
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
                        global_rollout_index,
                    ),
                    request_id=(
                        "conditional-is:"
                        f"step:{step_index}:candidate:{candidate_index}:"
                        f"rollout:{global_rollout_index}"
                    ),
                    uniforms=token_uniforms[rollout_index],
                    arithmetic_uniform=arithmetic_uniforms[rollout_index],
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
            score_samples(
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
        list[tuple[TokenSequence, float, float, str, str, tuple[float, ...], TokenSequence]]
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
                (),
                generated,
            )
        )
    for candidate_index, sample, base_logprob in zip(
        request_candidates, samples, base_totals, strict=True
    ):
        generated = (
            generated_prefix + candidates[candidate_index].token_ids + sample.token_ids
        )
        proposal_logprob = sample.logprob
        pending_by_candidate[candidate_index].append(
            (
                sample.token_ids,
                base_logprob,
                proposal_logprob,
                sample.model_id,
                sample.policy_id,
                sample.token_logprobs,
                generated,
            )
        )

    pending = [item for group in pending_by_candidate for item in group]
    generated_sequences = [item[-1] for item in pending]
    if not pending:
        rewards: tuple[float, ...] = ()
    elif reward_batch is not None:
        rewards = tuple(
            float(value) for value in reward_batch(prompt, generated_sequences)
        )
        if len(rewards) != len(pending):
            raise ValueError("reward_batch returned an invalid number of rewards")
    else:
        assert reward is not None
        rewards = tuple(
            float(reward(prompt, generated)) for generated in generated_sequences
        )
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
    if retained is not None:
        by_candidate[0].append(retained)
    reward_index = 0
    for candidate_index, group in enumerate(pending_by_candidate):
        for token_ids, base_logprob, proposal_logprob, model_id, policy_id, token_logprobs, _ in group:
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
                    token_logprobs=token_logprobs,
                )
            )

    evaluated: list[ConditionalCandidate] = []
    for candidate_index, candidate in enumerate(candidates):
        evaluations = by_candidate[candidate_index]
        if not evaluations:
            raise RuntimeError(
                "each candidate must have at least one weight contribution"
            )
        candidate_log_weight = logmeanexp([item.log_weight for item in evaluations])
        evaluated.append(
            ConditionalCandidate(
                token_ids=candidate.token_ids,
                base_token_logprobs=candidate.token_logprobs,
                rollouts=tuple(evaluations),
                log_weight=candidate_log_weight,
                planned_rollout_count=len(evaluations),
                log_weight_lower_bound=candidate_log_weight,
                log_weight_upper_bound=candidate_log_weight,
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
        validate_base_sampling(self.base_sampling)
        remaining = self.config.total_length - len(state)
        if remaining <= 0:
            raise ValueError("generated prefix has already reached total_length")
        return sample_candidates(
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
        # Non-terminal candidates all have this length; EOS-terminated ones are shorter.
        candidate_length = min(self.config.block_size, remaining)
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
            rollout_design=self.config.rollout_design,
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


@dataclass(frozen=True, slots=True)
class RetainedSequence:
    """Complete sequence kept between retained-sequence steps.

    Every candidate of the next step shares the first ``fixed`` tokens.  An
    empty sequence means that no step has run yet.
    """

    token_ids: TokenSequence = ()
    token_logprobs: tuple[float, ...] = ()
    reward: float = 0.0
    fixed: int = 0
    sweep: int = 0


@dataclass(frozen=True, slots=True)
class RetainedCandidate:
    """An evaluated candidate and the completion kept if it is selected."""

    candidate: ConditionalCandidate
    completion_index: int


class RetainedSequenceAdapter:
    """Conditional SIR moves on a complete sequence, cut at successive block boundaries.

    Candidate 0 is the kept sequence's next block and its remaining completion
    is one of that candidate's rollouts.  Other candidates and rollouts are
    fresh base-policy samples.  Selecting a candidate by its mean weight and
    then one of its completions by weight selects a whole suffix in proportion
    to its reward weight.  A sweep ends when the kept sequence is fixed up to
    its end.
    """

    def __init__(
        self,
        *,
        base_backend: AutoregressiveBackend,
        rollout_backend: AutoregressiveBackend,
        prompt: TokenSequence,
        config: ConditionalISConfig,
        sampling: SamplingConfig,
        reward: RewardFunction | None,
        reward_batch: RewardBatchFunction | None = None,
    ) -> None:
        self.base_backend = base_backend
        self.rollout_backend = rollout_backend
        self.prompt = prompt
        self.config = config
        self.sampling = sampling
        self.reward = reward
        self.reward_batch = reward_batch

    @property
    def initial_state(self) -> RetainedSequence:
        return RetainedSequence()

    def is_terminal(self, state: RetainedSequence) -> bool:
        return bool(state.token_ids) and state.fixed >= len(state.token_ids)

    def _block_length(self, state: RetainedSequence) -> int:
        return min(self.config.block_size, self.config.total_length - state.fixed)

    def propose(
        self,
        state: RetainedSequence,
        step_index: int,
        seeds: SeedStream,
    ) -> Sequence[SequenceSample]:
        validate_base_sampling(self.sampling)
        prefix = self.prompt + state.token_ids[: state.fixed]
        length = self._block_length(state)
        proposals: list[SequenceSample] = []
        if state.token_ids:
            end = state.fixed + length
            block = state.token_ids[state.fixed : end]
            eos = self.sampling.eos_token_id
            proposals.append(
                SequenceSample(
                    prefix=prefix,
                    token_ids=block,
                    token_logprobs=state.token_logprobs[state.fixed : end],
                    policy_id=self.sampling.policy_id,
                    model_id=self.base_backend.model_id,
                    request_id=f"conditional-is:step:{step_index}:retained",
                    finish_reason="eos" if eos is not None and block[-1] == eos else "length",
                )
            )
        fresh = self.config.candidate_count - len(proposals)
        if fresh:
            proposals.extend(
                sample_candidates(
                    self.base_backend,
                    prefix,
                    fresh,
                    length,
                    self.sampling,
                    seeds,
                    step_index,
                    first_index=len(proposals),
                )
            )
        return proposals

    def evaluate(
        self,
        state: RetainedSequence,
        proposals: Sequence[SequenceSample],
        step_index: int,
        seeds: SeedStream,
    ) -> Sequence[StepwiseCandidate[RetainedCandidate]]:
        length = self._block_length(state)
        retained = None
        if state.token_ids:
            start = state.fixed + len(proposals[0].token_ids)
            completion_logprobs = state.token_logprobs[start:]
            logprob = float(sum(completion_logprobs))
            retained = RolloutEvaluation(
                token_ids=state.token_ids[start:],
                reward=state.reward,
                base_logprob=logprob,
                proposal_logprob=logprob,
                raw_log_importance_ratio=0.0,
                applied_log_importance_ratio=0.0,
                log_weight=state.reward / self.config.reward_temperature,
                proposal_model_id=self.rollout_backend.model_id,
                proposal_policy_id=self.sampling.policy_id,
                token_logprobs=completion_logprobs,
            )
        evaluated = estimate_conditional_weights(
            base_backend=self.base_backend,
            rollout_backend=self.rollout_backend,
            prompt=self.prompt,
            generated_prefix=state.token_ids[: state.fixed],
            candidates=proposals,
            rollout_length=self.config.total_length - state.fixed - length,
            rollout_count=self.config.rollout_count,
            base_sampling=self.sampling,
            rollout_sampling=self.sampling,
            reward_temperature=self.config.reward_temperature,
            importance_log_ratio_clip=self.config.importance_log_ratio_clip,
            apply_importance_correction=self.config.apply_importance_correction,
            reward=self.reward,
            seeds=seeds,
            step_index=step_index,
            reward_batch=self.reward_batch,
            retained=retained,
        )
        kept: list[StepwiseCandidate[RetainedCandidate]] = []
        for index, candidate in enumerate(evaluated):
            # Drawn for every candidate so the kept completion is independent
            # of which candidate the step selects.
            probabilities = normalize_log_weights(
                [rollout.log_weight for rollout in candidate.rollouts]
            )
            uniform = float(
                seeds.generator("conditional_is", step_index, "candidate", index, "completion").random()
            )
            kept.append(
                StepwiseCandidate(
                    RetainedCandidate(
                        candidate, categorical_index_from_uniform(probabilities, uniform)
                    ),
                    candidate.log_weight,
                )
            )
        return tuple(kept)

    def advance(
        self,
        state: RetainedSequence,
        selected: RetainedCandidate,
        step_index: int,
    ) -> RetainedSequence:
        del step_index
        candidate = selected.candidate
        completion = candidate.rollouts[selected.completion_index]
        prefix = state.token_ids[: state.fixed]
        token_ids = prefix + candidate.token_ids + completion.token_ids
        token_logprobs = (
            state.token_logprobs[: state.fixed]
            + candidate.base_token_logprobs
            + completion.token_logprobs
        )
        if len(token_logprobs) != len(token_ids):
            raise RuntimeError("retained sequence lost token log-probabilities")
        fixed = state.fixed + len(candidate.token_ids)
        sweep = state.sweep
        if fixed >= len(token_ids) and sweep + 1 < self.config.sweeps:
            fixed, sweep = 0, sweep + 1
        return RetainedSequence(token_ids, token_logprobs, completion.reward, fixed, sweep)


def _bounded_conditional_is_step(
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
    reward_batch: RewardBatchFunction | None,
) -> ConditionalISStep:
    """Evaluate rollout batches until the fixed categorical choice is known."""

    if reward is None or reward_batch is not None:
        raise ValueError(
            "exact rollout early stopping requires a fixed pointwise reward"
        )
    if config.rollout_log_weight_bounds is None:
        raise ValueError("exact rollout early stopping requires log-weight bounds")
    if config.rollout_design != "iid":
        raise ValueError("exact rollout early stopping currently requires iid rollouts")
    validate_base_sampling(base_sampling)
    remaining_length = config.total_length - len(generated_prefix)
    if remaining_length <= 0:
        raise ValueError("generated prefix has already reached total_length")
    candidate_length = min(config.block_size, remaining_length)
    proposals = sample_candidates(
        base_backend,
        prompt + generated_prefix,
        config.candidate_count,
        candidate_length,
        base_sampling,
        seeds,
        step_index,
    )
    rollout_length = max(0, remaining_length - candidate_length)
    eos = rollout_sampling.eos_token_id
    terminal = tuple(
        rollout_length == 0 or (eos is not None and proposal.token_ids[-1] == eos)
        for proposal in proposals
    )
    planned = tuple(
        1 if is_terminal else config.rollout_count for is_terminal in terminal
    )
    planned_total = sum(planned)
    selection_uniform = float(
        seeds.generator("conditional_is", step_index, "select").random()
    )
    lower_log_weight, upper_log_weight = config.rollout_log_weight_bounds
    try:
        minimum_contribution = exp(lower_log_weight)
        maximum_contribution = exp(upper_log_weight)
    except OverflowError as error:
        raise ValueError("rollout log-weight bounds cannot be exponentiated") from error
    if (
        not isfinite(minimum_contribution)
        or not isfinite(maximum_contribution)
        or minimum_contribution <= 0.0
    ):
        raise ValueError(
            "rollout log-weight bounds must map to finite positive weights"
        )

    collected: list[list[RolloutEvaluation]] = [[] for _ in proposals]
    lower_candidate_weights: list[float] = []
    upper_candidate_weights: list[float] = []
    invariant_index: int | None = None
    rollout_offset = 0
    evaluation_batches = 0
    while rollout_offset < config.rollout_count:
        batch_size = min(
            config.rollout_evaluation_batch_size,
            config.rollout_count - rollout_offset,
        )
        batch = estimate_conditional_weights(
            base_backend=base_backend,
            rollout_backend=rollout_backend,
            prompt=prompt,
            generated_prefix=generated_prefix,
            candidates=proposals,
            rollout_length=rollout_length,
            rollout_count=batch_size,
            base_sampling=base_sampling,
            rollout_sampling=rollout_sampling,
            reward_temperature=config.reward_temperature,
            importance_log_ratio_clip=config.importance_log_ratio_clip,
            apply_importance_correction=config.apply_importance_correction,
            reward=reward,
            seeds=seeds,
            step_index=step_index,
            rollout_design="iid",
            rollout_index_offset=rollout_offset,
        )
        evaluation_batches += 1
        for candidate_index, evaluated in enumerate(batch):
            if terminal[candidate_index]:
                if not collected[candidate_index]:
                    collected[candidate_index].append(evaluated.rollouts[0])
                continue
            for rollout in evaluated.rollouts:
                if not lower_log_weight <= rollout.log_weight <= upper_log_weight:
                    raise ValueError(
                        "observed rollout log-weight lies outside the declared bounds"
                    )
                collected[candidate_index].append(rollout)
        rollout_offset += batch_size

        lower_candidate_weights = []
        upper_candidate_weights = []
        for candidate_index, evaluations in enumerate(collected):
            contributions = [exp(item.log_weight) for item in evaluations]
            if terminal[candidate_index]:
                exact_weight = contributions[0]
                lower_candidate_weights.append(exact_weight)
                upper_candidate_weights.append(exact_weight)
                continue
            unseen = config.rollout_count - len(evaluations)
            lower_candidate_weights.append(
                (sum(contributions) + unseen * minimum_contribution)
                / config.rollout_count
            )
            upper_candidate_weights.append(
                (sum(contributions) + unseen * maximum_contribution)
                / config.rollout_count
            )
        from inference_scaling.experimental.shared.bounded_selection import (
            invariant_categorical_index,
        )

        invariant_index = invariant_categorical_index(
            lower_candidate_weights,
            upper_candidate_weights,
            uniform=selection_uniform,
        )
        if invariant_index is not None:
            break

    evaluated_candidates: list[ConditionalCandidate] = []
    for candidate_index, proposal in enumerate(proposals):
        evaluations = collected[candidate_index]
        if not evaluations:
            raise RuntimeError("bounded evaluation omitted a candidate")
        lower_weight = lower_candidate_weights[candidate_index]
        upper_weight = upper_candidate_weights[candidate_index]
        representative_weight = (lower_weight + upper_weight) / 2.0
        evaluated_candidates.append(
            ConditionalCandidate(
                token_ids=proposal.token_ids,
                base_token_logprobs=proposal.token_logprobs,
                rollouts=tuple(evaluations),
                log_weight=log(representative_weight),
                planned_rollout_count=planned[candidate_index],
                log_weight_lower_bound=log(lower_weight),
                log_weight_upper_bound=log(upper_weight),
            )
        )
    probabilities = normalize_log_weights(
        [candidate.log_weight for candidate in evaluated_candidates]
    )
    selected_index = categorical_index_from_uniform(
        probabilities,
        selection_uniform,
    )
    if invariant_index is not None and selected_index != invariant_index:
        raise RuntimeError(
            "bounded categorical proof disagrees with representative weights"
        )
    performed_total = sum(len(candidate.rollouts) for candidate in evaluated_candidates)
    skipped_total = planned_total - performed_total
    return ConditionalISStep(
        generated_length_before=len(generated_prefix),
        candidates=tuple(evaluated_candidates),
        selected_index=selected_index,
        rollout_evaluations_planned=planned_total,
        rollout_evaluations_performed=performed_total,
        rollout_evaluations_skipped=skipped_total,
        rollout_evaluation_batches=evaluation_batches,
        exact_early_stop=skipped_total > 0,
        selection_invariant_verified=skipped_total > 0 and invariant_index is not None,
    )


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
    if config.exact_rollout_early_stop:
        return _bounded_conditional_is_step(
            base_backend=base_backend,
            rollout_backend=rollout_backend,
            prompt=prompt,
            generated_prefix=generated_prefix,
            config=config,
            base_sampling=base_sampling,
            rollout_sampling=rollout_sampling,
            reward=reward,
            seeds=seeds,
            step_index=step_index,
            reward_batch=reward_batch,
        )
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
    evaluated_candidates = tuple(candidate.value for candidate in selection.candidates)
    performed = sum(len(candidate.rollouts) for candidate in evaluated_candidates)
    return ConditionalISStep(
        generated_length_before=len(generated_prefix),
        candidates=evaluated_candidates,
        selected_index=selection.selected_index,
        rollout_evaluations_planned=performed,
        rollout_evaluations_performed=performed,
        rollout_evaluation_batches=1,
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
    """Generate a sequence by repeatedly applying finite conditional-IS steps.

    With ``config.retain_sequence`` the steps are conditional SIR moves on a
    kept complete sequence; completions must then come from the base policy.
    """

    base_sampling = base_sampling or SamplingConfig()
    rollout_backend = rollout_backend or base_backend
    rollout_sampling = rollout_sampling or base_sampling
    validate_base_sampling(base_sampling)
    validate_rollout_sampling(rollout_sampling)
    if base_sampling.eos_token_id != rollout_sampling.eos_token_id:
        raise ValueError("candidate and rollout policies must agree on eos_token_id")

    if config.retain_sequence:
        if (
            rollout_backend.model_id != base_backend.model_id
            or rollout_sampling != base_sampling
        ):
            raise ValueError("retained sequences require on-policy base-model completions")
        kept = run_stepwise_generation(
            RetainedSequenceAdapter(
                base_backend=base_backend,
                rollout_backend=rollout_backend,
                prompt=prompt,
                config=config,
                sampling=base_sampling,
                reward=reward,
                reward_batch=reward_batch,
            ),
            seeds,
            selection_namespace=("conditional_is",),
        )
        retained_steps: list[ConditionalISStep] = []
        for selection in kept.steps:
            evaluated = tuple(item.value.candidate for item in selection.candidates)
            carried = bool(selection.state_before.token_ids)
            # The kept completion is reused, not evaluated again.
            fresh = sum(len(candidate.rollouts) for candidate in evaluated) - int(carried)
            retained_steps.append(
                ConditionalISStep(
                    generated_length_before=selection.state_before.fixed,
                    candidates=evaluated,
                    selected_index=selection.selected_index,
                    rollout_evaluations_planned=fresh,
                    rollout_evaluations_performed=fresh,
                    rollout_evaluation_batches=1,
                    retained_candidate=carried,
                    completion_index=selection.selected.value.completion_index,
                    sweep=selection.state_before.sweep,
                )
            )
        return ConditionalISResult(
            prompt=prompt,
            token_ids=kept.final_state.token_ids,
            steps=tuple(retained_steps),
        )

    if config.exact_rollout_early_stop:
        generated: TokenSequence = ()
        steps: list[ConditionalISStep] = []
        step_index = 0
        eos = base_sampling.eos_token_id
        while len(generated) < config.total_length and (
            eos is None or eos not in generated
        ):
            step = conditional_is_step(
                base_backend=base_backend,
                rollout_backend=rollout_backend,
                prompt=prompt,
                generated_prefix=generated,
                config=config,
                base_sampling=base_sampling,
                rollout_sampling=rollout_sampling,
                reward=reward,
                seeds=seeds,
                step_index=step_index,
                reward_batch=reward_batch,
            )
            generated += step.selected.token_ids
            if eos is not None and eos in generated:
                generated = generated[: generated.index(eos) + 1]
            steps.append(step)
            step_index += 1
        return ConditionalISResult(
            prompt=prompt,
            token_ids=generated,
            steps=tuple(steps),
        )

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
    steps: list[ConditionalISStep] = []
    for step in generic.steps:
        candidates = tuple(candidate.value for candidate in step.candidates)
        performed = sum(len(candidate.rollouts) for candidate in candidates)
        steps.append(
            ConditionalISStep(
                generated_length_before=len(step.state_before),
                candidates=candidates,
                selected_index=step.selected_index,
                rollout_evaluations_planned=performed,
                rollout_evaluations_performed=performed,
                rollout_evaluation_batches=1,
            )
        )
    return ConditionalISResult(
        prompt=prompt,
        token_ids=generic.final_state,
        steps=tuple(steps),
    )
