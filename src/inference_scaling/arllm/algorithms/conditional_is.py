"""Conditional importance sampling on a kept complete sequence.

The target is ``p(y | x) exp(r(x, y) / tau)`` for a reward ``r`` of the complete
sequence.  Each step cuts the kept sequence at the next block boundary.
Candidate 0 is the kept sequence's next block, and the rest of the kept
sequence counts as one of that candidate's completions; the other candidates,
and the other completions of candidate 0, are fresh base-policy samples.  Every
completion runs to EOS or the length limit and is scored as a complete sequence.
A candidate is selected with probability proportional to the mean
``exp(r / tau)`` of its completions, and one of its completions is kept with
probability proportional to its own weight, so a step selects a whole suffix in
proportion to its reward weight.  This is a conditional SIR move: given the
prefix before the cut it leaves the target invariant, and every step ends with a
complete sequence.  The kept completion's reward is reused, so the reward must
depend only on the sequence it scores.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from inference_scaling.arllm.algorithms.candidates import sample_candidates, validate_base_sampling
from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.shared.sampling.importance import logmeanexp
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.stepwise import (
    StepwiseCandidate,
    StepwiseSelection,
    categorical_index_from_uniform,
    normalize_log_weights,
    run_stepwise_generation,
    stepwise_generation_step,
)
from inference_scaling.shared.types import TokenBatchReward, TokenReward
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
    """One base-policy completion of a candidate; its log weight is reward / temperature."""

    token_ids: TokenSequence
    reward: float
    log_weight: float
    # Base-policy log-probabilities of token_ids.
    token_logprobs: tuple[float, ...] = ()


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
    # Completion of the selected candidate kept in the sequence.
    completion_index: int
    # Whether candidate 0 continues the kept sequence (false on the first step).
    retained_candidate: bool
    # Completions generated and scored in this step; a reused one is excluded.
    rollout_evaluations_performed: int

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
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    generated_prefix: TokenSequence,
    candidates: Sequence[SequenceSample],
    rollout_length: int,
    rollout_count: int,
    sampling: SamplingConfig,
    reward_temperature: float,
    reward: RewardFunction | None,
    seeds: SeedStream,
    step_index: int,
    reward_batch: RewardBatchFunction | None = None,
    retained: RolloutEvaluation | None = None,
) -> tuple[ConditionalCandidate, ...]:
    """Estimate each candidate's conditional weight with base-policy completions.

    ``retained`` is an already evaluated completion of candidate 0.  It counts
    as one of that candidate's ``rollout_count`` rollouts and is neither
    regenerated nor re-scored.
    """

    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")
    if reward_temperature <= 0:
        raise ValueError("reward_temperature must be positive")
    if (reward is None) == (reward_batch is None):
        raise ValueError("provide exactly one of reward or reward_batch")

    requests: list[GenerationRequest] = []
    request_candidates: list[int] = []
    terminal_candidates: set[int] = set()
    eos = sampling.eos_token_id
    retained_tokens = None if retained is None else retained.token_ids

    for candidate_index, candidate in enumerate(candidates):
        terminal = rollout_length == 0 or (
            eos is not None and candidate.token_ids[-1] == eos
        )
        kept = retained_tokens is not None and candidate_index == 0
        if kept and retained_tokens and terminal:
            raise ValueError("a retained completion cannot follow a terminal block")
        if kept and not retained_tokens:
            # The kept sequence ends with this block: at EOS, the length limit
            # or a stop sequence of a scoped backend.
            continue
        if terminal:
            terminal_candidates.add(candidate_index)
            continue
        rollout_prefix = prompt + generated_prefix + candidate.token_ids
        for rollout_index in range(int(kept), rollout_count):
            requests.append(
                GenerationRequest(
                    prefix=rollout_prefix,
                    max_new_tokens=rollout_length,
                    sampling=sampling,
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
                        f"step:{step_index}:candidate:{candidate_index}:"
                        f"rollout:{rollout_index}"
                    ),
                )
            )
            request_candidates.append(candidate_index)

    samples = backend.sample_batch(requests) if requests else []
    if len(samples) != len(requests):
        raise RuntimeError("backend returned an invalid number of rollouts")

    # (completion tokens, their log-probabilities, the generated sequence the reward scores)
    pending_by_candidate: list[list[tuple[TokenSequence, tuple[float, ...], TokenSequence]]] = [
        [] for _ in candidates
    ]
    for candidate_index in terminal_candidates:
        pending_by_candidate[candidate_index].append(
            ((), (), generated_prefix + candidates[candidate_index].token_ids)
        )
    for candidate_index, sample in zip(request_candidates, samples, strict=True):
        generated = generated_prefix + candidates[candidate_index].token_ids + sample.token_ids
        pending_by_candidate[candidate_index].append((sample.token_ids, sample.token_logprobs, generated))

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

    by_candidate: list[list[RolloutEvaluation]] = [[] for _ in candidates]
    if retained is not None:
        by_candidate[0].append(retained)
    reward_values = iter(rewards)
    for candidate_index, group in enumerate(pending_by_candidate):
        for token_ids, token_logprobs, _ in group:
            reward_value = next(reward_values)
            by_candidate[candidate_index].append(
                RolloutEvaluation(token_ids, reward_value, reward_value / reward_temperature, token_logprobs)
            )

    evaluated: list[ConditionalCandidate] = []
    for candidate_index, candidate in enumerate(candidates):
        evaluations = by_candidate[candidate_index]
        if not evaluations:
            raise RuntimeError(
                "each candidate must have at least one weight contribution"
            )
        evaluated.append(
            ConditionalCandidate(
                token_ids=candidate.token_ids,
                base_token_logprobs=candidate.token_logprobs,
                rollouts=tuple(evaluations),
                log_weight=logmeanexp([item.log_weight for item in evaluations]),
            )
        )
    return tuple(evaluated)


@dataclass(frozen=True, slots=True)
class RetainedSequence:
    """Complete sequence kept between steps.

    Every candidate of the next step shares the first ``fixed`` tokens.  An
    empty sequence means that no step has run yet.
    """

    token_ids: TokenSequence = ()
    token_logprobs: tuple[float, ...] = ()
    reward: float = 0.0
    fixed: int = 0


@dataclass(frozen=True, slots=True)
class RetainedCandidate:
    """An evaluated candidate and the completion kept if it is selected."""

    candidate: ConditionalCandidate
    completion_index: int


class ConditionalISAdapter:
    """Expose conditional IS steps through the common stepwise protocol."""

    def __init__(
        self,
        *,
        backend: AutoregressiveBackend,
        prompt: TokenSequence,
        config: ConditionalISConfig,
        sampling: SamplingConfig,
        reward: RewardFunction | None,
        reward_batch: RewardBatchFunction | None = None,
    ) -> None:
        self.backend = backend
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
                    model_id=self.backend.model_id,
                    request_id=f"conditional-is:step:{step_index}:retained",
                    finish_reason="eos" if eos is not None and block[-1] == eos else "length",
                )
            )
        fresh = self.config.candidate_count - len(proposals)
        if fresh:
            proposals.extend(
                sample_candidates(
                    self.backend,
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
            retained = RolloutEvaluation(
                token_ids=state.token_ids[start:],
                reward=state.reward,
                log_weight=state.reward / self.config.reward_temperature,
                token_logprobs=state.token_logprobs[start:],
            )
        evaluated = estimate_conditional_weights(
            backend=self.backend,
            prompt=self.prompt,
            generated_prefix=state.token_ids[: state.fixed],
            candidates=proposals,
            rollout_length=self.config.total_length - state.fixed - length,
            rollout_count=self.config.rollout_count,
            sampling=self.sampling,
            reward_temperature=self.config.reward_temperature,
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
        token_ids = state.token_ids[: state.fixed] + candidate.token_ids + completion.token_ids
        token_logprobs = (
            state.token_logprobs[: state.fixed]
            + candidate.base_token_logprobs
            + completion.token_logprobs
        )
        if len(token_logprobs) != len(token_ids):
            raise RuntimeError("kept sequence lost token log-probabilities")
        eos = self.sampling.eos_token_id
        if eos is not None and eos in token_ids:
            # Absorbing backends pad after EOS; the kept sequence ends at EOS.
            end = token_ids.index(eos) + 1
            token_ids, token_logprobs = token_ids[:end], token_logprobs[:end]
        return RetainedSequence(
            token_ids,
            token_logprobs,
            completion.reward,
            state.fixed + len(candidate.token_ids),
        )


def _step_record(
    selection: StepwiseSelection[RetainedSequence, RetainedCandidate],
) -> ConditionalISStep:
    candidates = tuple(item.value.candidate for item in selection.candidates)
    carried = bool(selection.state_before.token_ids)
    return ConditionalISStep(
        generated_length_before=selection.state_before.fixed,
        candidates=candidates,
        selected_index=selection.selected_index,
        completion_index=selection.selected.value.completion_index,
        retained_candidate=carried,
        rollout_evaluations_performed=(
            sum(len(candidate.rollouts) for candidate in candidates) - int(carried)
        ),
    )


def conditional_is_step(
    *,
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    state: RetainedSequence,
    config: ConditionalISConfig,
    sampling: SamplingConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    step_index: int,
    reward_batch: RewardBatchFunction | None = None,
) -> tuple[ConditionalISStep, RetainedSequence]:
    """Run one step from ``state`` and return its record and the kept sequence."""

    adapter = ConditionalISAdapter(
        backend=backend,
        prompt=prompt,
        config=config,
        sampling=sampling,
        reward=reward,
        reward_batch=reward_batch,
    )
    selection = stepwise_generation_step(
        adapter,
        state,
        step_index,
        seeds,
        selection_namespace=("conditional_is",),
    )
    return _step_record(selection), adapter.advance(state, selection.selected.value, step_index)


def run_conditional_is(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: ConditionalISConfig,
    reward: RewardFunction | None,
    seeds: SeedStream,
    *,
    sampling: SamplingConfig | None = None,
    reward_batch: RewardBatchFunction | None = None,
) -> ConditionalISResult:
    """Generate a complete sequence by conditional SIR moves at successive block boundaries."""

    sampling = sampling or SamplingConfig()
    validate_base_sampling(sampling)
    generic = run_stepwise_generation(
        ConditionalISAdapter(
            backend=backend,
            prompt=prompt,
            config=config,
            sampling=sampling,
            reward=reward,
            reward_batch=reward_batch,
        ),
        seeds,
        selection_namespace=("conditional_is",),
    )
    return ConditionalISResult(
        prompt=prompt,
        token_ids=generic.final_state.token_ids,
        steps=tuple(_step_record(step) for step in generic.steps),
    )
