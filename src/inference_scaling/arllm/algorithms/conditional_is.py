"""Conditional importance sampling on a kept complete sequence.

The target is ``p(y | x) exp(r(x, y) / tau)`` for a reward ``r`` of the complete
sequence.  Each step cuts the kept sequence at the next block boundary.
Candidate 0 is the kept sequence's next block, and the rest of the kept
sequence counts as one of that candidate's completions; the other candidates,
and the other completions of candidate 0, are fresh base-policy samples.  A
fresh candidate is cut from a complete output, whose rest is its first
completion.  Every completion runs until it stops (at EOS or a scope boundary)
or reaches the length limit, and is scored as a complete sequence.
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

from inference_scaling.arllm.algorithms.candidates import (OWN_STREAM, Completion, OwnStream, cut_block, sample_outputs,
                                                           validate_base_sampling)
from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.importance import categorical_index_from_uniform, logmeanexp, normalize_log_weights
from inference_scaling.shared.types import GeneratedBatchReward
from inference_scaling.arllm.types import AutoregressiveBackend, GenerationRequest, SequenceSample, TokenSequence


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
    generated_prefix_logprobs: Sequence[float],
    candidates: Sequence[SequenceSample],
    first_completions: Sequence[Completion | OwnStream | None],
    total_length: int,
    rollout_count: int,
    sampling: SamplingConfig,
    reward_temperature: float,
    reward: GeneratedBatchReward,
    seeds: SeedStream,
    step_index: int,
    retained: RolloutEvaluation | None = None,
) -> tuple[ConditionalCandidate, ...]:
    """Estimate each candidate's weight with ``rollout_count`` base-policy completions.

    A candidate that ends the sequence has one empty completion. The first
    completion of any other candidate is given: ``retained``, an already
    evaluated completion of candidate 0 that is neither regenerated nor
    re-scored, the rest of the output the candidate was cut from, or
    ``OWN_STREAM`` for a block-first candidate, whose first completion continues
    its own request here. The others are generated here, all in one batch, and
    scored in one reward batch.
    """

    if rollout_count <= 0:
        raise ValueError("rollout_count must be positive")
    if reward_temperature <= 0:
        raise ValueError("reward_temperature must be positive")
    if len(first_completions) != len(candidates):
        raise ValueError("each candidate needs its first completion or None")

    requests: list[GenerationRequest] = []
    # The candidate and completion position that each request fills.
    slots: list[tuple[int, int]] = []
    # Completions of each candidate still to be scored.
    unscored: list[list[Completion | None]] = [[] for _ in candidates]

    def request(index: int, candidate: SequenceSample, length: int, seed: int, request_id: str, offset: int) -> None:
        slots.append((index, len(unscored[index])))
        unscored[index].append(None)
        requests.append(GenerationRequest(prefix=prompt + generated_prefix + candidate.token_ids, max_new_tokens=length,
                                          sampling=sampling, seed=seed, request_id=request_id, uniform_offset=offset))

    for index, candidate in enumerate(candidates):
        kept = retained is not None and index == 0
        rollout_length = total_length - len(generated_prefix) - len(candidate.token_ids)
        if rollout_length == 0 or candidate.finish_reason != "length":
            if not kept:
                unscored[index].append(((), ()))
            continue
        first = first_completions[index]
        if not kept:
            if first is None:
                raise ValueError("a continuing candidate needs its first completion")
            if isinstance(first, OwnStream):
                request(index, candidate, rollout_length, seeds.derive("conditional_is", step_index, "candidate", index),
                        f"conditional-is:step:{step_index}:candidate:{index}:continuation", len(candidate.token_ids))
            else:
                unscored[index].append(first)
        for rollout_index in range(1, rollout_count):
            request(index, candidate, rollout_length,
                    seeds.derive("conditional_is", step_index, "candidate", index, "rollout", rollout_index),
                    f"conditional-is:step:{step_index}:candidate:{index}:rollout:{rollout_index}", 0)
    samples = backend.sample_batch(requests) if requests else []
    if len(samples) != len(requests):
        raise RuntimeError("backend returned an invalid number of rollouts")
    for (index, position), sample in zip(slots, samples, strict=True):
        unscored[index][position] = (sample.token_ids, sample.token_logprobs)

    prefix_logprobs = tuple(generated_prefix_logprobs)
    pending = [(index, completion) for index, group in enumerate(unscored) for completion in group]
    rewards = tuple(float(value) for value in reward(
        prompt, [generated_prefix + candidates[index].token_ids + tokens for index, (tokens, _) in pending],
        [prefix_logprobs + candidates[index].token_logprobs + logprobs for index, (_, logprobs) in pending],
    )) if pending else ()
    if len(rewards) != len(pending):
        raise ValueError("reward returned an invalid number of values")
    if any(not isfinite(value) for value in rewards):
        raise ValueError("reward must be finite")
    by_candidate: list[list[RolloutEvaluation]] = [[] for _ in candidates]
    if retained is not None:
        by_candidate[0].append(retained)
    for (index, (tokens, logprobs)), value in zip(pending, rewards, strict=True):
        by_candidate[index].append(RolloutEvaluation(tokens, value, value / reward_temperature, logprobs))

    return tuple(
        ConditionalCandidate(candidate.token_ids, candidate.token_logprobs, tuple(group),
                             logmeanexp([item.log_weight for item in group]))
        for candidate, group in zip(candidates, by_candidate, strict=True)
    )


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


@dataclass(frozen=True, kw_only=True)
class ConditionalISAdapter:
    """One conditional IS move: propose candidates, weight them, keep one completion."""

    backend: AutoregressiveBackend
    prompt: TokenSequence
    config: ConditionalISConfig
    sampling: SamplingConfig
    reward: GeneratedBatchReward

    @property
    def initial_state(self) -> RetainedSequence:
        return RetainedSequence()

    def is_terminal(self, state: RetainedSequence) -> bool:
        return bool(state.token_ids) and state.fixed >= len(state.token_ids)

    def _block_length(self, state: RetainedSequence) -> int:
        return min(self.config.block_size, self.config.total_length - state.fixed)

    def propose(self, state: RetainedSequence, step_index: int,
                seeds: SeedStream) -> list[tuple[SequenceSample, Completion | OwnStream | None]]:
        """Candidate 0 continues the kept sequence; fresh candidates are cut from complete outputs or drawn block-first."""

        validate_base_sampling(self.sampling)
        prefix = self.prompt + state.token_ids[: state.fixed]
        length = self._block_length(state)
        proposals: list[tuple[SequenceSample, Completion | OwnStream | None]] = []
        if state.token_ids:
            end = state.fixed + length
            proposals.append((SequenceSample(
                prefix=prefix,
                token_ids=state.token_ids[state.fixed : end],
                token_logprobs=state.token_logprobs[state.fixed : end],
                policy_id=self.sampling.policy_id,
                model_id=self.backend.model_id,
                request_id=f"conditional-is:step:{step_index}:retained",
                # The kept sequence either continues after the block or ends with it.
                finish_reason="length" if end < len(state.token_ids) else "stop",
            ), None))
        fresh = self.config.candidate_count - len(proposals)
        if fresh:
            block_first = self.config.block_first
            outputs = sample_outputs(self.backend, prefix, fresh, length if block_first else self.config.total_length - state.fixed,
                                     self.sampling, seeds, step_index, first_index=len(proposals))
            proposals.extend((output, OWN_STREAM) if block_first else cut_block(output, length) for output in outputs)
        return proposals

    def step(
        self, state: RetainedSequence, step_index: int, seeds: SeedStream,
    ) -> tuple[ConditionalISStep, RetainedSequence]:
        """Select a candidate in proportion to its weight, keep one of its completions and advance."""

        if self.is_terminal(state):
            raise ValueError("cannot advance a terminal generation state")
        proposals = self.propose(state, step_index, seeds)
        retained = None
        if state.token_ids:
            start = state.fixed + len(proposals[0][0].token_ids)
            retained = RolloutEvaluation(state.token_ids[start:], state.reward,
                                         state.reward / self.config.reward_temperature, state.token_logprobs[start:])
        candidates = estimate_conditional_weights(
            backend=self.backend, prompt=self.prompt, generated_prefix=state.token_ids[: state.fixed],
            generated_prefix_logprobs=state.token_logprobs[: state.fixed],
            candidates=[candidate for candidate, _ in proposals], first_completions=[first for _, first in proposals],
            total_length=self.config.total_length, rollout_count=self.config.rollout_count, sampling=self.sampling,
            reward_temperature=self.config.reward_temperature, reward=self.reward, seeds=seeds,
            step_index=step_index, retained=retained,
        )
        # Drawn for every candidate so the kept completion is independent of which candidate the step selects.
        completions = [categorical_index_from_uniform(
            normalize_log_weights([rollout.log_weight for rollout in candidate.rollouts]),
            float(seeds.generator("conditional_is", step_index, "candidate", index, "completion").random()),
        ) for index, candidate in enumerate(candidates)]
        selected = categorical_index_from_uniform(normalize_log_weights([candidate.log_weight for candidate in candidates]),
                                                  float(seeds.generator("conditional_is", step_index, "select").random()))
        candidate = candidates[selected]
        completion = candidate.rollouts[completions[selected]]
        carried = bool(state.token_ids)
        record = ConditionalISStep(
            generated_length_before=state.fixed, candidates=candidates, selected_index=selected,
            completion_index=completions[selected], retained_candidate=carried,
            # The carried completion was evaluated in an earlier step.
            rollout_evaluations_performed=sum(len(item.rollouts) for item in candidates) - int(carried),
        )
        return record, RetainedSequence(
            state.token_ids[: state.fixed] + candidate.token_ids + completion.token_ids,
            state.token_logprobs[: state.fixed] + candidate.base_token_logprobs + completion.token_logprobs,
            completion.reward, state.fixed + len(candidate.token_ids))


def conditional_is_step(
    *,
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    state: RetainedSequence,
    config: ConditionalISConfig,
    sampling: SamplingConfig,
    reward: GeneratedBatchReward,
    seeds: SeedStream,
    step_index: int,
) -> tuple[ConditionalISStep, RetainedSequence]:
    """Run one step from ``state`` and return its record and the kept sequence."""

    adapter = ConditionalISAdapter(backend=backend, prompt=prompt, config=config, sampling=sampling, reward=reward)
    return adapter.step(state, step_index, seeds)


def run_conditional_is(
    backend: AutoregressiveBackend,
    prompt: TokenSequence,
    config: ConditionalISConfig,
    reward: GeneratedBatchReward,
    seeds: SeedStream,
    *,
    sampling: SamplingConfig | None = None,
) -> ConditionalISResult:
    """Generate a complete sequence by conditional SIR moves at successive block boundaries."""

    sampling = sampling or SamplingConfig()
    validate_base_sampling(sampling)
    adapter = ConditionalISAdapter(backend=backend, prompt=prompt, config=config, sampling=sampling, reward=reward)
    state = adapter.initial_state
    steps: list[ConditionalISStep] = []
    while not adapter.is_terminal(state):
        record, state = adapter.step(state, len(steps), seeds)
        steps.append(record)
    return ConditionalISResult(prompt=prompt, token_ids=state.token_ids, steps=tuple(steps))
