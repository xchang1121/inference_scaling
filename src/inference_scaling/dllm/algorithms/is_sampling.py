"""Blockwise reward-weighted importance sampling for masked dLLMs.

Each step proposes ``candidate_count`` next blocks from the base policy, completes
every candidate ``rollout_count`` times with the same policy and selects one
candidate in proportion to the mean ``exp(reward / temperature)`` of its
completions. Generation stops after a block of EOS; a candidate that ends there
completes the sequence and has one empty completion.

The model reads the whole canvas, masked positions included, so a block requested
alone (``candidate_canvas = "block"``) need not follow the block distribution of a
generation whose canvas reaches the end, which completions and plain sampling
use. With ``"full"`` a fresh candidate is cut from a complete output of the
remaining canvas and the rest of that output is its first completion, so
candidates and completions come from one policy. Without a kept sequence the
completions are discarded after the selection: a blockwise SIR approximation of
``p(y) exp(r(y) / temperature)``. With ``kept_sequence`` the step also keeps one
completion of the selected candidate in proportion to its weight, and the next
step's candidate 0 is that sequence's next block with the rest as its first,
already scored completion: the conditional SIR move of the AR sampler, which
leaves the target invariant.
"""

from __future__ import annotations

from inference_scaling.dllm.algorithms.config import DiffusionISConfig
from inference_scaling.dllm.config import DiffusionSamplingConfig, diffusion_decision_stage_lengths
from inference_scaling.dllm.types import DiffusionBackend, DiffusionGenerationRequest
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import (ConditionalISResult, ConditionalISStep, RolloutEvaluation,
                                                              select, weigh)
from inference_scaling.shared.types import TokenBatchReward, TokenSequence


def run_conditional_diffusion_is(
    *,
    backend: DiffusionBackend,
    prompt: TokenSequence,
    config: DiffusionISConfig,
    sampling: DiffusionSamplingConfig,
    seed: int,
    reward: TokenBatchReward,
    start: RolloutEvaluation | None = None,
) -> ConditionalISResult:
    """Blockwise IS; ``start`` is a complete, scored continuation to keep before the first step."""

    if start is not None and not config.kept_sequence:
        raise ValueError("a starting sequence needs kept_sequence")
    seeds = SeedStream(seed)
    stages = diffusion_decision_stage_lengths(total_length=config.total_length, decision_block_size=config.block_size,
                                              sampling=sampling)
    full = config.candidate_canvas == "full"
    state: TokenSequence = ()
    kept = start
    steps: list[ConditionalISStep] = []
    for step_index, length in enumerate(stages):
        remaining = config.total_length - len(state) - length
        # (block, whether it ends the sequence, its given first completion: scored, to score, or none).
        proposals: list[tuple[TokenSequence, bool, RolloutEvaluation | TokenSequence | None]] = []
        if kept is not None:
            ends = len(kept.token_ids) <= length
            proposals.append((kept.token_ids[:length], ends, None if ends else RolloutEvaluation(
                kept.token_ids[length:], kept.reward, kept.log_weight)))
        outputs = backend.sample_batch([DiffusionGenerationRequest(
            prefix=prompt + state, generation_length=length + remaining if full else length, sampling=sampling,
            seed=seeds.derive("dllm-is", step_index, "candidate", index),
            request_id=f"dllm-is:step:{step_index}:candidate:{index}", stop_at_eos=True,
        ) for index in range(len(proposals), config.candidate_count)])
        if len(outputs) != config.candidate_count - len(proposals):
            raise RuntimeError("backend returned an invalid number of dLLM candidates")
        for output in outputs:
            if full:
                ends = len(output.token_ids) <= length
                proposals.append((output.token_ids[:length], ends, None if ends else output.token_ids[length:]))
            else:
                proposals.append((output.token_ids, not remaining or output.finish_reason == "eos", None))
        # (owner, completion tokens or the index of its rollout request); a terminal candidate has one empty completion.
        slots: list[tuple[int, TokenSequence | int]] = []
        requests: list[DiffusionGenerationRequest] = []
        for owner, (block, ends, first) in enumerate(proposals):
            if ends:
                if kept is None or owner:
                    slots.append((owner, ()))
                continue
            if first is not None and not isinstance(first, RolloutEvaluation):
                slots.append((owner, tuple(first)))
            for rollout in range(first is not None, config.rollout_count):
                slots.append((owner, len(requests)))
                requests.append(DiffusionGenerationRequest(
                    prefix=prompt + state + block, generation_length=remaining, sampling=sampling,
                    seed=seeds.derive("dllm-is", step_index, "rollout", owner, rollout),
                    request_id=f"dllm-is:step:{step_index}:candidate:{owner}:rollout:{rollout}", stop_at_eos=True,
                ))
        samples = backend.sample_batch(requests) if requests else []
        if len(samples) != len(requests):
            raise RuntimeError("backend returned an invalid number of dLLM rollouts")
        completions = [(owner, samples[item].token_ids if isinstance(item, int) else item) for owner, item in slots]
        rewards = [float(value) for value in reward(prompt, [
            state + proposals[owner][0] + tokens for owner, tokens in completions])] if completions else []
        if len(rewards) != len(completions):
            raise RuntimeError("reward evaluator returned an invalid number of values")
        grouped: list[list[RolloutEvaluation]] = [[] for _ in proposals]
        if kept is not None:
            # The kept sequence's rest, or its empty completion when it ends here, was scored before.
            rest = proposals[0][2]
            grouped[0].append(rest if isinstance(rest, RolloutEvaluation)
                              else RolloutEvaluation((), kept.reward, kept.log_weight))
        for (owner, tokens), value in zip(completions, rewards, strict=True):
            grouped[owner].append(RolloutEvaluation(tokens, value, value / config.reward_temperature))
        evaluated = weigh([block for block, _, _ in proposals], grouped)
        selected, drawn = select(evaluated, seeds, "dllm-is", step_index)
        completion = drawn if config.kept_sequence else None
        steps.append(ConditionalISStep(len(state), evaluated, selected, completion, kept is not None, len(completions)))
        state += evaluated[selected].token_ids
        if proposals[selected][1]:
            break
        if completion is not None:
            kept = evaluated[selected].rollouts[completion]
    return ConditionalISResult(prompt, state, tuple(steps))


__all__ = ["run_conditional_diffusion_is"]
