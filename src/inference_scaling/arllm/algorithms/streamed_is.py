"""Conditional IS's fixed tree, released as soon as its dependencies are known.

The original full request remains completion zero. Neither completion order nor
runtime enters the weights; every result returns to its predetermined slot.
"""

from __future__ import annotations

from math import isfinite
from typing import TYPE_CHECKING

from inference_scaling.arllm.algorithms.candidates import Block, validate_base_sampling
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import RolloutEvaluation, weigh

if TYPE_CHECKING:
    from inference_scaling.arllm.algorithms.conditional_is import ConditionalISAdapter, RetainedSequence


def streamed_candidates(adapter: ConditionalISAdapter, state: RetainedSequence, step: int, seeds: SeedStream):
    """This step's candidates, and the forward token slots they cost when every one of them was measured."""
    validate_base_sampling(adapter.sampling)
    config, prompt = adapter.config, adapter.prompt
    fixed = state.token_ids[:state.fixed]
    remaining = config.total_length - state.fixed
    width = min(config.block_size, remaining)
    blocks: list[Block | None] = [None] * config.candidate_count
    leaves: list[dict[int, RolloutEvaluation]] = [{} for _ in blocks]
    routes: dict[str, tuple[int, int, bool]] = {}
    buffers: dict[str, list[int]] = {}
    submitted: dict[str, GenerationRequest] = {}
    finished: set[str] = set()
    session = getattr(adapter.backend, "open_generation_session")()

    def request(index, rollout, prefix, length, *, root=False, continuation=False):
        stem = f"conditional-is:step:{step}:candidate:{index}"
        seed = seeds.derive("conditional_is", step, "candidate", index) if root or continuation else seeds.derive(
            "conditional_is", step, "candidate", index, "rollout", rollout)
        key = stem if root else stem + (":continuation" if continuation else f":rollout:{rollout}")
        routes[key] = index, rollout, root
        buffers[key] = []
        job = GenerationRequest(prompt + fixed + prefix, length, adapter.sampling, seed, key,
                                uniform_offset=len(prefix) if continuation else 0)
        submitted[key] = job
        return job

    def branch(index, *, continuation=False):
        block = blocks[index]
        assert block is not None
        length = remaining - len(block.token_ids)
        if block.finish_reason != "length" or length == 0:
            return
        jobs = [request(index, k, block.token_ids, length) for k in range(1, config.rollout_count)]
        if continuation:
            jobs.insert(0, request(index, 0, block.token_ids, length, continuation=True))
        if jobs:
            session.submit(jobs)

    def evaluate(index, rollout, suffix):
        block = blocks[index]
        assert block is not None
        values = tuple(adapter.reward(prompt, [fixed + block.token_ids + suffix]))
        if len(values) != 1 or not isfinite(float(values[0])):
            raise ValueError("reward must return one finite value per completion")
        value = float(values[0])
        leaves[index][rollout] = RolloutEvaluation(suffix, value, value / config.reward_temperature)

    try:
        carried = int(bool(state.token_ids))
        # Forward token slots of the requests this step measured; an unmeasured one makes the total unknown.
        slots: int = 0
        measured = True
        if carried:
            end = state.fixed + width
            blocks[0] = Block(state.token_ids[state.fixed:end], "length" if end < len(state.token_ids) else "stop")
            leaves[0][0] = RolloutEvaluation(state.token_ids[end:], state.reward, state.reward / config.reward_temperature)
        roots = [request(index, 0, (), width if config.block_first else remaining, root=True)
                 for index in range(carried, config.candidate_count)]
        if roots:
            session.submit(roots, phase="prefix", checkpoint=width)
        if carried:
            branch(0)
        sealed = False
        while len(finished) < len(routes):
            if not sealed and all(block is not None for block in blocks):
                session.seal()
                sealed = True
            event = session.next_event()
            sample = event.delta
            key = sample.request_id
            if key not in routes or key in finished:
                raise RuntimeError("stream returned an unknown or completed request")
            job = submitted[key]
            if (sample.prefix != job.prefix or sample.model_id != adapter.backend.model_id
                    or sample.policy_id != job.sampling.policy_id):
                raise RuntimeError("stream changed the requested base policy or prefix")
            index, rollout, root = routes[key]
            tokens = buffers[key]
            tokens.extend(sample.token_ids)
            if len(tokens) > job.max_new_tokens:
                raise RuntimeError("stream exceeded its generation limit")
            if root and blocks[index] is None:
                if config.block_first:
                    if event.finished:
                        blocks[index] = Block(tuple(tokens), sample.finish_reason)
                        continuing = sample.finish_reason == "length" and len(tokens) < remaining
                        branch(index, continuation=continuing)
                        if not continuing:
                            evaluate(index, 0, ())
                elif len(tokens) >= width or event.finished:
                    # A stopped sequence of exactly B tokens has one empty leaf.
                    terminal = event.finished and len(tokens) <= width
                    blocks[index] = Block(tuple(tokens[:width]), "stop" if terminal else "length")
                    branch(index)
            if event.finished:
                if not tokens:
                    raise RuntimeError("generation produced an empty sequence")
                if sample.finish_reason == "length" and len(tokens) != job.max_new_tokens:
                    raise RuntimeError("generation ended before its length limit without a stop")
                finished.add(key)
                if sample.forward_token_slots is None:
                    measured = False
                else:
                    slots += sample.forward_token_slots
                if not (root and config.block_first):
                    block = blocks[index]
                    assert block is not None
                    evaluate(index, rollout, tuple(tokens[len(block.token_ids):] if root else tokens))
        if any(block is None for block in blocks):
            raise RuntimeError("stream omitted a candidate prefix")
        canonical = []
        for block, group in zip(blocks, leaves, strict=True):
            assert block is not None
            count = config.rollout_count if block.finish_reason == "length" and len(block.token_ids) < remaining else 1
            if set(group) != set(range(count)):
                raise RuntimeError("stream omitted a required completion")
            canonical.append([group[index] for index in range(count)])
        return weigh([block.token_ids for block in blocks if block is not None], canonical), (slots if measured else None)
    finally:
        session.close()


__all__ = ["streamed_candidates"]
