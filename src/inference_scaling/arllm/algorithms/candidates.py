"""Complete-rollout conditional IS with streamed branching and two-stage exploration.

Roots keep running while K-1 siblings branch at the fixed checkpoint. The retained
complete sequence is reused without rescoring. Optional selection-ready stopping
estimates unfinished weights from cached prefixes; this approximation can change
the target distribution. Complete waiting is the reference selection.
"""

from __future__ import annotations

from dataclasses import replace
from math import exp, floor, isfinite, log
from typing import TYPE_CHECKING, NamedTuple

from inference_scaling.arllm.backends.scheduling import GenerationSession
from inference_scaling.arllm.backends.streaming import BatchGenerationSession
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import RolloutEvaluation, weigh
from inference_scaling.shared.sampling.importance import (
    categorical_index_from_uniform, logmeanexp, normalize_log_weights,
)

if TYPE_CHECKING:
    from inference_scaling.arllm.algorithms.conditional_is import ConditionalISAdapter, RetainedSequence


class Block(NamedTuple):
    token_ids: tuple[int, ...]
    finish_reason: str


def _complete_priority_group(logs, complete, keys, planned, lengths, priorities, uniforms, width):
    """A complete cold group outranking empirical pending weights, or None."""
    groups = sorted({group for group, _ in keys})
    positions = [[i for i, key in enumerate(keys) if key[0] == group] for group in groups]
    done = [len(ids) == planned[group] and all(complete[i] for i in ids)
            for group, ids in zip(groups, positions, strict=True)]
    if sum(done) < min(2, len(groups)):
        return None
    if any(not ended and any(lengths[i] < width for i in ids)
           for ended, ids in zip(done, positions, strict=True)):
        return None
    raw = [value + log(planned[group]) for value, (group, _) in zip(logs, keys, strict=True)]
    maximum = max(value for value, ended in zip(raw, complete, strict=True) if ended)
    protected = [value if ended else max(value, maximum) for value, ended in zip(raw, complete, strict=True)]
    # Normalize each group's mean locally, avoiding underflow across groups.
    winner = max(range(len(groups)), key=lambda i: logmeanexp([protected[j] for j in positions[i]])
                 + priorities[groups[i]])
    if not done[winner]:
        return None
    group, ids = groups[winner], positions[winner]
    leaf = categorical_index_from_uniform(normalize_log_weights([raw[i] for i in ids]), uniforms[group])
    return keys[ids[leaf]]


def evaluate_candidates(adapter: ConditionalISAdapter, state: RetainedSequence, step: int, seeds: SeedStream):
    """Candidates, slots, sources, score count, preselection, and next-pool forecast."""
    if not adapter.sampling.full_support:
        raise ValueError("base candidates require full support: top_p=1 and top_k=None")
    config, prompt = adapter.config, adapter.prompt
    final_pool = (config.exploration_schedule == "two_stage" and step == 1 and bool(state.token_ids)
                  and config.rollout_count == 1 and config.block_size >= config.total_length - state.fixed)
    first_pool = (config.exploration_schedule == "two_stage" and step == 0 and not state.token_ids
                  and config.candidate_count > 1)
    cold_priority = first_pool and config.selection_ready
    selection_ready = config.selection_ready and (first_pool or final_pool)
    if config.selection_ready and adapter.prefix_reward is None:
        raise ValueError("weight-based stopping requires a cached prefix reward")
    if selection_ready and state.token_ids and not final_pool:
        raise ValueError("selection_ready requires the initial or final two-stage pool")
    chosen = None
    forecast = None
    # Keep the ordinary final SIR draw, including when prefix statistics are
    # unavailable and all fresh requests must resolve before selection.
    uniform = (float(seeds.generator("conditional_is", step, "select" if final_pool else "selection_ready").random())
               if selection_ready else None)
    candidate_count = config.candidate_count
    rollout_count = config.rollout_count
    priorities = tuple(-log(max(float(seeds.generator("conditional_is", step, "uniform_race", group).random()), 1e-300))
                       for group in range(candidate_count)) if cold_priority else ()
    completion_uniforms = tuple(float(seeds.generator("conditional_is", step, "candidate", group, "completion").random())
                                for group in range(candidate_count)) if cold_priority else ()
    fixed = state.token_ids[:state.fixed]
    remaining = config.total_length - state.fixed
    width = min(config.block_size, remaining)
    minimum_length = None
    length_limit = remaining
    if final_pool and candidate_count > 1:
        # q(t | x) = 1/(t log 2) on len(x) <= t < 2 len(x). With the
        # first block fixed, this density is constant for eligible suffixes.
        # Over-limit samples have zero conditional weight, so their reward is
        # unnecessary. The retained complete suffix always remains eligible.
        bound = len(state.token_ids) * 2 ** float(seeds.generator("conditional_is", step, "length_window").random())
        length_limit = min(remaining, floor(bound - state.fixed))
        minimum_length = bound / 2 - state.fixed
    excluded: set[int] = set()
    blocks: list[Block | None] = [None] * candidate_count
    leaves: list[dict[int, RolloutEvaluation]] = [{} for _ in blocks]
    routes: dict[str, tuple[int, int, bool]] = {}
    leaf_requests: dict[tuple[int, int], str] = {}
    buffers: dict[str, list[int]] = {}
    submitted: dict[str, GenerationRequest] = {}
    finished: set[str] = set()
    cancelled: set[str] = set()
    evaluations = 0
    streaming = bool(getattr(adapter.backend, "supports_streaming_generation", False))
    if selection_ready and not streaming:
        raise ValueError("selection_ready requires a streaming generation backend")
    session: GenerationSession = (getattr(adapter.backend, "open_generation_session")() if streaming
                                  else BatchGenerationSession(adapter.backend))
    pending: dict[tuple[int, int], tuple[int, ...]] = {}

    def request(index, rollout, prefix, length, *, root=False, continuation=False):
        stem = f"conditional-is:step:{step}:candidate:{index}"
        seed = seeds.derive("conditional_is", step, "candidate", index) if root or continuation else seeds.derive(
            "conditional_is", step, "candidate", index, "rollout", rollout)
        key = stem if root else stem + (":continuation" if continuation else f":rollout:{rollout}")
        routes[key] = index, rollout, root
        if selection_ready:
            leaf_requests[index, rollout] = key
        buffers[key] = []
        job = GenerationRequest(prompt + fixed + prefix, length, adapter.sampling, seed, key,
                                uniform_offset=len(prefix) if continuation else 0,
                                intermediate_events=selection_ready)
        submitted[key] = job
        return job

    def branch(index, *, continuation=False):
        block = blocks[index]
        assert block is not None
        length = remaining - len(block.token_ids)
        if block.finish_reason != "length" or length == 0:
            return
        jobs = [request(index, k, block.token_ids, length) for k in range(1, rollout_count)]
        if continuation:
            jobs.insert(0, request(index, 0, block.token_ids, length, continuation=True))
        if jobs:
            session.submit(jobs)

    def evaluate(index, rollout, suffix):
        if not streaming:
            pending[index, rollout] = suffix
            return
        score([(index, rollout, suffix)])

    def score(items):
        nonlocal evaluations
        values = tuple(adapter.reward(prompt, [fixed + blocks[index].token_ids + suffix for index, _, suffix in items]))
        if len(values) != len(items) or any(not isfinite(float(value)) for value in values):
            raise ValueError("reward must return one finite value per completion")
        for (index, rollout, suffix), value in zip(items, values, strict=True):
            value = float(value)
            leaves[index][rollout] = RolloutEvaluation(suffix, value, value / config.reward_temperature)
        evaluations += len(items)

    try:
        carried = int(bool(state.token_ids))
        # Forward token slots of the requests this step measured; an unmeasured one makes the total unknown.
        slots: int = 0
        measured = True
        if carried:
            end = state.fixed + width
            blocks[0] = Block(state.token_ids[state.fixed:end], "length" if end < len(state.token_ids) else "stop")
            leaves[0][0] = RolloutEvaluation(state.token_ids[end:], state.reward, state.reward / config.reward_temperature)
        roots = [request(index, 0, (), min(width, length_limit) if config.block_first else length_limit, root=True)
                 for index in range(carried, candidate_count)]
        if roots:
            checkpoint = width if rollout_count > 1 and not config.block_first else None
            session.submit(roots, phase="prefix", checkpoint=checkpoint)
        for index in range(carried):
            branch(index)
        sealed = rollout_count == 1 and not config.block_first
        if sealed:
            session.seal()

        def expected(block):
            if block is None:
                # A missing prefix can still create K leaves. Use that upper bound
                # until it terminates, rather than waiting for an unusable group.
                return rollout_count if width < remaining else 1
            return rollout_count if block.finish_reason == "length" and len(block.token_ids) < remaining else 1

        def selection_is_ready():
            nonlocal chosen, forecast
            if not selection_ready:
                return False
            resolved = all(name in finished or name in cancelled for name in routes)
            # Final selection already has a complete reference. A fresh
            # completion is not required when the same draw retains it.
            if (not resolved and not (final_pool and selection_ready)
                    and sum(bool(group) for group in leaves) < min(candidate_count, 2)):
                return False
            logs, complete, keys, observed_lengths = [], [], [], []
            for index, (block, group) in enumerate(zip(blocks, leaves, strict=True)):
                if index in excluded:
                    continue
                planned = expected(block)
                for rollout in range(planned):
                    done = rollout in group
                    if done:
                        value = group[rollout].log_weight
                        observed_length = len(block.token_ids) + len(group[rollout].token_ids)
                    else:
                        key = leaf_requests.get((index, rollout))
                        if key is not None:
                            root = routes[key][2]
                            sequence = (() if root else block.token_ids) + tuple(buffers[key])
                        else:
                            # An unreleased sibling has observed only its root block.
                            root_key = leaf_requests[index, 0]
                            sequence = block.token_ids if block is not None else tuple(buffers[root_key][:width])
                        if not sequence:
                            return False
                        proxy = adapter.prefix_reward(prompt, fixed + sequence)
                        if proxy is None:
                            return False
                        if not isfinite(proxy):
                            raise ValueError("prefix reward must be finite or None")
                        value = proxy / config.reward_temperature
                        observed_length = len(sequence)
                    logs.append(value - log(planned))
                    complete.append(done)
                    keys.append((index, rollout))
                    observed_lengths.append(observed_length)
            if cold_priority:
                planned = [expected(block) for block in blocks]
                chosen = _complete_priority_group(logs, complete, keys, planned, observed_lengths,
                                                  priorities, completion_uniforms, width)
                if chosen is None:
                    return False
                from inference_scaling.arllm.algorithms.conditional_is import forecast_candidate_count

                observations = tuple(
                    (key[0], planned[key[0]], state.fixed + length, value + log(planned[key[0]]))
                    for key, length, value in zip(keys, observed_lengths, logs, strict=True)
                    if key[0] != chosen[0])
                forecast = forecast_candidate_count(
                    observations, state.fixed + len(blocks[chosen[0]].token_ids), config, seeds)
                return True
            peak = max(logs)
            masses = [exp(value - peak) for value in logs]
            # Reuse one uniform across events; redrawing would repeatedly
            # favor already-complete leaves. Prefix estimates still bias
            # the stopping rule relative to full-reward SIR.
            threshold = uniform * sum(masses)
            cumulative = 0.
            selected = len(masses) - 1
            for position, mass in enumerate(masses):
                cumulative += mass
                if threshold < cumulative:
                    selected = position
                    break
            if not complete[selected]:
                return False
            chosen = keys[selected]
            return True
        while len(finished) + len(cancelled) < len(routes):
            if not sealed and all(block is not None for block in blocks):
                session.seal()
                sealed = True
            event = session.next_event()
            sample = event.delta
            key = sample.request_id
            if key not in routes or key in finished or key in cancelled:
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
                if minimum_length is not None and (
                        (sample.finish_reason == "length" and length_limit < remaining)
                        or len(tokens) <= minimum_length):
                    excluded.add(index)
                    del buffers[key]
                    continue
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
                if not (root and config.block_first):
                    block = blocks[index]
                    assert block is not None
                    evaluate(index, rollout, tuple(tokens[len(block.token_ids):] if root else tokens))
                del buffers[key]
            if selection_is_ready():
                for pending_key in routes:
                    if pending_key not in finished and pending_key not in cancelled:
                        session.stop(pending_key)
                        cancelled.add(pending_key)
                        buffers.pop(pending_key, None)
                break
        if pending:
            score([(index, rollout, suffix) for (index, rollout), suffix in sorted(pending.items())])
            pending.clear()
        if not selection_ready and any(block is None and index not in excluded for index, block in enumerate(blocks)):
            raise RuntimeError("stream omitted a candidate prefix")
        canonical = []
        canonical_blocks = []
        indices = []
        for index, (block, group) in enumerate(zip(blocks, leaves, strict=True)):
            complete = set(group) == set(range(expected(block)))
            if not selection_ready and not complete and index not in excluded:
                raise RuntimeError("stream omitted a required completion")
            if group:
                assert block is not None
                indices.append(index)
                canonical_blocks.append(block)
                # Keep the planned K denominator when some siblings are unfinished.
                correction = log(len(group) / expected(block)) if selection_ready else 0.
                canonical.append([replace(group[k], log_weight=group[k].log_weight + correction)
                                  if selection_ready else group[k] for k in sorted(group)])
        result = weigh([block.token_ids for block in canonical_blocks], canonical)
        if len(finished) != len(routes):
            measured = False
        sources = tuple(indices) if len(indices) != candidate_count else None
        if selection_ready and chosen is None:
            # The last event may exclude a fresh request by its length. The
            # retained complete sequence remains eligible even if all fresh
            # proposals were excluded, and can be selected without new work.
            selection_is_ready()
        if selection_ready and chosen is None:
            raise RuntimeError("selection-ready pool ended without a completed selection")
        selection = (indices.index(chosen[0]), sorted(leaves[chosen[0]]).index(chosen[1])) if chosen else None
        return result, (slots if measured else None), sources, evaluations, selection, forecast
    finally:
        session.close()


__all__ = ["evaluate_candidates"]
