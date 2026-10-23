"""Conditional IS's tree, released as soon as its dependencies are known.

The original full request remains completion zero. Neither completion order nor
runtime enters ordinary IS weights. The two-stage final pool can exclude samples
outside an auxiliary length interval. Tail filtering uses a common
observed token depth and inverse-continuation weights. Completion quotas and per-block
races instead select by completion time and change the sampling law. A weighted
quota estimates unfinished weights from cached prefixes, without extra scoring.
Selection-ready stopping uses those estimates with one fixed categorical draw;
single-pool K>1 siblings branch only after the first root finishes. In two-stage
exploration the first tree stays complete and only the final pool may stop early.
"""

from __future__ import annotations

from dataclasses import replace
from math import ceil, exp, floor, isfinite, log
from typing import TYPE_CHECKING, NamedTuple

from inference_scaling.arllm.backends.scheduling import GenerationSession
from inference_scaling.arllm.backends.streaming import BatchGenerationSession
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import RolloutEvaluation, weigh

if TYPE_CHECKING:
    from inference_scaling.arllm.algorithms.conditional_is import ConditionalISAdapter, RetainedSequence


class Block(NamedTuple):
    token_ids: tuple[int, ...]
    finish_reason: str


class TailDecision(NamedTuple):
    candidate: int
    rollout: int
    probability: float
    keep: bool


def evaluate_candidates(adapter: ConditionalISAdapter, state: RetainedSequence, step: int, seeds: SeedStream):
    """Candidates, measured slots, source groups, score count, and optional preselected indices."""
    if not adapter.sampling.full_support:
        raise ValueError("base candidates require full support: top_p=1 and top_k=None")
    config, prompt = adapter.config, adapter.prompt
    if config.last_rollout_filter and adapter.prefix_reward is None:
        raise ValueError("last_rollout_filter requires a cached prefix reward")
    final_pool = (config.exploration_schedule == "two_stage" and step == 1 and bool(state.token_ids)
                  and config.rollout_count == 1 and config.block_size >= config.total_length - state.fixed)
    selection_ready = config.selection_ready and (config.exploration_schedule == "single_pool" or final_pool)
    weighted = config.weighted_completion_quota or selection_ready
    if (config.weighted_completion_quota or config.selection_ready) and adapter.prefix_reward is None:
        raise ValueError("weight-based stopping requires a cached prefix reward")
    if selection_ready and state.token_ids and not final_pool:
        raise ValueError("selection_ready requires a fresh single pool or the two-stage final pool")
    chosen = None
    # Keep the ordinary final SIR draw, including when prefix statistics are
    # unavailable and all fresh requests must resolve before selection.
    uniform = (float(seeds.generator("conditional_is", step, "select" if final_pool else "selection_ready").random())
               if selection_ready else None)
    candidate_count = config.candidate_count
    rollout_count = config.rollout_count
    fixed = state.token_ids[:state.fixed]
    remaining = config.total_length - state.fixed
    width = min(config.block_size, remaining)
    branch_open = not selection_ready or rollout_count == 1
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
    tail: TailDecision | None = None
    evaluations = 0
    backup_groups: list[int] = []
    streaming = bool(getattr(adapter.backend, "supports_streaming_generation", False))
    partial = config.completion_fraction < 1 or weighted
    if (partial or config.rollout_race or config.selection_ready) and not streaming:
        raise ValueError("completion quotas and rollout races require a streaming generation backend")
    session: GenerationSession = (getattr(adapter.backend, "open_generation_session")() if streaming
                                  else BatchGenerationSession(adapter.backend))
    pending: dict[tuple[int, int], tuple[int, ...]] = {}

    def request(index, rollout, prefix, length, *, root=False, continuation=False):
        stem = f"conditional-is:step:{step}:candidate:{index}"
        seed = seeds.derive("conditional_is", step, "candidate", index) if root or continuation else seeds.derive(
            "conditional_is", step, "candidate", index, "rollout", rollout)
        key = stem if root else stem + (":continuation" if continuation else f":rollout:{rollout}")
        routes[key] = index, rollout, root
        if weighted:
            leaf_requests[index, rollout] = key
        buffers[key] = []
        job = GenerationRequest(prompt + fixed + prefix, length, adapter.sampling, seed, key,
                                uniform_offset=len(prefix) if continuation else 0,
                                intermediate_events=config.last_rollout_filter or weighted)
        submitted[key] = job
        return job

    def branch(index, *, continuation=False):
        block = blocks[index]
        assert block is not None
        length = remaining - len(block.token_ids)
        if block.finish_reason != "length" or length == 0:
            return
        if config.rollout_race:
            backup_groups.append(index)
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

    def decide_tail(index, rollout, sequence, frontier):
        """The same logical decision applies whether the last stream is still cancellable or has finished."""
        assert adapter.prefix_reward is not None
        value = adapter.prefix_reward(prompt, fixed + sequence[:frontier])
        if value is None:
            return TailDecision(index, rollout, 1., True)
        if not isfinite(value):
            raise ValueError("prefix reward must be finite or None")
        proxy = value / config.reward_temperature
        other = [item.log_weight for m, group in enumerate(leaves) for k, item in group.items()
                 if (m, k) != (index, rollout)]
        count = sum(expected(block) for block in blocks)
        probability = (1 + (count - 1) * exp(proxy - max(proxy, *other))) / count
        # Force the reference through the auxiliary subset update, while still
        # dividing its final weight by this probability. This preserves the
        # joint target; simply exempting its weight would not.
        keep = (index, rollout) == (0, 0) or float(
            seeds.generator("conditional_is", step, "last_tail_keep").random()) < probability
        return TailDecision(index, rollout, probability, keep)

    try:
        carried = int(bool(state.token_ids))
        # Forward token slots of the requests this step measured; an unmeasured one makes the total unknown.
        slots: int = 0
        measured = True
        if carried:
            end = state.fixed + width
            blocks[0] = Block(state.token_ids[state.fixed:end], "length" if end < len(state.token_ids) else "stop")
            leaves[0][0] = RolloutEvaluation(state.token_ids[end:], state.reward, state.reward / config.reward_temperature)
        recycled = state.siblings[:candidate_count - carried] if config.sibling_replay else ()
        for index, sibling in enumerate(recycled, start=carried):
            blocks[index] = Block(sibling.token_ids[:width], "length" if width < len(sibling.token_ids) else "stop")
            leaves[index][0] = RolloutEvaluation(sibling.token_ids[width:], sibling.reward,
                                                sibling.reward / config.reward_temperature)
        saved = carried + len(recycled)
        roots = [request(index, 0, (), min(width, length_limit) if config.block_first else length_limit, root=True)
                 for index in range(saved, candidate_count)]
        if roots:
            checkpoint = width if (rollout_count > 1 or config.rollout_race) and not config.block_first else None
            session.submit(roots, phase="prefix", checkpoint=checkpoint)
        for index in range(saved):
            branch(index)
        sealed = rollout_count == 1 and not config.block_first and not config.rollout_race
        if sealed:
            session.seal()

        def expected(block):
            if block is None:
                # A missing prefix can still create K leaves. Use that upper bound
                # until it terminates, rather than waiting for an unusable group.
                return rollout_count if width < remaining else 1
            return rollout_count if block.finish_reason == "length" and len(block.token_ids) < remaining else 1

        def quota_ready():
            nonlocal chosen
            if not partial:
                return False
            if weighted:
                resolved = all(name in finished or name in cancelled for name in routes)
                # Final selection already has a complete reference. A fresh
                # completion is not required when the same draw retains it.
                if (not resolved and not (final_pool and selection_ready)
                        and sum(bool(group) for group in leaves) < min(candidate_count, 2)):
                    return False
                logs, complete, keys = [], [], []
                for index, (block, group) in enumerate(zip(blocks, leaves, strict=True)):
                    if index in excluded:
                        continue
                    planned = expected(block)
                    for rollout in range(planned):
                        done = rollout in group
                        if done:
                            value = group[rollout].log_weight
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
                        logs.append(value - log(planned))
                        complete.append(done)
                        keys.append((index, rollout))
                peak = max(logs)
                masses = [exp(value - peak) for value in logs]
                if selection_ready:
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
                # Prefix scores are estimates, not bounds on the final reward.
                return (sum(value for value, done in zip(masses, complete, strict=True) if done)
                        / sum(masses)) >= 1 - 1 / candidate_count
            # A terminal block already has a known weight: its K empty
            # continuations coincide. Count each group's completed fraction
            # in units of 1/K, without generating duplicate empty continuations.
            completed = sum(len(group) * rollout_count // expected(block)
                            for block, group in zip(blocks, leaves, strict=True))
            required = ceil(config.completion_fraction * candidate_count * rollout_count)
            fresh = saved + int(bool(roots) or rollout_count > 1)
            return sum(map(len, leaves)) >= fresh and completed >= required

        def cancel_completed_group(index):
            if not config.rollout_race or len(leaves[index]) < expected(blocks[index]):
                return
            for key, (candidate, _, _) in routes.items():
                if candidate == index and key not in finished and key not in cancelled:
                    session.stop(key)
                    cancelled.add(key)
                    buffers.pop(key, None)

        while len(finished) + len(cancelled) < len(routes):
            if not sealed and all(block is not None for block in blocks):
                # Submit all original branches before backups can use free slots.
                # Already completed/reused leaves reduce the missing quota.
                for index in backup_groups:
                    block = blocks[index]
                    assert block is not None
                    missing = max(0, expected(block) - len(leaves[index]))
                    if missing:
                        session.submit([request(index, k, block.token_ids, remaining - len(block.token_ids))
                                        for k in range(rollout_count, rollout_count + missing)], phase="spare")
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
                elif (branch_open and len(tokens) >= width) or event.finished:
                    if not branch_open:
                        # All roots stay live. The first completed root sets a
                        # common minimum fork depth; existing seeds are reused
                        # for siblings of roots that continue beyond that depth.
                        width = max(width, len(tokens))
                        branch_open = True
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
                cancel_completed_group(index)
            if config.last_rollout_filter and not sealed and all(block is not None for block in blocks):
                session.seal()
                sealed = True
            if (config.last_rollout_filter and tail is None and sealed
                    and all(block is not None for block in blocks)
                    and sum(map(len, leaves)) == sum(expected(block) for block in blocks) - 1
                    and sum(map(len, leaves)) > 0):
                outstanding = [name for name in routes if name not in finished and name not in cancelled]
                if len(outstanding) != 1:
                    raise RuntimeError("invalid last-rollout accounting")
                last = outstanding[0]
                m, k, root = routes[last]
                block = blocks[m]
                assert block is not None
                sequence = (() if root else block.token_ids) + tuple(buffers[last])
                frontier = max(width, max(len(blocks[i].token_ids) + len(item.token_ids)
                                         for i, group in enumerate(leaves) for item in group.values()))
                # Strictly past the frontier excludes equal-length completions,
                # even when their empty final event has not yet been delivered.
                if len(sequence) > frontier:
                    tail = decide_tail(m, k, sequence, frontier)
                    if not tail.keep:
                        session.stop(last)
                        cancelled.add(last)
                        buffers.pop(last)
            if quota_ready():
                if weighted:
                    for pending_key in routes:
                        if pending_key not in finished and pending_key not in cancelled:
                            session.stop(pending_key)
                            cancelled.add(pending_key)
                            buffers.pop(pending_key, None)
                break
        if pending:
            score([(index, rollout, suffix) for (index, rollout), suffix in sorted(pending.items())])
            pending.clear()
        if config.last_rollout_filter and tail is None:
            ordered = sorted(((len(blocks[m].token_ids) + len(item.token_ids), m, k)
                              for m, group in enumerate(leaves) for k, item in group.items()))
            if len(ordered) > 1:
                length, m, k = ordered[-1]
                frontier = max(width, ordered[-2][0])
                if length > frontier:
                    tail = decide_tail(m, k, blocks[m].token_ids + leaves[m][k].token_ids, frontier)
        if tail is not None and not tail.keep:
            leaves[tail.candidate].pop(tail.rollout, None)
        if not partial and any(block is None and index not in excluded for index, block in enumerate(blocks)):
            raise RuntimeError("stream omitted a candidate prefix")
        canonical = []
        canonical_blocks = []
        indices = []
        for index, (block, group) in enumerate(zip(blocks, leaves, strict=True)):
            required = set(range(expected(block)))
            if tail is not None and not tail.keep and tail.candidate == index:
                required.remove(tail.rollout)
            complete = (len(group) == expected(block) if config.rollout_race
                        else set(group) == required)
            if not partial and not complete and index not in excluded:
                raise RuntimeError("stream omitted a required completion")
            if group:
                assert block is not None
                indices.append(index)
                canonical_blocks.append(block)
                if tail is None:
                    correction = log(len(group) / expected(block)) if weighted else 0.
                    canonical.append([replace(group[k], log_weight=group[k].log_weight + correction)
                                      if weighted else group[k] for k in sorted(group)])
                else:
                    # weigh()/selection divide by the kept group size. Undo
                    # that denominator and retain the originally planned K_m.
                    correction = log(len(group) / expected(block))
                    canonical.append([replace(group[k], log_weight=group[k].log_weight + correction
                                      - (log(tail.probability) if (index, k) == tail[:2] else 0.))
                                      for k in sorted(group)])
        result = weigh([block.token_ids for block in canonical_blocks], canonical)
        if partial and not weighted:
            # Missing leaves contribute zero; keep the planned denominator.
            result = tuple(replace(item, log_weight=item.log_weight + log(len(item.rollouts) / expected(block)))
                           for item, block in zip(result, canonical_blocks, strict=True))
        if len(finished) != len(routes):
            measured = False
        sources = tuple(indices) if len(indices) != candidate_count else None
        if selection_ready and chosen is None:
            # The last event may exclude a fresh request by its length. The
            # retained complete sequence remains eligible even if all fresh
            # proposals were excluded, and can be selected without new work.
            quota_ready()
        if selection_ready and chosen is None:
            raise RuntimeError("selection-ready pool ended without a completed selection")
        selection = (indices.index(chosen[0]), sorted(leaves[chosen[0]]).index(chosen[1])) if chosen else None
        return result, (slots if measured else None), sources, evaluations, selection
    finally:
        session.close()


__all__ = ["evaluate_candidates"]
