"""Complete-rollout conditional IS with streamed branching and adaptive/two-stage exploration.

Roots keep running while K-1 siblings branch at the fixed checkpoint. The retained
complete sequence is reused without rescoring. Under ``exploration_schedule="adaptive"``:
1. A B-anchored pilot gate runs candidate 0 first up to ``pilot_tokens`` (default ``B // 2``);
   cleanly finished short thoughts exit immediately at 1x cost, while harder problems
   reuse candidate 0's prefix seamlessly and launch candidates ``1..M-1`` in parallel.
2. When ``K > 1``, unbiased selective K-lookahead allocates ``K`` rollouts to the top
   ``ceil(M/2)`` prefixes (scored purely from prefix tokens up to ``B``) and ``1`` to the rest.
3. Stagnation-only straggler pruning aborts runaway loops on both Stage 1 and Stage 2
   when they substantially exceed median completed length and 2B while trailing median reward.
4. ESS-bounded reward-temperature calibration keeps pool weights well-conditioned across tasks.
"""

from __future__ import annotations

from dataclasses import replace
from math import ceil, exp, floor, isfinite, log
from statistics import median
from typing import TYPE_CHECKING, NamedTuple

from inference_scaling.arllm.backends.scheduling import GenerationSession
from inference_scaling.arllm.backends.streaming import BatchGenerationSession
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import (
    RolloutEvaluation, calibrate_reward_temperature, weigh,
)
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
    config, prompt = adapter.config, adapter.prompt
    # Fixed schedule is the exact full-support reference; two_stage/adaptive perform on-policy
    # conditional IS targeting q(y) * exp(r(y)/tau) where q is the configured sampling policy.
    if config.exploration_schedule == "fixed" and not adapter.sampling.full_support:
        raise ValueError("base candidates require full support: top_p=1 and top_k=None")
    adaptive = config.exploration_schedule in {"adaptive", "progressive"}
    two_pool = config.exploration_schedule in {"two_stage", "adaptive", "progressive"}
    final_pool = (two_pool and step >= 1 and bool(state.token_ids)
                  and config.rollout_count == 1 and config.block_size >= config.total_length - state.fixed)
    first_pool = (two_pool and step == 0 and not state.token_ids
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
    pilot_active = adaptive and first_pool and config.pilot_gate and candidate_count > 1 and width < remaining
    default_pilot = min(width, max(1, width // 2))
    pilot_width = min(config.pilot_tokens if config.pilot_tokens is not None else default_pilot, width)
    selective_k = (
        adaptive
        and config.selective_lookahead
        and rollout_count > 1
        and candidate_count > 1
        and width < remaining
        and (first_pool or config.exploration_schedule == "progressive")
    )
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
    planned_rollouts: list[int | None] = [None] * candidate_count
    routes: dict[str, tuple[int, int, bool]] = {}
    leaf_requests: dict[tuple[int, int], str] = {}
    buffers: dict[str, list[int]] = {}
    submitted: dict[str, GenerationRequest] = {}
    finished: set[str] = set()
    cancelled: set[str] = set()
    clean_completed_lengths: dict[int, list[int]] = {}
    clean_completed_rewards: dict[int, list[float]] = {}
    evaluations = 0
    streaming = bool(getattr(adapter.backend, "supports_streaming_generation", False))
    if selection_ready and not streaming:
        raise ValueError("selection_ready requires a streaming generation backend")
    straggler_guard = adaptive and config.straggler_pruning and streaming and adapter.prefix_reward is not None
    need_intermediate = (
        selection_ready
        or straggler_guard
        or (pilot_active and pilot_width < width and rollout_count > 1)
    )
    session: GenerationSession = (getattr(adapter.backend, "open_generation_session")() if streaming
                                  else BatchGenerationSession(adapter.backend))
    pending: dict[tuple[int, int], tuple[int, ...]] = {}
    pilot_escalated = not pilot_active
    siblings_dispatched = not selective_k

    def request(index, rollout, prefix, length, *, root=False, continuation=False):
        stem = f"conditional-is:step:{step}:candidate:{index}"
        seed = seeds.derive("conditional_is", step, "candidate", index) if root or continuation else seeds.derive(
            "conditional_is", step, "candidate", index, "rollout", rollout)
        key = stem if root else stem + (":continuation" if continuation else f":rollout:{rollout}")
        routes[key] = index, rollout, root
        if selection_ready or straggler_guard:
            leaf_requests[index, rollout] = key
        buffers[key] = []
        job = GenerationRequest(prompt + fixed + prefix, length, adapter.sampling, seed, key,
                                uniform_offset=len(prefix) if continuation else 0,
                                intermediate_events=need_intermediate)
        submitted[key] = job
        return job

    def branch(index, *, continuation=False, count: int | None = None):
        block = blocks[index]
        assert block is not None
        length = remaining - len(block.token_ids)
        target_k = rollout_count if count is None else count
        if block.finish_reason != "length" or length == 0:
            planned_rollouts[index] = 1
            return
        planned_rollouts[index] = target_k
        jobs = [request(index, k, block.token_ids, length) for k in range(1, target_k)]
        if continuation:
            jobs.insert(0, request(index, 0, block.token_ids, length, continuation=True))
        if jobs:
            session.submit(jobs)

    def dispatch_selective_siblings():
        """Assign K rollouts to the top ceil(M/2) non-terminal prefixes, 1 to the rest."""
        nonlocal siblings_dispatched
        if siblings_dispatched:
            return
        active = [idx for idx, blk in enumerate(blocks)
                  if blk is not None and blk.finish_reason == "length" and len(blk.token_ids) < remaining]
        for idx, blk in enumerate(blocks):
            if blk is not None and idx not in active:
                planned_rollouts[idx] = 1
        top_set: set[int]
        if len(active) > 1 and adapter.prefix_reward is not None:
            scored_prefixes: list[tuple[float, int]] = []
            for idx in active:
                blk = blocks[idx]
                assert blk is not None
                val = adapter.prefix_reward(prompt, fixed + blk.token_ids)
                if val is None or not isfinite(val):
                    scored_prefixes.clear()
                    break
                scored_prefixes.append((float(val), idx))
            if scored_prefixes:
                keep_count = max(1, ceil(len(active) / 2))
                scored_prefixes.sort(key=lambda item: (-item[0], item[1]))
                top_set = {idx for _, idx in scored_prefixes[:keep_count]}
            else:
                top_set = set(active)
        else:
            top_set = set(active)
        for idx in active:
            branch(idx, count=rollout_count if idx in top_set else 1)
        siblings_dispatched = True

    def evaluate(index, rollout, suffix, *, clean: bool = True):
        if not streaming:
            pending[index, rollout] = suffix
            return
        score([(index, rollout, suffix, clean)])

    def score(items):
        nonlocal evaluations
        values = tuple(adapter.reward(prompt, [fixed + blocks[index].token_ids + suffix for index, _, suffix, _ in items]))
        if len(values) != len(items) or any(not isfinite(float(value)) for value in values):
            raise ValueError("reward must return one finite value per completion")
        for (index, rollout, suffix, clean), value in zip(items, values, strict=True):
            value = float(value)
            leaves[index][rollout] = RolloutEvaluation(suffix, value, value / config.reward_temperature)
            if clean:
                blk = blocks[index]
                assert blk is not None
                clean_completed_lengths.setdefault(index, []).append(len(blk.token_ids) + len(suffix))
                clean_completed_rewards.setdefault(index, []).append(value)
        evaluations += len(items)

    def prune_stagnating_stragglers():
        """Abort extreme runaway rollouts that substantially outrun completed peers while trailing median reward."""
        if not straggler_guard or not pilot_escalated:
            return
        done_groups = len(clean_completed_lengths)
        min_required = 1 if final_pool else max(2, ceil(candidate_count / 2))
        if done_groups < min_required:
            return
        all_lengths = [length for group_lens in clean_completed_lengths.values() for length in group_lens]
        all_rewards = [rew for group_rews in clean_completed_rewards.values() for rew in group_rews]
        if not all_lengths or not all_rewards:
            return
        med_len = median(all_lengths)
        med_reward = median(all_rewards)
        # Anchor floor to block size B (state.fixed on Stage 2 where width == remaining).
        block_anchor = min(state.fixed, width) if final_pool and state.fixed > 0 else width
        if final_pool:
            threshold_len = (
                max(2 * block_anchor, int(1.75 * med_len))
                if done_groups == 1
                else max(2 * block_anchor, int(1.5 * med_len))
            )
        else:
            threshold_len = max(2 * block_anchor, int(2.0 * med_len))
        assert adapter.prefix_reward is not None
        for key in list(routes):
            if key in finished or key in cancelled:
                continue
            idx, rollout, is_root = routes[key]
            # On Step 0, wait until block B is known so selective K and group structure stay intact.
            # On Stage 2 (final_pool), blocks[idx] stays None until completion, so read the root buffer directly.
            if not final_pool and blocks[idx] is None:
                continue
            buf = buffers.get(key)
            if not buf:
                continue
            if is_root:
                obs_seq = tuple(buf)
            else:
                blk = blocks[idx]
                assert blk is not None
                obs_seq = blk.token_ids + tuple(buf)
            if len(obs_seq) < threshold_len:
                continue
            proxy = adapter.prefix_reward(prompt, fixed + obs_seq)
            if proxy is not None and isfinite(proxy) and proxy < med_reward:
                session.stop(key)
                cancelled.add(key)
                buffers.pop(key, None)

    try:
        carried = int(bool(state.token_ids))
        # Forward token slots of the requests this step measured; an unmeasured one makes the total unknown.
        slots: int = 0
        measured = True
        if carried:
            end = state.fixed + width
            carried_block = Block(state.token_ids[state.fixed:end], "length" if end < len(state.token_ids) else "stop")
            blocks[0] = carried_block
            leaves[0][0] = RolloutEvaluation(state.token_ids[end:], state.reward, state.reward / config.reward_temperature)
            planned_rollouts[0] = 1 if carried_block.finish_reason != "length" or end >= len(state.token_ids) else rollout_count
            if len(state.token_ids) < config.total_length:
                clean_completed_lengths.setdefault(0, []).append(len(state.token_ids) - state.fixed)
                clean_completed_rewards.setdefault(0, []).append(state.reward)
        if pilot_active:
            pilot_job = request(0, 0, (), length_limit, root=True)
            session.submit([pilot_job], phase="prefix", checkpoint=pilot_width)
            sealed = False
        else:
            roots = [request(index, 0, (), min(width, length_limit) if config.block_first else length_limit, root=True)
                     for index in range(carried, candidate_count)]
            if roots:
                checkpoint = width if rollout_count > 1 and not config.block_first else None
                session.submit(roots, phase="prefix", checkpoint=checkpoint)
            if not selective_k:
                for index in range(carried):
                    branch(index)
            sealed = rollout_count == 1 and not config.block_first
            if sealed:
                session.seal()

        def expected(index: int, block: Block | None) -> int:
            planned_count = planned_rollouts[index]
            if planned_count is not None:
                return planned_count
            if block is None:
                # A missing prefix can still create K leaves. Use that upper bound
                # until it terminates, rather than waiting for an unusable group.
                return rollout_count if width < remaining else 1
            return rollout_count if block.finish_reason == "length" and len(block.token_ids) < remaining else 1

        def selection_is_ready():
            nonlocal chosen, forecast
            if not selection_ready:
                return False
            assert adapter.prefix_reward is not None and uniform is not None
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
                planned = expected(index, block)
                for rollout in range(planned):
                    done = rollout in group
                    if done:
                        assert block is not None
                        value = group[rollout].log_weight
                        observed_length = len(block.token_ids) + len(group[rollout].token_ids)
                    else:
                        key = leaf_requests.get((index, rollout))
                        if key is not None:
                            root = routes[key][2]
                            if root:
                                sequence = tuple(buffers[key])
                            else:
                                assert block is not None
                                sequence = block.token_ids + tuple(buffers[key])
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
                planned_counts = [expected(idx, block) for idx, block in enumerate(blocks)]
                chosen = _complete_priority_group(logs, complete, keys, planned_counts, observed_lengths,
                                                  priorities, completion_uniforms, width)
                if chosen is None:
                    return False
                from inference_scaling.arllm.algorithms.conditional_is import forecast_candidate_count

                observations = tuple(
                    (key[0], planned_counts[key[0]], state.fixed + length, value + log(planned_counts[key[0]]))
                    for key, length, value in zip(keys, observed_lengths, logs, strict=True)
                    if key[0] != chosen[0])
                chosen_block = blocks[chosen[0]]
                assert chosen_block is not None
                forecast = forecast_candidate_count(
                    observations, state.fixed + len(chosen_block.token_ids), config, seeds)
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
            if (not sealed and pilot_escalated and siblings_dispatched
                    and all(block is not None for block in blocks)):
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

            # Pilot gate escalation check on candidate 0 before finishing or crossing pilot_width.
            if pilot_active and not pilot_escalated and index == 0 and root:
                if event.finished and sample.finish_reason != "length" and len(tokens) < pilot_width:
                    # Clean short completion within the pilot horizon: early-exit at 1x cost!
                    finished.add(key)
                    if sample.forward_token_slots is None:
                        measured = False
                    else:
                        slots += sample.forward_token_slots
                    pilot_block = Block(tuple(tokens), "stop")
                    blocks[0] = pilot_block
                    planned_rollouts[0] = 1
                    evaluate(0, 0, (), clean=True)
                    del buffers[key]
                    if pending:
                        score([(i, r, s, True) for (i, r), s in sorted(pending.items())])
                        pending.clear()
                    # Only candidate 0 ran; mark forecast=0 and exit cleanly.
                    result = weigh([pilot_block.token_ids], [[leaves[0][0]]])
                    return result, (slots if measured else None), (0,), evaluations, (0, 0), 0
                if len(tokens) >= pilot_width or event.finished:
                    pilot_escalated = True
                    other_roots = [request(other, 0, (), length_limit, root=True)
                                   for other in range(1, candidate_count)]
                    if other_roots:
                        checkpoint = width if rollout_count > 1 else None
                        session.submit(other_roots, phase="prefix", checkpoint=checkpoint)
                    if rollout_count == 1:
                        session.seal()
                        sealed = True

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
                            evaluate(index, 0, (), clean=sample.finish_reason != "length")
                elif len(tokens) >= width or event.finished:
                    # A stopped sequence of exactly B tokens has one empty leaf.
                    terminal = event.finished and len(tokens) <= width
                    blocks[index] = Block(tuple(tokens[:width]), "stop" if terminal else "length")
                    if selective_k:
                        if pilot_escalated and all(blk is not None for blk in blocks):
                            dispatch_selective_siblings()
                    else:
                        branch(index)
            if event.finished:
                if not (root and config.block_first):
                    block = blocks[index]
                    assert block is not None
                    clean = sample.finish_reason != "length"
                    evaluate(index, rollout, tuple(tokens[len(block.token_ids):] if root else tokens), clean=clean)
                del buffers[key]
            if straggler_guard:
                prune_stagnating_stragglers()
            if selection_is_ready():
                # No further event is consumed after selection. The finally
                # block closes this pool and cancels all producers together,
                # instead of waiting for each request's abort in sequence.
                cancelled.update(routes.keys() - finished)
                buffers.clear()
                break
        if pending:
            score([(index, rollout, suffix, True) for (index, rollout), suffix in sorted(pending.items())])
            pending.clear()
        pruned_any = bool(cancelled)
        if not selection_ready and not pruned_any and any(
                block is None and index not in excluded for index, block in enumerate(blocks)):
            raise RuntimeError("stream omitted a candidate prefix")
        canonical = []
        canonical_blocks = []
        indices = []
        for index, (block, group) in enumerate(zip(blocks, leaves, strict=True)):
            complete = set(group) == set(range(expected(index, block)))
            if not selection_ready and not pruned_any and not complete and index not in excluded:
                raise RuntimeError("stream omitted a required completion")
            if group:
                assert block is not None
                indices.append(index)
                canonical_blocks.append(block)
                # Keep the planned K denominator when selection_ready left siblings unfinished.
                correction = log(len(group) / expected(index, block)) if selection_ready else 0.
                canonical.append([replace(group[k], log_weight=group[k].log_weight + correction)
                                  if selection_ready else group[k] for k in sorted(group)])
        # ESS-bounded dynamic reward temperature calibration in adaptive/progressive modes.
        if adaptive and config.adaptive_temperature and len(canonical) >= 2 and not selection_ready:
            raw_groups = [[item.reward for item in group] for group in canonical]
            min_scale = 0.15 if (config.exploration_schedule == "progressive" and step >= 1) else 0.5
            tau_eff = calibrate_reward_temperature(
                raw_groups, config.reward_temperature, min_scale=min_scale, min_ess=1.5,
            )
            if tau_eff != config.reward_temperature:
                canonical = [
                    [replace(item, log_weight=item.reward / tau_eff) for item in group]
                    for group in canonical
                ]
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


def evaluate_particle_pool(
    adapter: ConditionalISAdapter,
    particles: list[RetainedSequence],
    step: int,
    seeds: SeedStream,
    *,
    block_width: int,
    total_candidates: int,
    rollout_count: int,
    is_final: bool,
    base_block_size: int,
):
    """Jointly expand ``P >= 2`` retained particles inside one shared generation session.

    Each particle contributes its retained complete sequence as slot 0 of its
    sub-tree (zero recomputation) and expands fresh suffixes from its committed
    prefix. Exact parent-mass normalization (``-log(surviving_children_p)``)
    conserves equal ``1/P`` prior particle mass even when some particles have
    already terminated cleanly while others continue branching.
    """
    config, prompt = adapter.config, adapter.prompt
    p_count = len(particles)
    if p_count < 1:
        raise ValueError("particle pool requires at least one particle")

    is_term = [bool(p.token_ids) and p.fixed >= len(p.token_ids) for p in particles]
    rems = [max(0, config.total_length - p.fixed) for p in particles]
    widths = [rems[idx] if is_final else min(block_width, rems[idx]) for idx in range(p_count)]
    fixed_prefixes = [p.token_ids[:p.fixed] for p in particles]

    length_limits: list[int] = list(rems)
    min_lengths: list[float | None] = [None] * p_count
    if is_final:
        for idx, part in enumerate(particles):
            if not is_term[idx] and rems[idx] > 0:
                bound = len(part.token_ids) * (
                    2.0 ** float(seeds.generator("conditional_is", step, "particle", idx, "length_window").random())
                )
                length_limits[idx] = min(rems[idx], max(0, floor(bound - part.fixed)))
                min_lengths[idx] = bound / 2.0 - part.fixed

    active_ps = [idx for idx in range(p_count) if not is_term[idx] and length_limits[idx] > 0]
    fresh_counts = [0] * p_count
    if active_ps:
        fresh_budget = max(len(active_ps), total_candidates - p_count)
        if config.final_pool_limit is not None:
            fresh_budget = min(fresh_budget, config.final_pool_limit)
        q, r = divmod(fresh_budget, len(active_ps))
        for pos, p_idx in enumerate(active_ps):
            fresh_counts[p_idx] = q + (1 if pos < r else 0)

    slot_parent: list[int] = []
    slot_local: list[int] = []
    for p_idx in range(p_count):
        slot_parent.append(p_idx)
        slot_local.append(0)
        for local_i in range(1, 1 + fresh_counts[p_idx]):
            slot_parent.append(p_idx)
            slot_local.append(local_i)

    total_slots = len(slot_parent)
    blocks: list[Block | None] = [None] * total_slots
    leaves: list[dict[int, RolloutEvaluation]] = [{} for _ in range(total_slots)]
    planned_rollouts: list[int | None] = [None] * total_slots
    excluded: set[int] = set()
    routes: dict[str, tuple[int, int, bool]] = {}
    buffers: dict[str, list[int]] = {}
    submitted: dict[str, GenerationRequest] = {}
    finished: set[str] = set()
    cancelled: set[str] = set()
    clean_completed_lengths: dict[int, list[int]] = {}
    clean_completed_rewards: dict[int, list[float]] = {}
    evaluations = 0
    pending: dict[tuple[int, int], tuple[int, ...]] = {}

    streaming = bool(getattr(adapter.backend, "supports_streaming_generation", False))
    straggler_guard = config.straggler_pruning and streaming and adapter.prefix_reward is not None
    effective_k = 1 if is_final else rollout_count
    selective_k = config.selective_lookahead and effective_k > 1 and total_slots > 1 and not is_final
    need_intermediate = straggler_guard
    session: GenerationSession = (getattr(adapter.backend, "open_generation_session")() if streaming
                                  else BatchGenerationSession(adapter.backend))
    siblings_dispatched = not selective_k

    def make_request(slot: int, rollout: int, prefix_extra: tuple[int, ...], max_len: int, *, root: bool = False):
        p_idx, local_i = slot_parent[slot], slot_local[slot]
        stem = f"conditional-is:step:{step}:particle:{p_idx}:candidate:{local_i}"
        seed = (
            seeds.derive("conditional_is", step, "particle", p_idx, "candidate", local_i)
            if root
            else seeds.derive("conditional_is", step, "particle", p_idx, "candidate", local_i, "rollout", rollout)
        )
        key = stem if root else f"{stem}:rollout:{rollout}"
        routes[key] = (slot, rollout, root)
        buffers[key] = []
        job = GenerationRequest(
            prompt + fixed_prefixes[p_idx] + prefix_extra,
            max_len,
            adapter.sampling,
            seed,
            key,
            intermediate_events=need_intermediate,
        )
        submitted[key] = job
        return job

    def branch_slot(slot: int, *, count: int | None = None) -> None:
        blk = blocks[slot]
        assert blk is not None
        p_idx = slot_parent[slot]
        rem_after = rems[p_idx] - len(blk.token_ids)
        target_k = effective_k if count is None else count
        if blk.finish_reason != "length" or rem_after <= 0 or target_k <= 1:
            planned_rollouts[slot] = 1
            return
        planned_rollouts[slot] = target_k
        jobs = [make_request(slot, k, blk.token_ids, rem_after) for k in range(1, target_k)]
        if jobs:
            session.submit(jobs)

    def dispatch_selective_siblings() -> None:
        nonlocal siblings_dispatched
        if siblings_dispatched:
            return
        active = [
            slot for slot, blk in enumerate(blocks)
            if blk is not None and blk.finish_reason == "length" and len(blk.token_ids) < rems[slot_parent[slot]]
        ]
        for slot, blk in enumerate(blocks):
            if blk is not None and slot not in active:
                planned_rollouts[slot] = 1
        top_set: set[int]
        if len(active) > 1 and adapter.prefix_reward is not None:
            scored: list[tuple[float, int]] = []
            for slot in active:
                blk = blocks[slot]
                assert blk is not None
                val = adapter.prefix_reward(prompt, fixed_prefixes[slot_parent[slot]] + blk.token_ids)
                if val is None or not isfinite(val):
                    scored.clear()
                    break
                scored.append((float(val), slot))
            if scored:
                keep_count = max(1, ceil(len(active) / 2))
                scored.sort(key=lambda item: (-item[0], item[1]))
                top_set = {slot for _, slot in scored[:keep_count]}
            else:
                top_set = set(active)
        else:
            top_set = set(active)
        for slot in active:
            branch_slot(slot, count=effective_k if slot in top_set else 1)
        siblings_dispatched = True

    def score_items(items: list[tuple[int, int, tuple[int, ...], bool]]) -> None:
        nonlocal evaluations
        full_seqs = [
            fixed_prefixes[slot_parent[slot]] + blocks[slot].token_ids + suffix  # type: ignore[union-attr]
            for slot, _, suffix, _ in items
        ]
        values = tuple(adapter.reward(prompt, full_seqs))
        if len(values) != len(items) or any(not isfinite(float(v)) for v in values):
            raise ValueError("reward must return one finite value per completion")
        for (slot, rollout, suffix, clean), value in zip(items, values, strict=True):
            fval = float(value)
            leaves[slot][rollout] = RolloutEvaluation(suffix, fval, fval / config.reward_temperature)
            if clean:
                blk = blocks[slot]
                assert blk is not None
                clean_completed_lengths.setdefault(slot, []).append(len(blk.token_ids) + len(suffix))
                clean_completed_rewards.setdefault(slot, []).append(fval)
        evaluations += len(items)

    def evaluate_leaf(slot: int, rollout: int, suffix: tuple[int, ...], *, clean: bool = True) -> None:
        if not streaming:
            pending[slot, rollout] = suffix
            return
        score_items([(slot, rollout, suffix, clean)])

    def prune_stragglers() -> None:
        if not straggler_guard:
            return
        done_groups = len(clean_completed_lengths)
        min_required = 1 if is_final else max(2, ceil(total_slots / 2))
        if done_groups < min_required:
            return
        all_lens = [length for group_lens in clean_completed_lengths.values() for length in group_lens]
        all_rews = [rew for group_rews in clean_completed_rewards.values() for rew in group_rews]
        if not all_lens or not all_rews:
            return
        med_len = median(all_lens)
        med_rew = median(all_rews)
        if is_final:
            threshold_len = (
                max(2 * base_block_size, int(1.75 * med_len))
                if done_groups == 1
                else max(2 * base_block_size, int(1.5 * med_len))
            )
        else:
            threshold_len = max(2 * base_block_size, int(2.0 * med_len))
        assert adapter.prefix_reward is not None
        for key in list(routes):
            if key in finished or key in cancelled:
                continue
            slot, _rollout, is_root = routes[key]
            if not is_final and blocks[slot] is None:
                continue
            buf = buffers.get(key)
            if not buf:
                continue
            if is_root:
                obs_seq = tuple(buf)
            else:
                blk = blocks[slot]
                assert blk is not None
                obs_seq = blk.token_ids + tuple(buf)
            if len(obs_seq) < threshold_len:
                continue
            proxy = adapter.prefix_reward(prompt, fixed_prefixes[slot_parent[slot]] + obs_seq)
            if proxy is not None and isfinite(proxy) and proxy < med_rew:
                session.stop(key)
                cancelled.add(key)
                buffers.pop(key, None)

    try:
        slots_cost = 0
        measured = True
        root_jobs: list[GenerationRequest] = []
        retained_slots: list[int] = []
        for slot in range(total_slots):
            p_idx, local_i = slot_parent[slot], slot_local[slot]
            part = particles[p_idx]
            if local_i == 0:
                retained_slots.append(slot)
                if is_term[p_idx]:
                    blocks[slot] = Block((), "stop")
                    leaves[slot][0] = RolloutEvaluation((), part.reward, part.reward / config.reward_temperature)
                    planned_rollouts[slot] = 1
                else:
                    end = part.fixed + widths[p_idx]
                    blk_toks = part.token_ids[part.fixed:end]
                    blk_reason = "length" if end < len(part.token_ids) else "stop"
                    blocks[slot] = Block(blk_toks, blk_reason)
                    leaves[slot][0] = RolloutEvaluation(
                        part.token_ids[end:], part.reward, part.reward / config.reward_temperature,
                    )
                    planned_rollouts[slot] = (
                        1 if blk_reason != "length" or end >= len(part.token_ids) else effective_k
                    )
                if len(part.token_ids) < config.total_length:
                    clean_completed_lengths.setdefault(slot, []).append(len(part.token_ids) - part.fixed)
                    clean_completed_rewards.setdefault(slot, []).append(part.reward)
            else:
                root_jobs.append(make_request(slot, 0, (), length_limits[p_idx], root=True))

        if root_jobs:
            checkpoint = block_width if effective_k > 1 and not is_final else None
            session.submit(root_jobs, phase="prefix", checkpoint=checkpoint)
        if not selective_k:
            for slot in retained_slots:
                branch_slot(slot)
        elif not root_jobs:
            dispatch_selective_siblings()
        sealed = effective_k == 1
        if sealed:
            session.seal()

        while len(finished) + len(cancelled) < len(routes):
            if not sealed and siblings_dispatched and all(blk is not None for blk in blocks):
                session.seal()
                sealed = True
            event = session.next_event()
            sample = event.delta
            key = sample.request_id
            if key not in routes or key in finished or key in cancelled:
                raise RuntimeError("stream returned an unknown or completed request")
            job = submitted[key]
            slot, rollout, is_root = routes[key]
            p_idx = slot_parent[slot]
            tokens = buffers[key]
            tokens.extend(sample.token_ids)
            if len(tokens) > job.max_new_tokens:
                raise RuntimeError("stream exceeded its generation limit")
            if event.finished:
                if not tokens:
                    raise RuntimeError("generation produced an empty sequence")
                finished.add(key)
                if sample.forward_token_slots is None:
                    measured = False
                else:
                    slots_cost += sample.forward_token_slots
                min_len = min_lengths[p_idx]
                if min_len is not None and (
                    (sample.finish_reason == "length" and length_limits[p_idx] < rems[p_idx])
                    or len(tokens) <= min_len
                ):
                    excluded.add(slot)
                    del buffers[key]
                    continue
            if is_root and blocks[slot] is None and (len(tokens) >= widths[p_idx] or event.finished):
                terminal = event.finished and len(tokens) <= widths[p_idx]
                blocks[slot] = Block(tuple(tokens[:widths[p_idx]]), "stop" if terminal else "length")
                if selective_k:
                    if all(blk is not None for blk in blocks):
                        dispatch_selective_siblings()
                else:
                    branch_slot(slot)
            if event.finished:
                blk = blocks[slot]
                assert blk is not None
                clean = sample.finish_reason != "length"
                evaluate_leaf(slot, rollout, tuple(tokens[len(blk.token_ids):] if is_root else tokens), clean=clean)
                del buffers[key]
            if straggler_guard:
                prune_stragglers()

        if pending:
            score_items([(slot, rollout, suffix, True) for (slot, rollout), suffix in sorted(pending.items())])
            pending.clear()

        canonical: list[list[RolloutEvaluation]] = []
        canonical_blocks: list[Block] = []
        origins: list[tuple[int, tuple[int, ...]]] = []
        surviving_per_parent: dict[int, int] = {}
        for slot, (blk, group) in enumerate(zip(blocks, leaves, strict=True)):
            if group:
                assert blk is not None
                p_idx = slot_parent[slot]
                surviving_per_parent[p_idx] = surviving_per_parent.get(p_idx, 0) + 1
                origins.append((p_idx, fixed_prefixes[p_idx]))
                canonical_blocks.append(blk)
                canonical.append([group[k] for k in sorted(group)])

        if config.adaptive_temperature and len(canonical) >= 2:
            raw_groups = [[item.reward for item in group] for group in canonical]
            tau_eff = calibrate_reward_temperature(
                raw_groups, config.reward_temperature, min_scale=0.15, min_ess=1.5,
            )
            if tau_eff != config.reward_temperature:
                canonical = [
                    [replace(item, log_weight=item.reward / tau_eff) for item in group]
                    for group in canonical
                ]

        # Exact SMC parent-mass conservation: each parent particle splits its 1/P prior
        # equally across its surviving children so terminal particles are never downweighted.
        normalized_groups: list[list[RolloutEvaluation]] = []
        for (p_idx, _), grp in zip(origins, canonical, strict=True):
            parent_children = surviving_per_parent[p_idx]
            shift = log(parent_children)
            normalized_groups.append(
                [replace(item, log_weight=item.log_weight - shift) for item in grp]
            )
        result = weigh([blk.token_ids for blk in canonical_blocks], normalized_groups)
        if len(finished) != len(routes):
            measured = False
        return result, origins, (slots_cost if measured else None), evaluations
    finally:
        session.close()


__all__ = ["evaluate_candidates", "evaluate_particle_pool"]
