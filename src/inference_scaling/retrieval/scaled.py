"""Budgeted suffix search after a frozen, step-boundary evidence checkpoint.

The old stepwise experiment is deliberately unchanged. A search epoch has one
immutable prompt; no score or retained candidate crosses an evidence update.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import math

from inference_scaling.arllm.algorithms.conditional_is import ConditionalISAdapter
from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, TokenStatistic
from inference_scaling.retrieval.stepwise import StepController, StepState, token_hash
from inference_scaling.shared.rng import SeedStream


@dataclass(frozen=True)
class ScalingConfig:
    work_tokens: int = 8192
    answer_tokens: int = 512
    leaf_tokens: int = 1024
    candidates: int = 8
    block_tokens: int = 256
    is_candidates: int = 4
    rollouts: int = 2
    reward_temperature: float = .25
    probe_tokens: int = 256
    context_tokens: int = 8192


class CheckpointReady(Exception):
    pass


class ProbeController(StepController):
    """Reuse the frozen FLARE/gap gate, then hand a complete-step prefix to search."""
    scan_tokens = 256

    def _sample(self, state, prefix, count, purpose):
        if purpose == "thinking":
            count = min(count, self.scan_tokens-(state.work-state.auxiliary))
            if count <= 0:
                raise CheckpointReady("scan_cap")
        return super()._sample(state, prefix, count, purpose)

    def _gap(self, state, n):
        packet = super()._gap(state, n)
        if packet:
            return packet
        if state.retrievals or state.checks >= self.cfg.planner_checks:
            raise CheckpointReady("checks_complete")
        return None

    def probe(self, question, base, seed):
        state = StepState(tuple(base), seed, question)
        packet = None
        try:
            packet = self.thinking(state, "flare_gap")
        except CheckpointReady as ready:
            state.discarded = state.work-state.auxiliary-len(state.reason)
            state.events.append({"kind": "search_checkpoint", "cause": str(ready),
                "confirmed_offset": len(state.reason), "discarded_ids": list(state.pending),
                "discarded_work_including_terminal": state.discarded})
            state.pending.clear(); state.pending_lp.clear(); state.pending_sc.clear()
        self.answer(state)  # Only a natural close before the checkpoint permits this.
        return state, packet


def pool_upper_bound(config, fixed, carried):
    """Reserve a whole tree before generating; terminal leaves can only cost less."""
    remaining = config.total_length-fixed
    width = min(config.block_size, remaining)
    fresh = config.candidate_count-int(carried)
    return (fresh*(width+config.rollout_count*(remaining-width))
            + int(carried)*(config.rollout_count-1)*(remaining-width))


class SearchLedger:
    """Raw generation and statistics, scoped to exactly one frozen search root.

    This batch-only adapter uses the repository's existing complete-pool IS
    implementation. Reading a reward never causes another model forward.
    """
    supports_streaming_generation = False

    def __init__(self, backend, root, close, cap, seed, context_tokens):
        self.backend, self.root, self.close = backend, tuple(root), tuple(close)
        self.model_id = backend.model_id
        self.cap, self.seed, self.context_tokens = cap, seed, context_tokens
        self.eos = backend.tokenizer.eos_token_id
        self.sampling = SamplingConfig(eos_token_id=self.eos)
        self.calls, self.work = [], 0

    def sample_batch(self, requests):
        jobs = []
        reservation = sum(r.max_new_tokens for r in requests)
        if reservation > self.cap-self.work:
            raise RuntimeError("unfunded generation batch")
        for r in requests:
            if tuple(r.prefix[:len(self.root)]) != self.root:
                raise RuntimeError("search context changed inside an epoch")
            if len(r.prefix)+r.max_new_tokens+len(self.close)+512 > self.context_tokens:
                raise RuntimeError("search would consume the reserved answer context")
            jobs.append(replace(r, statistic=TokenStatistic(r.sampling),
                stop_sequences=(self.close,), intermediate_events=False))
        samples = self.backend.sample_batch(jobs)
        if len(samples) != len(jobs):
            raise RuntimeError("backend omitted a sample")
        for job, sample in zip(jobs, samples, strict=True):
            ids = tuple(sample.token_ids)
            if tuple(sample.prefix) != tuple(job.prefix) or len(ids) > job.max_new_tokens:
                raise RuntimeError("generation contract violated")
            stats = tuple(sample.token_statistics or ())
            if len(stats) != len(ids) or any(not math.isfinite(x) for x in stats):
                raise RuntimeError("missing/nonfinite native SC statistic")
            if self.close and self.close[0] in ids:
                at = ids.index(self.close[0])
                if ids[at:] != self.close:
                    raise RuntimeError("thinking stop did not terminate the request")
            self.work += len(ids)
            self.calls.append({"purpose": "search", "request_id": job.request_id, "seed": job.seed,
                "prefix_ids": list(job.prefix), "prefix_sha256": token_hash(job.prefix),
                "max_new_tokens": job.max_new_tokens, "token_ids": list(ids),
                "token_logprobs": list(sample.token_logprobs), "token_statistics": list(stats),
                "finish_reason": sample.finish_reason})
        if self.work > self.cap:
            raise RuntimeError("work budget exceeded")
        return samples

    def reward(self, prompt, completions):
        if tuple(prompt) != self.root:
            raise RuntimeError("reward root differs from the generation root")
        rewards = []
        for completion in completions:
            ids = tuple(completion)
            end = next((i for i, t in enumerate(ids) if t in {self.eos, self.close[0]}), len(ids))
            target = self.root+ids[:end]
            values = [None]*end
            for call in self.calls:
                prefix, generated = tuple(call["prefix_ids"]), tuple(call["token_ids"])
                start = len(prefix)-len(self.root)
                if start < 0 or target[:len(prefix)] != prefix:
                    continue
                for i, (token, stat) in enumerate(zip(generated, call["token_statistics"], strict=True)):
                    at = start+i
                    if at >= end or token != ids[at]:
                        break
                    # Identical context distributions can have tiny batching
                    # rounding differences. Keep the first actual observation.
                    if values[at] is None:
                        values[at] = stat
            if any(value is None for value in values):
                raise RuntimeError("SC reward requires uncached statistics")
            rewards.append(math.fsum(values)/end if end else 0.)
        return tuple(rewards)

    def request(self, length, index, prefix=()):
        return GenerationRequest(self.root+tuple(prefix), length, self.sampling,
            SeedStream(self.seed).derive("candidate", index), f"candidate:{index}",
            stop_sequences=(self.close,), intermediate_events=False)


def search_suffix(ledger, method, cfg):
    available_context = cfg.context_tokens-len(ledger.root)-len(ledger.close)-cfg.answer_tokens
    if available_context <= 0:
        raise RuntimeError("no context left for search")
    if method == "single":
        length = min(ledger.cap, available_context)
        sample, = ledger.sample_batch([ledger.request(length, 0)])
        return tuple(sample.token_ids), {"method": method, "candidate_count": 1,
            "selected_index": 0, "context_limited": length < ledger.cap, "steps": []}
    horizon = min(cfg.leaf_tokens, available_context)
    if method == "sc_bon":
        count = min(cfg.candidates, ledger.cap//horizon)
        if count < 1:
            raise RuntimeError("cannot fund a complete BoN candidate")
        samples = ledger.sample_batch([ledger.request(horizon, i) for i in range(count)])
        values = ledger.reward(ledger.root, [s.token_ids for s in samples])
        chosen = max(range(count), key=lambda i: (values[i], -i))
        return tuple(samples[chosen].token_ids), {"method": method, "candidate_count": count,
            "scores": values, "selected_index": chosen, "horizon": horizon, "steps": []}
    if method != "block_met_sc":
        raise ValueError("unknown search method")
    config = ConditionalISConfig(candidate_count=cfg.is_candidates, rollout_count=cfg.rollouts,
        block_size=min(cfg.block_tokens, horizon), total_length=horizon,
        reward_temperature=cfg.reward_temperature, block_first=True,
        exploration_schedule="fixed", selection_ready=False, selection="block_metropolized")
    adapter = ConditionalISAdapter(backend=ledger, prompt=ledger.root, config=config,
        sampling=ledger.sampling, reward=ledger.reward)
    state, records = adapter.initial_state, []
    seeds = SeedStream(ledger.seed)
    while not adapter.is_terminal(state):
        bound = pool_upper_bound(config, state.fixed, bool(state.token_ids))
        if bound > ledger.cap-ledger.work:
            if not state.token_ids:
                raise RuntimeError("cannot fund the initial complete IS pool")
            break
        before = ledger.work
        record, state = adapter.step(state, len(records), seeds)
        if ledger.work-before > bound:
            raise RuntimeError("IS reservation underestimated generation")
        records.append({**asdict(record), "reserved_work": bound, "actual_work": ledger.work-before})
    return tuple(state.token_ids), {"method": method, "config": asdict(config),
        "steps": records, "fixed_tokens": state.fixed,
        "stop": "terminal" if adapter.is_terminal(state) else "whole_pool_budget",
        "score": state.reward, "horizon": horizon}


def scaled_condition(ctrl, common, packet, content, method, cfg):
    state = copy.deepcopy(common)
    state.calls = []
    if state.phase != "thinking" or state.finish:
        return state, {"method": method, "bypassed_natural_end": True}
    if packet:
        state.payload = ctrl.evidence.variant(packet, content)
        state.placement = "user_reference_rebuild_at_step_boundary"
        state.events.append({"kind": "injection", "position": state.placement,
            "confirmed_offset": len(state.reason), "work_spent": state.work,
            "content": content, "payload_ids": list(state.payload), "packet": packet})
    root = ctrl.prefix(state)
    ledger = SearchLedger(ctrl.backend, root, ctrl.fmt.end_token_ids,
        cfg.work_tokens-state.work, state.seed, cfg.context_tokens)
    selected, audit = search_suffix(ledger, method, cfg)
    state.calls.extend(ledger.calls)
    state.work += ledger.work
    state.main_calls += len(ledger.calls)
    state.first_post = list(selected[:32]) if packet else []
    close = ctrl.fmt.end_token_ids
    if close and selected[-len(close):] == close:
        state.reason.extend(selected[:-len(close)])
        state.sampled_close, state.phase = True, "answer"
    elif ctrl.eos in selected:
        state.reason.extend(selected[:selected.index(ctrl.eos)])
        state.eos_in_thinking, state.finish = True, "eos"
    else:
        state.reason.extend(selected)
        state.forced, state.phase = True, "answer"
    ctrl.answer(state)
    audit.update({"root_sha256": token_hash(root), "root_ids": list(root),
        "selected_suffix_ids": list(selected), "search_work": ledger.work,
        "common_work": common.work, "unused_work_budget": cfg.work_tokens-state.work,
        "raw_search_eos_count": sum(ctrl.eos in c["token_ids"] for c in ledger.calls),
        "raw_immediate_close_count": sum(tuple(c["token_ids"][:len(close)]) == close for c in ledger.calls),
        "score_scope": "new suffix thinking positions only; excludes terminal markers and frozen common prefix"})
    return state, audit
