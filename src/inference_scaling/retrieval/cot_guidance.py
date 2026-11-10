"""Frozen short evidence, preserved reasoning, and complete-pool suffix search.

Experiment-local adapters leave the production backend and earlier studies intact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import types

from inference_scaling.arllm.algorithms.conditional_is import ConditionalISAdapter
from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, TokenStatistic
from inference_scaling.retrieval.scaled import pool_upper_bound
from inference_scaling.retrieval.stepwise import step_char_boundaries, token_hash
from inference_scaling.shared.rewards.consilience import confidence_windows
from inference_scaling.shared.rng import SeedStream


@dataclass(frozen=True)
class GuidanceConfig:
    context_tokens: int = 32768
    thinking_tokens: int = 8192
    answer_tokens: int = 16384
    answer_chunk: int = 4096
    search_tokens: int = 65536
    evidence_tokens: int = 256
    probe_tokens: int = 512
    probe_minimum: int = 64
    candidates: int = 8
    is_candidates: int = 3
    rollouts: int = 2
    block_tokens: int = 512


class DualStatistics:
    """Read both native SC and top-5 statistics from the SAME generated rows.

    The existing Ascend wire format already contains both, so no second forward
    or second sampling run is necessary. Scope is this backend instance only.
    """
    def __init__(self, backend):
        from inference_scaling.arllm.backends.vllm_statistics import scalar_generated_logprobs
        self.rows = {}
        original = backend._sample_from_output
        self.original, self.backend = original, backend

        def capture(instance, request, output, fused=None):
            result = original(request, output, fused)
            if request.statistic is not None and request.statistic.top_k == 5:
                sample = result[0]
                positions = instance._completion(output).logprobs
                _, sc = scalar_generated_logprobs(positions, sample.token_ids, None)
                _, cn = scalar_generated_logprobs(positions, sample.token_ids, 5)
                self.rows[request.request_id] = (tuple(sample.prefix), tuple(sample.token_ids), sc, cn)
            return result
        backend._sample_from_output = types.MethodType(capture, backend)

    def take(self, request, sample):
        prefix, ids, sc, cn = self.rows.pop(request.request_id)
        if prefix != tuple(request.prefix) or ids != tuple(sample.token_ids):
            raise RuntimeError("dual-statistic alignment failure")
        return sc, cn

    def close(self):
        self.backend._sample_from_output = self.original


def trajectory_score(sc, cn, lp, kind):
    if not sc:
        return 0.0
    if not (len(sc) == len(cn) == len(lp)):
        raise ValueError("unaligned trajectory statistics")
    if kind == "sc":
        return math.fsum(sc)/len(sc)
    if kind == "consilience":
        return confidence_windows(cn).score
    if kind == "logprob":
        return math.fsum(lp)/len(lp)
    raise ValueError(kind)


def completed_boundary(backend, ids, minimum):
    """Earliest complete step at/after minimum; never re-tokenize retained text."""
    text = backend.decode(tuple(ids), skip_special_tokens=False)
    positions = step_char_boundaries(text)
    for n in range(minimum, len(ids)+1):
        prefix = backend.decode(tuple(ids[:n]), skip_special_tokens=False)
        if text.startswith(prefix) and any(at <= len(prefix) and not text[at:len(prefix)].strip() for at in positions):
            return n
    return None


class GuidanceLedger:
    supports_streaming_generation = False

    def __init__(self, backend, tap, root, close, seed, cfg, *, carried=None, reward="sc", cap=None):
        self.backend, self.tap, self.root = backend, tap, tuple(root)
        self.close, self.seed, self.cfg = tuple(close), seed, cfg
        self.eos, self.model_id = backend.tokenizer.eos_token_id, backend.model_id
        self.sampling = SamplingConfig(eos_token_id=self.eos)
        self.calls, self.work = [], 0
        self.cap = cfg.search_tokens if cap is None else cap
        self.carried = carried or {"sc": [], "cn": [], "lp": []}
        self.reward_kind = reward

    def sample_batch(self, requests):
        if sum(r.max_new_tokens for r in requests) > self.cap-self.work:
            raise RuntimeError("incomplete candidate pool would exceed search cap")
        jobs = []
        for request in requests:
            if tuple(request.prefix[:len(self.root)]) != self.root:
                raise RuntimeError("evidence/root changed within search epoch")
            if len(request.prefix)+request.max_new_tokens+len(self.close)+self.cfg.answer_tokens > self.cfg.context_tokens:
                raise RuntimeError("insufficient context reserved for full final answer")
            jobs.append(replace(request, statistic=TokenStatistic(request.sampling, top_k=5),
                                stop_sequences=(self.close,), intermediate_events=False))
        samples = self.backend.sample_batch(jobs)
        if len(samples) != len(jobs):
            raise RuntimeError("missing sample")
        for job, sample in zip(jobs, samples, strict=True):
            sc, cn = self.tap.take(job, sample)
            if len(sample.token_ids) > job.max_new_tokens:
                raise RuntimeError("overlong sample")
            self.work += len(sample.token_ids)
            self.calls.append({"request_id": job.request_id, "seed": job.seed,
                "prefix_ids": list(job.prefix), "prefix_sha256": token_hash(job.prefix),
                "max_new_tokens": job.max_new_tokens, "token_ids": list(sample.token_ids),
                "lp": list(sample.token_logprobs), "sc": list(sc), "cn": list(cn),
                "finish_reason": sample.finish_reason})
        return samples

    def request(self, length, index, prefix=()):
        return GenerationRequest(self.root+tuple(prefix), length, self.sampling,
            SeedStream(self.seed).derive("candidate", index), f"{self.seed}:candidate:{index}")

    def statistics(self, completion):
        ids = tuple(completion)
        end = next((i for i, t in enumerate(ids) if t in {self.eos, self.close[0]}), len(ids))
        target = self.root+ids[:end]
        values = {k: [None]*end for k in ("sc", "cn", "lp")}
        for call in self.calls:
            prefix = tuple(call["prefix_ids"])
            offset = len(prefix)-len(self.root)
            if offset < 0 or target[:len(prefix)] != prefix:
                continue
            for j, token in enumerate(call["token_ids"]):
                at = offset+j
                if at >= end or token != ids[at]:
                    break
                for k in values:
                    if values[k][at] is None:
                        values[k][at] = call[k][j]
        if any(v is None or not math.isfinite(v) for row in values.values() for v in row):
            raise RuntimeError("missing trajectory statistic")
        return {k: list(self.carried[k])+values[k] for k in values}

    def scores(self, completions, kind):
        result = []
        for ids in completions:
            values = self.statistics(ids)
            result.append(trajectory_score(values["sc"], values["cn"], values["lp"], kind))
        return tuple(result)

    def reward(self, prompt, completions):
        if tuple(prompt) != self.root:
            raise RuntimeError("reward root mismatch")
        return self.scores(completions, self.reward_kind)


def search(ledger, method):
    cfg = ledger.cfg
    horizon = cfg.thinking_tokens-len(ledger.carried["sc"])
    if method == "bon":
        samples = ledger.sample_batch([ledger.request(horizon, i) for i in range(cfg.candidates)])
        candidates = [tuple(s.token_ids) for s in samples]
        scores = {k: ledger.scores(candidates, k) for k in ("sc", "consilience", "logprob")}
        selection = {"single": 0, **{"bon_"+k: max(range(len(candidates)), key=lambda i: (v[i], -i)) for k,v in scores.items()}}
        return candidates, {"method": method, "scores": scores, "selection": selection}
    selection, reward = {
        "is_sir_sc": ("sir", "sc"),
        "is_met_sc": ("metropolized", "sc"),
        "is_block_sc": ("block_metropolized", "sc"),
        "is_block_consilience": ("block_metropolized", "consilience"),
    }[method]
    ledger.reward_kind = reward
    config = ConditionalISConfig(candidate_count=cfg.is_candidates, rollout_count=cfg.rollouts,
        block_size=cfg.block_tokens, total_length=horizon,
        reward_temperature=.25 if reward == "sc" else 2., block_first=True,
        exploration_schedule="fixed", selection_ready=False, selection=selection)
    adapter = ConditionalISAdapter(backend=ledger, prompt=ledger.root, config=config,
        sampling=ledger.sampling, reward=ledger.reward)
    state, records = adapter.initial_state, []
    while not adapter.is_terminal(state):
        bound = pool_upper_bound(config, state.fixed, bool(state.token_ids))
        if bound > ledger.cap-ledger.work:
            if not state.token_ids:
                raise RuntimeError("cannot afford first complete IS pool")
            break
        before = ledger.work
        record, state = adapter.step(state, len(records), SeedStream(ledger.seed))
        if ledger.work-before > bound:
            raise RuntimeError("IS work reservation failed")
        records.append({**asdict(record), "reserved": bound, "work": ledger.work-before})
    return [tuple(state.token_ids)], {"method": method, "config": asdict(config), "steps": records,
        "fixed_tokens": state.fixed, "reward": state.reward,
        "stop": "terminal" if adapter.is_terminal(state) else "whole_pool_budget", "selection": {method: 0}}


def finish_answer(backend, root, suffix, close, cfg, seed):
    """Continue answers in large chunks until natural EOS; never silently accept a cap."""
    eos, close = backend.tokenizer.eos_token_id, tuple(close)
    suffix = tuple(suffix)
    if eos in suffix:
        return {"answer_text": "", "answer_ids": [], "calls": [], "empty_answer": True,
            "eos_inside_thinking": True, "truncated": False, "forced_thinking_end": False}
    natural = bool(close) and suffix[-len(close):] == close
    prefix = tuple(root)+suffix+( () if natural else close)
    answer, calls = [], []
    sampling = SamplingConfig(eos_token_id=eos)
    stopped = False
    while len(answer) < cfg.answer_tokens:
        count = min(cfg.answer_chunk, cfg.answer_tokens-len(answer))
        if len(prefix)+len(answer)+count > cfg.context_tokens:
            raise RuntimeError("answer context exhausted")
        request = GenerationRequest(prefix+tuple(answer), count, sampling,
            SeedStream(seed).derive("answer", len(calls)), f"{seed}:answer:{len(calls)}", intermediate_events=False)
        sample, = backend.sample_batch([request])
        calls.append({"seed": request.seed, "prefix_sha256": token_hash(request.prefix),
            "max_new_tokens": count, "token_ids": list(sample.token_ids), "finish_reason": sample.finish_reason})
        answer.extend(sample.token_ids)
        if eos in sample.token_ids or sample.finish_reason in {"eos", "eos_token", "stop"}:
            stopped = True
            break
        if not sample.token_ids:
            raise RuntimeError("answer generation made no progress")
    text = backend.decode(tuple(answer), skip_special_tokens=True).strip()
    return {"answer_text": text, "answer_ids": answer, "calls": calls,
        "empty_answer": not bool(text), "eos_inside_thinking": False,
        "truncated": not stopped, "forced_thinking_end": not natural}
