"""Admission and streaming contracts, independent of the model and sampling algorithm.

Scheduling changes when a fixed request runs, never its seed, policy or length.
Only observed history and request metadata are used to estimate service time.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from inference_scaling.arllm.types import GenerationRequest, SequenceSample


@dataclass(frozen=True)
class SchedulerConfig:
    enabled: bool = True
    max_in_flight: int = 32
    max_prefill_tokens: int = 16384
    kv_high_watermark: float = 0.9
    aging_seconds: float = 2.0
    tail_prefill_tokens: int = 4096
    policy: str = "fair_lpt"

    def __post_init__(self) -> None:
        if min(self.max_in_flight, self.max_prefill_tokens, self.tail_prefill_tokens) <= 0:
            raise ValueError("scheduler capacities must be positive")
        if not 0 < self.kv_high_watermark <= 1 or not 0 < self.aging_seconds < float("inf"):
            raise ValueError("invalid scheduler pressure or aging limit")
        if self.policy not in {"fair_lpt", "fifo"}:
            raise ValueError("unknown scheduler policy")


@dataclass(frozen=True)
class GenerationEvent:
    """New tokens only; ``delta.prefix`` is the original request prefix.

    A terminal event includes its last tokens and actual finish reason. Empty
    nonterminal events are permitted but do not signal a completed request.
    """

    delta: SequenceSample
    finished: bool


class GenerationSession(Protocol):
    def submit(self, requests: Sequence[GenerationRequest], *, phase: str = "rollout", checkpoint: int | None = None) -> None: ...
    def next_event(self) -> GenerationEvent: ...
    def seal(self) -> None: ...
    def stop(self, request_id: str) -> None: ...
    def close(self) -> None: ...


class SampleBuffer:
    """Accumulate deltas without repeatedly copying the growing sequence."""

    def __init__(self) -> None:
        self.last: SequenceSample | None = None
        self.tokens: list[int] = []
        self.logs: list[float] = []
        self.references: list[float] | None = []
        self.bounds: list[tuple[float, float]] | None = []
        self.statistics: list[float] | None = []

    def append(self, sample: SequenceSample) -> None:
        if self.last is not None and (sample.prefix, sample.request_id, sample.model_id, sample.policy_id) != (
                self.last.prefix, self.last.request_id, self.last.model_id, self.last.policy_id):
            raise RuntimeError("stream changed request identity")
        self.last = sample
        self.tokens.extend(sample.token_ids)
        self.logs.extend(sample.token_logprobs)
        for name, source in (("references", sample.reference_token_logprobs),
                             ("bounds", sample.token_cdf_bounds), ("statistics", sample.token_statistics)):
            target = getattr(self, name)
            if target is not None and source is not None:
                target.extend(source)
            else:
                setattr(self, name, None)

    def sample(self) -> SequenceSample:
        if self.last is None:
            raise RuntimeError("empty generation stream")
        from dataclasses import replace
        return replace(self.last, token_ids=tuple(self.tokens), token_logprobs=tuple(self.logs),
                       reference_token_logprobs=None if self.references is None else tuple(self.references),
                       reference_policy_id=None if self.references is None else self.last.reference_policy_id,
                       token_cdf_bounds=None if self.bounds is None else tuple(self.bounds),
                       token_statistics=None if self.statistics is None else tuple(self.statistics))


@dataclass(frozen=True)
class Admission:
    key: str
    group: str
    prefix_tokens: int
    remaining_tokens: int
    phase: str = "rollout"


class AdmissionQueue:
    """Pure queue shared by production admission and the CPU event simulation."""

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.pending: dict[str, tuple[Admission, float, int, float]] = {}
        self.active: dict[str, Admission] = {}
        self.prefilling: set[str] = set()
        self.sealed: set[str] = set()
        self.history: dict[int, float] = {}
        self.counter = itertools.count()
        self.maximum_active = 0

    def add(self, item: Admission, now: float) -> None:
        if item.key in self.pending or item.key in self.active:
            raise ValueError("duplicate admission key")
        bucket = max(1, item.prefix_tokens).bit_length()
        estimate = min(item.remaining_tokens, self.history.get(bucket, float(item.remaining_tokens)))
        self.pending[item.key] = item, now, next(self.counter), estimate

    def admit(self, now: float, *, kv_usage: float = 0.0) -> Admission | None:
        config = self.config
        if not self.pending or len(self.active) >= config.max_in_flight:
            return None
        if self.active and kv_usage >= config.kv_high_watermark:
            return None
        groups = Counter(item.group for item in self.active.values())
        queued = {item.group for item, _, _, _ in self.pending.values()}
        draining = {group for group, count in groups.items() if count <= 2 and group in self.sealed and group not in queued}
        prefill = sum(self.active[key].prefix_tokens for key in self.prefilling)

        def priority(entry):
            item, entered, order, estimate = entry
            if config.policy == "fifo":
                return (0, order, 0, 0, 0)
            aged = now - entered >= config.aging_seconds
            return (0, entered, 0, 0, order) if aged else (
                1, groups[item.group], {"prefix": 0, "spare": 2}.get(item.phase, 1), -estimate, order)

        for item, entered, _, _ in sorted(self.pending.values(), key=priority):
            if self.active:
                # One oversized prefill is allowed on its own; it cannot starve forever.
                if self.prefilling and prefill + item.prefix_tokens > config.max_prefill_tokens:
                    if now - entered >= config.aging_seconds:
                        return None
                    continue
                if (draining and item.group not in draining and item.prefix_tokens > config.tail_prefill_tokens
                        and now - entered < config.aging_seconds):
                    continue
            del self.pending[item.key]
            self.active[item.key] = item
            self.prefilling.add(item.key)
            self.maximum_active = max(self.maximum_active, len(self.active))
            return item
        return None

    def progress(self, key: str) -> None:
        self.prefilling.discard(key)

    def finish(self, key: str, generated: int | None = None) -> None:
        self.pending.pop(key, None)
        item = self.active.pop(key, None)
        self.prefilling.discard(key)
        if item is not None and generated is not None:
            bucket = max(1, item.prefix_tokens).bit_length()
            self.history[bucket] = .8 * self.history.get(bucket, float(generated)) + .2 * generated


class AsyncAdmissionPool:
    """All methods run on the engine loop; no reward callback runs on that loop."""

    def __init__(self, config: SchedulerConfig, pressure: Callable[[], float]) -> None:
        self.queue = AdmissionQueue(config)
        self.pressure = pressure
        self.waiters: dict[str, asyncio.Future[None]] = {}
        self.timer: asyncio.TimerHandle | None = None

    def pump(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None
        pressure = self.pressure() if self.queue.pending and len(self.queue.active) < self.queue.config.max_in_flight else 0.0
        while (item := self.queue.admit(time.monotonic(), kv_usage=pressure)) is not None:
            waiter = self.waiters[item.key]
            if not waiter.done():
                waiter.set_result(None)
        if self.queue.pending and len(self.queue.active) < self.queue.config.max_in_flight:
            self.timer = asyncio.get_running_loop().call_later(.05, self.pump)

    async def acquire(self, item: Admission) -> None:
        waiter = asyncio.get_running_loop().create_future()
        self.waiters[item.key] = waiter
        self.queue.add(item, time.monotonic())
        self.pump()
        try:
            await waiter
        except BaseException:
            self.finish(item.key)
            raise

    def progress(self, key: str) -> None:
        self.queue.progress(key)
        self.pump()

    def finish(self, key: str, generated: int | None = None) -> None:
        self.queue.finish(key, generated)
        self.waiters.pop(key, None)
        self.pump()


__all__ = ["Admission", "AdmissionQueue", "AsyncAdmissionPool", "GenerationEvent", "GenerationSession",
           "SampleBuffer", "SchedulerConfig"]
