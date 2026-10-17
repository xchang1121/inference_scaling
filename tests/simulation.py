"""CPU test engine for checking scheduling without a GPU.

The wrapped backend supplies exact samples. The simulator privately materializes
them at admission, then exposes only emitted deltas. Admission never reads the
future sample, reward or realized service time. Time is in virtual seconds.
"""

from __future__ import annotations

import heapq
import itertools
from collections import deque
from dataclasses import replace
from typing import Any

from inference_scaling.arllm.backends.scheduling import Admission, AdmissionQueue, GenerationEvent, SampleBuffer, SchedulerConfig
from inference_scaling.arllm.types import GenerationRequest, SequenceSample


class SimulatedBackend:
    def __init__(self, backend, config: SchedulerConfig, *, streaming: bool = True,
                 token_seconds=None, prefill_seconds: float = .0001) -> None:
        self.backend = backend
        self.queue = AdmissionQueue(config)
        self.supports_streaming_generation = streaming
        self.token_seconds = token_seconds or (lambda request, index, sample: .001)
        self.prefill_seconds = prefill_seconds
        self.now = 0.0
        self.counter = itertools.count()
        self.jobs: dict[str, tuple[SimulatedSession, GenerationRequest, SequenceSample | None, int]] = {}
        self.heap: list[tuple[float, int, str]] = []
        self.trace: list[dict[str, Any]] = []

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def open_generation_session(self):
        return SimulatedSession(self, f"session:{next(self.counter)}")

    def sample_batch(self, requests):
        if not requests:
            return []
        session = self.open_generation_session()
        buffers = {request.request_id: SampleBuffer() for request in requests}
        done: set[str] = set()
        try:
            session.submit(requests)
            session.seal()
            while len(done) < len(requests):
                event = session.next_event()
                buffers[event.delta.request_id].append(event.delta)
                if event.finished:
                    done.add(event.delta.request_id)
            return [buffers[request.request_id].sample() for request in requests]
        finally:
            session.close()

    def _admit(self):
        while (item := self.queue.admit(self.now)) is not None:
            session, request, _, _ = self.jobs[item.key]
            sample = self.backend.sample_batch([request])[0]
            if not sample.token_ids:
                raise RuntimeError("simulation requires a nonempty generated sample")
            self.jobs[item.key] = session, request, sample, 0
            self.trace.append({"event": "admit", "time": self.now, "group": item.group, "id": request.request_id})
            delay = len(request.prefix) * self.prefill_seconds + self.token_seconds(request, 0, sample)
            heapq.heappush(self.heap, (self.now + delay, next(self.counter), item.key))

    def _advance(self):
        self._admit()
        while self.heap:
            when, _, key = heapq.heappop(self.heap)
            if key in self.jobs:
                break
        else:
            if self.queue.pending:
                # A protected tail can defer a prefill until its aging deadline.
                self.now += self.queue.config.aging_seconds
                self._admit()
                if self.heap:
                    return self._advance()
            raise RuntimeError("simulation has no event to advance")
        self.now = when
        session, request, sample, offset = self.jobs[key]
        assert sample is not None
        end = offset + 1
        finished = end == len(sample.token_ids)
        delta = replace(sample, token_ids=sample.token_ids[offset:end], token_logprobs=sample.token_logprobs[offset:end],
                        reference_token_logprobs=None if sample.reference_token_logprobs is None else sample.reference_token_logprobs[offset:end],
                        token_cdf_bounds=None if sample.token_cdf_bounds is None else sample.token_cdf_bounds[offset:end],
                        token_statistics=None if sample.token_statistics is None else sample.token_statistics[offset:end],
                        finish_reason=sample.finish_reason if finished else "length")
        self.queue.progress(key)
        session.events.append(GenerationEvent(delta, finished))
        if finished:
            self.queue.finish(key, end)
            del self.jobs[key]
            self.trace.append({"event": "finish", "time": self.now, "group": session.group, "id": request.request_id})
        else:
            self.jobs[key] = session, request, sample, end
            delay = self.token_seconds(request, end, sample)
            heapq.heappush(self.heap, (self.now + delay, next(self.counter), key))
        # Let the consumer release newly ready children before refilling slots.


class SimulatedSession:
    def __init__(self, backend, group):
        self.backend, self.group = backend, group
        self.events: deque[GenerationEvent] = deque()
        self.keys: dict[str, str] = {}
        self.stopped: set[str] = set()
        self.closed = False

    def submit(self, requests, *, phase="rollout", checkpoint=None):
        if self.closed:
            raise RuntimeError("simulation session is closed")
        for request in requests:
            if request.request_id in self.keys:
                raise ValueError("duplicate logical request id")
            key = f"{self.group}:{request.request_id}"
            self.keys[request.request_id] = key
            self.backend.jobs[key] = self, request, None, 0
            self.backend.queue.add(Admission(key, self.group, len(request.prefix), request.max_new_tokens, phase),
                                   self.backend.now)
            self.backend.trace.append({"event": "submit", "time": self.backend.now, "group": self.group,
                                       "id": request.request_id, "phase": phase})

    def next_event(self):
        while True:
            while not self.events:
                self.backend._advance()
            event = self.events.popleft()
            if event.delta.request_id not in self.stopped:
                return event

    def seal(self):
        self.backend.queue.sealed.add(self.group)

    def stop(self, request_id):
        self.stopped.add(request_id)
        key = self.keys[request_id]
        self.backend.queue.finish(key)
        self.backend.jobs.pop(key, None)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for key in self.keys.values():
            self.backend.queue.finish(key)
            self.backend.jobs.pop(key, None)
        self.backend.queue.sealed.discard(self.group)


__all__ = ["SimulatedBackend"]
