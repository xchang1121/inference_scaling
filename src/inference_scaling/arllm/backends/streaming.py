"""Thread-facing sessions for an asynchronous engine."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Sequence

from inference_scaling.arllm.backends.scheduling import GenerationEvent, SampleBuffer
from inference_scaling.arllm.types import GenerationRequest


class BatchGenerationSession:
    """The same dependency interface over synchronous batches; prefixes run first."""

    def __init__(self, backend) -> None:
        self.backend = backend
        self.pending: list[tuple[GenerationRequest, str]] = []
        self.events: deque[GenerationEvent] = deque()
        self.ids: set[str] = set()

    def submit(self, requests, *, phase="rollout", checkpoint=None) -> None:
        for request in requests:
            if request.request_id in self.ids:
                raise ValueError("duplicate logical request id")
            self.ids.add(request.request_id)
            self.pending.append((request, phase))

    def next_event(self) -> GenerationEvent:
        if not self.events:
            prefixes = any(phase == "prefix" for _, phase in self.pending)
            jobs = [job for job, phase in self.pending if (phase == "prefix") == prefixes]
            self.pending = [(job, phase) for job, phase in self.pending if (phase == "prefix") != prefixes]
            if not jobs:
                raise RuntimeError("batch session has no pending request")
            samples = self.backend.sample_batch(jobs)
            if len(samples) != len(jobs):
                raise RuntimeError("backend returned an invalid sample count")
            self.events.extend(GenerationEvent(sample, True) for sample in samples)
        return self.events.popleft()

    def seal(self) -> None:
        pass

    def stop(self, request_id: str) -> None:
        self.pending = [(job, phase) for job, phase in self.pending if job.request_id != request_id]
        self.events = deque(event for event in self.events if event.delta.request_id != request_id)

    def close(self) -> None:
        self.pending.clear()
        self.events.clear()


class EventMailbox:
    """One notification per unread request, with linear-size token buffers.

    The engine must keep draining while a caller computes a reward: that reward
    may itself submit scoring to the same admission pool. Blocking the producer
    on a full token-event queue would deadlock that dependency.
    """

    def __init__(self) -> None:
        self.ready: asyncio.Queue[str | BaseException] = asyncio.Queue()
        self.pending: dict[str, tuple[SampleBuffer, bool]] = {}

    async def put(self, event: GenerationEvent | BaseException) -> None:
        if isinstance(event, BaseException):
            self.ready.put_nowait(event)
            return
        key = event.delta.request_id
        if key not in self.pending:
            self.pending[key] = SampleBuffer(), False
            self.ready.put_nowait(key)
        buffer, finished = self.pending[key]
        if finished:
            raise RuntimeError("event arrived after request completion")
        buffer.append(event.delta)
        self.pending[key] = buffer, event.finished

    async def get(self) -> GenerationEvent | BaseException:
        key = await self.ready.get()
        if isinstance(key, BaseException):
            return key
        buffer, finished = self.pending.pop(key)
        return GenerationEvent(buffer.sample(), finished)


class AsyncGenerationSession:
    def __init__(self, backend) -> None:
        self.backend = backend
        self.group = backend._next_request_id()
        self.events = EventMailbox()
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.stopped: set[str] = set()
        self.closed = False

    def submit(self, requests: Sequence[GenerationRequest], *, phase: str = "rollout", checkpoint: int | None = None) -> None:
        if self.closed:
            raise RuntimeError("generation session is closed")

        async def submit() -> None:
            for request in requests:
                if request.request_id in self.tasks:
                    raise ValueError("duplicate logical request id in session")

                async def run(request=request) -> None:
                    try:
                        await self.backend._stream_request(request, self.group, phase, checkpoint, self.events.put)
                    except asyncio.CancelledError:
                        raise
                    except BaseException as error:
                        await self.events.put(error)

                self.tasks[request.request_id] = asyncio.create_task(run())

        self.backend._runner.run(submit())

    def next_event(self) -> GenerationEvent:
        if self.closed:
            raise RuntimeError("generation session is closed")
        async def next_event():
            while True:
                event = await self.events.get()
                if isinstance(event, BaseException):
                    raise event
                if event.delta.request_id not in self.stopped:
                    return event
        return self.backend._runner.run(next_event())

    def seal(self) -> None:
        async def seal() -> None:
            self.backend._admission.queue.sealed.add(self.group)
        self.backend._runner.run(seal())

    def stop(self, request_id: str) -> None:
        # Explicit semantic stop; whole-pool quotas cancel through close().
        self.stopped.add(request_id)

        async def stop() -> None:
            task = self.tasks.get(request_id)
            if task is not None:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        self.backend._runner.run(stop())

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        async def close() -> None:
            for task in self.tasks.values():
                task.cancel()
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
            self.backend._admission.queue.sealed.discard(self.group)
        self.backend._runner.run(close())


__all__ = ["AsyncGenerationSession", "BatchGenerationSession", "EventMailbox"]
