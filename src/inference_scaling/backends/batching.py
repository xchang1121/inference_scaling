"""Cross-request continuous batching for synchronous algorithm callers."""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Literal

from inference_scaling.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
)

_Kind = Literal["sample", "score"]
_STOP = object()


@dataclass(slots=True)
class _QueuedRequest:
    kind: _Kind
    request: GenerationRequest | ScoreRequest
    future: Future
    sequence_count: int
    token_cost: int


@dataclass(frozen=True, slots=True)
class BatchingSnapshot:
    sample_batches: int
    score_batches: int
    sample_requests: int
    score_sequences: int
    maximum_sample_batch: int
    maximum_score_batch: int

    @property
    def average_sample_batch(self) -> float:
        return self.sample_requests / self.sample_batches if self.sample_batches else 0.0

    @property
    def average_score_batch(self) -> float:
        return self.score_sequences / self.score_batches if self.score_batches else 0.0


class ContinuousBatchingBackend:
    """Merge requests from concurrent prompts without changing their random streams.

    Algorithms keep using the synchronous backend protocol.  Independent prompt
    workers can share this wrapper; a background worker combines requests that
    become ready within ``batch_wait_seconds`` and dispatches one backend call.
    """

    def __init__(
        self,
        backend: AutoregressiveBackend,
        *,
        max_batch_size: int = 32,
        max_batch_tokens: int = 4096,
        batch_wait_seconds: float = 0.002,
    ) -> None:
        if max_batch_size <= 0 or max_batch_tokens <= 0:
            raise ValueError("batch limits must be positive")
        if batch_wait_seconds < 0:
            raise ValueError("batch_wait_seconds must be non-negative")
        self._backend = backend
        self._max_batch_size = int(max_batch_size)
        self._max_batch_tokens = int(max_batch_tokens)
        self._batch_wait_seconds = float(batch_wait_seconds)
        self._queue: queue.Queue[_QueuedRequest | object] = queue.Queue()
        self._state_lock = threading.Lock()
        self._statistics_lock = threading.Lock()
        self._closed = False
        self._sample_batches = 0
        self._score_batches = 0
        self._sample_requests = 0
        self._score_sequences = 0
        self._maximum_sample_batch = 0
        self._maximum_score_batch = 0
        self._worker = threading.Thread(
            target=self._run,
            name=f"batching-backend:{backend.model_id}",
            daemon=True,
        )
        self._worker.start()

    @property
    def model_id(self) -> str:
        return self._backend.model_id

    @staticmethod
    def _generation_token_cost(request: GenerationRequest) -> int:
        return max(1, len(request.prefix) + request.max_new_tokens)

    @staticmethod
    def _score_token_cost(request: ScoreRequest) -> int:
        return max(
            1,
            sum(len(request.prefix) + len(continuation) for continuation in request.continuations),
        )

    def _submit(self, item: _QueuedRequest) -> Future:
        with self._state_lock:
            if self._closed:
                raise RuntimeError("continuous batching backend is closed")
            self._queue.put(item)
        return item.future

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        futures = [
            self._submit(
                _QueuedRequest(
                    kind="sample",
                    request=request,
                    future=Future(),
                    sequence_count=1,
                    token_cost=self._generation_token_cost(request),
                )
            )
            for request in requests
        ]
        return [future.result()[0] for future in futures]

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        futures = [
            self._submit(
                _QueuedRequest(
                    kind="score",
                    request=request,
                    future=Future(),
                    sequence_count=len(request.continuations),
                    token_cost=self._score_token_cost(request),
                )
            )
            for request in requests
        ]
        flattened: list[tuple[float, ...]] = []
        for future in futures:
            flattened.extend(future.result())
        return flattened

    def _fits(
        self,
        batch: list[_QueuedRequest],
        candidate: _QueuedRequest,
        sequence_count: int,
        token_cost: int,
    ) -> bool:
        if candidate.kind != batch[0].kind:
            return False
        return (
            sequence_count + candidate.sequence_count <= self._max_batch_size
            and token_cost + candidate.token_cost <= self._max_batch_tokens
        )

    def _record_batch(self, kind: _Kind, sequence_count: int) -> None:
        with self._statistics_lock:
            if kind == "sample":
                self._sample_batches += 1
                self._sample_requests += sequence_count
                self._maximum_sample_batch = max(
                    self._maximum_sample_batch, sequence_count
                )
            else:
                self._score_batches += 1
                self._score_sequences += sequence_count
                self._maximum_score_batch = max(self._maximum_score_batch, sequence_count)

    def _dispatch(self, batch: list[_QueuedRequest]) -> None:
        kind = batch[0].kind
        sequence_count = sum(item.sequence_count for item in batch)
        self._record_batch(kind, sequence_count)
        try:
            if kind == "sample":
                requests = [item.request for item in batch]
                outputs = self._backend.sample_batch(requests)  # type: ignore[arg-type]
                if len(outputs) != len(batch):
                    raise RuntimeError("underlying backend returned an invalid sample batch")
                for item, output in zip(batch, outputs, strict=True):
                    item.future.set_result([output])
                return

            requests = [item.request for item in batch]
            outputs = self._backend.score_batch(requests)  # type: ignore[arg-type]
            expected = sum(item.sequence_count for item in batch)
            if len(outputs) != expected:
                raise RuntimeError("underlying backend returned an invalid score batch")
            offset = 0
            for item in batch:
                end = offset + item.sequence_count
                item.future.set_result(outputs[offset:end])
                offset = end
        except BaseException as error:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(error)

    @staticmethod
    def _take_compatible_pending(
        pending: deque[_QueuedRequest],
        batch: list[_QueuedRequest],
        sequence_count: int,
        token_cost: int,
        fits,
    ) -> tuple[int, int]:
        retained: deque[_QueuedRequest] = deque()
        while pending:
            candidate = pending.popleft()
            if fits(batch, candidate, sequence_count, token_cost):
                batch.append(candidate)
                sequence_count += candidate.sequence_count
                token_cost += candidate.token_cost
            else:
                retained.append(candidate)
        pending.extend(retained)
        return sequence_count, token_cost

    def _run(self) -> None:
        pending: deque[_QueuedRequest] = deque()
        stopping = False
        while True:
            if pending:
                first = pending.popleft()
            else:
                queued = self._queue.get()
                if queued is _STOP:
                    break
                assert isinstance(queued, _QueuedRequest)
                first = queued
            batch = [first]
            sequence_count = first.sequence_count
            token_cost = first.token_cost
            sequence_count, token_cost = self._take_compatible_pending(
                pending,
                batch,
                sequence_count,
                token_cost,
                self._fits,
            )
            deadline = time.monotonic() + self._batch_wait_seconds
            while sequence_count < self._max_batch_size:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    queued = self._queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if queued is _STOP:
                    stopping = True
                    break
                assert isinstance(queued, _QueuedRequest)
                if self._fits(batch, queued, sequence_count, token_cost):
                    batch.append(queued)
                    sequence_count += queued.sequence_count
                    token_cost += queued.token_cost
                else:
                    pending.append(queued)
            self._dispatch(batch)
            if stopping and not pending:
                break

    def snapshot(self) -> BatchingSnapshot:
        with self._statistics_lock:
            return BatchingSnapshot(
                sample_batches=self._sample_batches,
                score_batches=self._score_batches,
                sample_requests=self._sample_requests,
                score_sequences=self._score_sequences,
                maximum_sample_batch=self._maximum_sample_batch,
                maximum_score_batch=self._maximum_score_batch,
            )

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(_STOP)
        self._worker.join()

    def __enter__(self) -> "ContinuousBatchingBackend":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()
