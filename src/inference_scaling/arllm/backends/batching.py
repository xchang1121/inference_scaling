"""Cross-request continuous batching for synchronous algorithm callers."""

from __future__ import annotations

import queue
import threading
import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Literal

from inference_scaling.arllm.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
)

_Kind = Literal["sample", "score"]
_STOP = object()


@dataclass(slots=True)
class _QueuedRequestGroup:
    kind: _Kind
    requests: tuple[GenerationRequest | ScoreRequest, ...]
    future: Future
    sequence_count: int
    token_cost: int
    batch_key: tuple[object, ...]


class ContinuousBatchingBackend:
    """Merge requests from concurrent prompts without changing their random streams.

    Algorithms keep using the synchronous backend protocol.  Independent prompt
    workers can share this wrapper; a background worker combines compatible caller
    groups that become ready within ``batch_wait_seconds`` and dispatches one backend
    call.  Keeping each caller group intact preserves repeated-prefix KV reuse.  An
    oversized generation group is split only at repeated-prefix run boundaries when
    possible.
    """

    def __init__(
        self,
        backend: AutoregressiveBackend,
        *,
        max_batch_size: int,
        max_batch_tokens: int,
        batch_wait_seconds: float,
    ) -> None:
        if max_batch_size <= 0 or max_batch_tokens <= 0:
            raise ValueError("batch limits must be positive")
        if batch_wait_seconds < 0:
            raise ValueError("batch_wait_seconds must be non-negative")
        self._backend = backend
        self._max_batch_size, self._max_batch_tokens = int(max_batch_size), int(max_batch_tokens)
        self._batch_wait_seconds = float(batch_wait_seconds)
        self._queue: queue.Queue[_QueuedRequestGroup | object] = queue.Queue()
        self._state_lock = threading.Lock()
        self._closed = False
        self._worker = threading.Thread(target=self._run, name=f"batching-backend:{backend.model_id}", daemon=True)
        self._worker.start()

    @property
    def model_id(self) -> str:
        return self._backend.model_id

    def __getattr__(self, name: str):
        # Tokenizer, decoding, direct generation and score statistics go straight
        # to the wrapped backend, which serializes its own model access.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._backend, name)

    @staticmethod
    def _generation_token_cost(request: GenerationRequest) -> int:
        return max(1, len(request.prefix) + request.max_new_tokens)

    @staticmethod
    def _score_token_cost(request: ScoreRequest) -> int:
        return max(1, sum(len(request.prefix) + len(continuation) for continuation in request.continuations))

    def _submit_groups(self, kind: _Kind, groups: Sequence[tuple[Any, ...]], count: Callable[[Any], int],
                       cost: Callable[[Any], int], key: Callable[[Any], tuple[object, ...]]) -> list[Any]:
        futures: list[Future] = []
        for group in groups:
            item = _QueuedRequestGroup(kind, tuple(group), Future(), count(group), sum(map(cost, group)), key(group))
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("continuous batching backend is closed")
                self._queue.put(item)
            futures.append(item.future)
        return [output for future in futures for output in future.result()]

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        if not requests:
            return []
        return self._submit_groups("sample", self._sample_request_groups(requests), len,
                                   self._generation_token_cost, self._sample_batch_key)

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        if not requests:
            return []
        return self._submit_groups("score", self._score_request_groups(requests),
                                   lambda group: sum(len(item.continuations) for item in group),
                                   self._score_token_cost, self._score_batch_key)

    @staticmethod
    def _sample_batch_key(requests: Sequence[GenerationRequest]) -> tuple[object, ...]:
        first = requests[0]
        sampling = first.sampling if all(request.sampling == first.sampling for request in requests) else None
        length = first.max_new_tokens if all(request.max_new_tokens == first.max_new_tokens for request in requests) else None
        repeat_counts = set(Counter(request.prefix for request in requests).values())
        return "sample", sampling, length, repeat_counts.pop() if len(repeat_counts) == 1 else None

    @staticmethod
    def _score_batch_key(requests: Sequence[ScoreRequest]) -> tuple[object, ...]:
        first_sampling = requests[0].sampling
        sampling = first_sampling if all(request.sampling == first_sampling for request in requests) else None
        maximum_length = max((len(request.prefix) + len(continuation)
                              for request in requests for continuation in request.continuations), default=0)
        return "score", sampling, ((maximum_length + 63) // 64) * 64

    def _within_limits(self, sequence_count: int, token_cost: int) -> bool:
        return sequence_count <= self._max_batch_size and token_cost <= self._max_batch_tokens

    def _sample_request_groups(self, requests: Sequence[GenerationRequest]) -> list[tuple[GenerationRequest, ...]]:
        requests = tuple(requests)
        cost = self._generation_token_cost
        if self._within_limits(len(requests), sum(map(cost, requests))):
            return [requests]

        runs: list[list[GenerationRequest]] = []
        run_keys: list[tuple[object, ...]] = []
        for request in requests:
            key = (request.sampling, request.prefix, request.max_new_tokens)
            if not runs or key != run_keys[-1]:
                runs.append([])
                run_keys.append(key)
            runs[-1].append(request)

        groups: list[tuple[GenerationRequest, ...]] = []
        current: list[GenerationRequest] = []
        current_cost = 0

        def flush() -> None:
            nonlocal current, current_cost
            if current:
                groups.append(tuple(current))
                current, current_cost = [], 0

        for run in runs:
            remaining = list(run)
            while remaining:
                remaining_cost = sum(map(cost, remaining))
                if self._within_limits(len(current) + len(remaining), current_cost + remaining_cost):
                    current.extend(remaining)
                    current_cost += remaining_cost
                    remaining.clear()
                    continue
                if current:
                    flush()
                    continue
                take = taken_cost = 0
                for request in remaining:
                    if take and not self._within_limits(take + 1, taken_cost + cost(request)):
                        break
                    take += 1
                    taken_cost += cost(request)
                    if not self._within_limits(take, taken_cost):
                        break
                take = max(1, take)
                current.extend(remaining[:take])
                current_cost = sum(map(cost, current))
                del remaining[:take]
                if remaining or not self._within_limits(len(current), current_cost):
                    flush()
        flush()
        return groups

    def _score_request_groups(self, requests: Sequence[ScoreRequest]) -> list[tuple[ScoreRequest, ...]]:
        groups: list[tuple[ScoreRequest, ...]] = []
        current: list[ScoreRequest] = []
        sequence_count = token_cost = 0
        for request in requests:
            request_sequences, request_cost = len(request.continuations), self._score_token_cost(request)
            if current and not self._within_limits(sequence_count + request_sequences, token_cost + request_cost):
                groups.append(tuple(current))
                current, sequence_count, token_cost = [], 0, 0
            current.append(request)
            sequence_count += request_sequences
            token_cost += request_cost
            if not self._within_limits(sequence_count, token_cost):
                groups.append(tuple(current))
                current, sequence_count, token_cost = [], 0, 0
        if current:
            groups.append(tuple(current))
        return groups

    def _fits(self, batch: list[_QueuedRequestGroup], candidate: _QueuedRequestGroup, sequence_count: int,
              token_cost: int) -> bool:
        if candidate.kind != batch[0].kind or candidate.batch_key != batch[0].batch_key:
            return False
        return (sequence_count + candidate.sequence_count <= self._max_batch_size
                and token_cost + candidate.token_cost <= self._max_batch_tokens)

    def _dispatch(self, batch: list[_QueuedRequestGroup]) -> None:
        kind = batch[0].kind
        try:
            requests = [request for item in batch for request in item.requests]
            call = self._backend.sample_batch if kind == "sample" else self._backend.score_batch
            outputs = call(requests)  # type: ignore[arg-type]
            if len(outputs) != sum(item.sequence_count for item in batch):
                raise RuntimeError(f"underlying backend returned an invalid {kind} batch")
            offset = 0
            for item in batch:
                item.future.set_result(outputs[offset : offset + item.sequence_count])
                offset += item.sequence_count
        except BaseException as error:
            for item in batch:
                if not item.future.done():
                    item.future.set_exception(error)

    def _take_compatible_pending(self, pending: deque[_QueuedRequestGroup], batch: list[_QueuedRequestGroup],
                                 sequence_count: int, token_cost: int) -> tuple[int, int]:
        retained: deque[_QueuedRequestGroup] = deque()
        while pending:
            candidate = pending.popleft()
            if self._fits(batch, candidate, sequence_count, token_cost):
                batch.append(candidate)
                sequence_count += candidate.sequence_count
                token_cost += candidate.token_cost
            else:
                retained.append(candidate)
        pending.extend(retained)
        return sequence_count, token_cost

    def _run(self) -> None:
        pending: deque[_QueuedRequestGroup] = deque()
        stopping = False
        while True:
            if pending:
                first = pending.popleft()
            else:
                queued = self._queue.get()
                if queued is _STOP:
                    break
                assert isinstance(queued, _QueuedRequestGroup)
                first = queued
            batch = [first]
            sequence_count, token_cost = self._take_compatible_pending(
                pending, batch, first.sequence_count, first.token_cost)
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
                assert isinstance(queued, _QueuedRequestGroup)
                if self._fits(batch, queued, sequence_count, token_cost):
                    batch.append(queued)
                    sequence_count += queued.sequence_count
                    token_cost += queued.token_cost
                else:
                    pending.append(queued)
            self._dispatch(batch)
            if stopping and not pending:
                break

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
