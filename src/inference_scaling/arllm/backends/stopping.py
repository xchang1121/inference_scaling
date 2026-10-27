"""Generation that ends at the close of a thinking segment.

The thinking scope samples only the thinking segment: generation ends at the
segment's closing boundary or at EOS, whichever comes first, and the sample is
returned as it stopped. The boundary tokens keep their model probability and
nothing follows them, so under ``score_batch`` a continuation that runs past a
boundary has probability zero. The end markers go to the wrapped backend as
stop sequences; a marker that closes no boundary (an empty block) resumes the
generation where it stopped in the same uniform stream, and a backend without
stop sequences is cut at the boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import AutoregressiveBackend, GenerationRequest, ScoreRequest, SequenceSample
from inference_scaling.shared.model.output import OutputParser
from inference_scaling.shared.types import TokenSequence


class StoppedSequenceBackend:
    def __init__(
        self,
        backend: AutoregressiveBackend,
        *,
        thinking_parser: OutputParser,
        thinking_prompt: TokenSequence,
        eos_token_id: int,
    ) -> None:
        self.backend = backend
        self.thinking_parser = thinking_parser
        # The prompt is never searched for boundaries.
        self.thinking_prompt = tuple(thinking_prompt)
        self.eos_token_id = eos_token_id
        formats = getattr(thinking_parser, "formats", (thinking_parser,))
        self.stop_sequences = tuple(dict.fromkeys(
            tuple(marker) for format_ in formats if (marker := getattr(format_, "end_token_ids", None))
        ))
        self.model_id = f"{backend.model_id}|thinking={thinking_parser.describe()};eos={eos_token_id}"

    @property
    def tokenizer(self):
        return getattr(self.backend, "tokenizer")

    def stratified_initialization(self, entries):
        context = getattr(self.backend, "stratified_initialization", None)
        if context is None:
            raise ValueError("the backend does not support stratified initialization")
        return context(entries)

    @property
    def supports_streaming_generation(self) -> bool:
        return bool(getattr(self.backend, "supports_streaming_generation", False))

    def open_generation_session(self):
        return StoppedGenerationSession(self)

    def token_statistics(self, requests, statistic):
        # A reward chooses its span, so these are the unstopped model's statistics.
        return getattr(self.backend, "token_statistics")(requests, statistic)

    def _stop_end(self, generated: TokenSequence) -> int | None:
        """End of the first boundary in the generated tokens: the thinking close or EOS."""

        ends = []
        segments = self.thinking_parser.split(self.thinking_prompt, generated, eos_token_id=self.eos_token_id)
        if segments.has_complete_thinking and segments.boundary_end is not None:
            ends.append(segments.boundary_end)
        if self.eos_token_id in generated:
            ends.append(generated.index(self.eos_token_id) + 1)
        return min(ends) if ends else None

    def _generated(self, prefix: TokenSequence) -> TokenSequence:
        if prefix[: len(self.thinking_prompt)] != self.thinking_prompt:
            raise ValueError("the prefix does not start with the scoped prompt")
        return prefix[len(self.thinking_prompt):]

    def _inner_policy(self, sampling: SamplingConfig | None) -> SamplingConfig:
        source = sampling or SamplingConfig()
        if source.eos_token_id not in (None, self.eos_token_id):
            raise ValueError("sampling uses a different EOS token")
        return replace(source, eos_token_id=self.eos_token_id)

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        tokens: list[list[int]] = [[] for _ in requests]
        logs: list[list[float]] = [[] for _ in requests]
        references: list[list[float] | None] = [[] for _ in requests]
        bounds: list[list[Any] | None] = [[] for _ in requests]
        statistics: list[list[Any] | None] = [[] if request.statistic else None for request in requests]
        # Sum actual inner requests, including repeated prefills after a native stop.
        slots: list[int | None] = [0] * len(requests)
        # Why each request ended: None while it continues, else "stop" or "rejected".
        done: list[str | None] = [None] * len(requests)
        weights = [None if request.log_weight_stop is None else request.log_weight_stop.start for request in requests]
        for request in requests:
            if self._stop_end(self._generated(request.prefix)) is not None:
                raise ValueError("the prefix already ended at a stop boundary")
        while True:
            pending, indices = list[GenerationRequest](), list[int]()
            for index, request in enumerate(requests):
                offset = len(tokens[index])
                if done[index] or offset >= request.max_new_tokens:
                    continue
                draft, stop = request.draft, request.log_weight_stop
                pending.append(replace(
                    request, prefix=request.prefix + tuple(tokens[index]), max_new_tokens=request.max_new_tokens - offset,
                    sampling=self._inner_policy(request.sampling), request_id=f"{request.request_id}:stop:{offset}",
                    stop_sequences=self.stop_sequences, uniform_offset=request.uniform_offset + offset,
                    # The draft goes on only while the output has followed it.
                    draft=draft.after(offset) if draft and tuple(tokens[index]) == draft.token_ids[:offset] else None,
                    log_weight_stop=None if stop is None else replace(stop, start=weights[index]),
                ))
                indices.append(index)
            if not pending:
                break
            sampled = self.backend.sample_batch(pending)
            if len(sampled) != len(pending):
                raise RuntimeError("wrapped backend returned an invalid sample count")
            for index, inner_request, sample in zip(indices, pending, sampled, strict=True):
                if (sample.prefix != inner_request.prefix or sample.policy_id != inner_request.sampling.policy_id
                        or sample.model_id != self.backend.model_id
                        or not 0 < len(sample.token_ids) <= inner_request.max_new_tokens):
                    raise RuntimeError("wrapped backend violated the generation contract")
                generated = self._generated(inner_request.prefix)
                end = self._stop_end(generated + sample.token_ids)
                keep = len(sample.token_ids) if end is None else end - len(generated)
                tokens[index].extend(sample.token_ids[:keep])
                logs[index].extend(sample.token_logprobs[:keep])
                slots[index] = (None if slots[index] is None or sample.forward_token_slots is None
                                else slots[index] + sample.forward_token_slots)
                reference = references[index]
                if reference is not None and sample.reference_policy_id == inner_request.reference_policy.policy_id:
                    reference.extend((sample.reference_token_logprobs or ())[:keep])
                else:
                    references[index] = None
                for kept, values in ((bounds, sample.token_cdf_bounds), (statistics, sample.token_statistics)):
                    current = kept[index]
                    if current is not None and values is not None:
                        current.extend(values[:keep])
                    else:
                        kept[index] = None
                stop = inner_request.log_weight_stop
                if stop is not None:
                    weights[index] = replace(stop, start=weights[index]).weight(
                        (sample.reference_token_logprobs or ())[:keep], sample.token_logprobs[:keep])
                if end is not None:
                    done[index] = "stop"
                elif sample.finish_reason == "rejected":
                    done[index] = "rejected"
                elif sample.finish_reason != "stop" and len(sample.token_ids) != inner_request.max_new_tokens:
                    raise RuntimeError("generation ended without EOS or a thinking boundary")
        return [
            SequenceSample(
                prefix=request.prefix, token_ids=tuple(tokens[index]), token_logprobs=tuple(logs[index]),
                model_id=self.model_id, policy_id=request.sampling.policy_id, request_id=request.request_id,
                finish_reason=done[index] or "length",
                reference_token_logprobs=None if references[index] is None else tuple(references[index] or ()),
                reference_policy_id=None if references[index] is None else request.reference_policy.policy_id,
                token_cdf_bounds=None if bounds[index] is None else tuple(bounds[index] or ()),
                token_statistics=None if statistics[index] is None else tuple(statistics[index] or ()),
                forward_token_slots=slots[index],
            )
            for index, request in enumerate(requests)
        ]

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        pending: list[ScoreRequest] = []
        # (result index, tokens past the boundary)
        metadata: list[tuple[int, int]] = []
        results: list[tuple[float, ...] | None] = []
        for request in requests:
            generated = self._generated(request.prefix)
            terminal = self._stop_end(generated) is not None
            for continuation in request.continuations:
                index = len(results)
                results.append(None)
                if terminal or not continuation:
                    results[index] = (float("-inf"),) * len(continuation)
                    continue
                end = self._stop_end(generated + continuation)
                keep = len(continuation) if end is None else end - len(generated)
                pending.append(ScoreRequest(request.prefix, (continuation[:keep],), self._inner_policy(request.sampling)))
                metadata.append((index, len(continuation) - keep))
        scored = self.backend.score_batch(pending) if pending else []
        if len(scored) != len(pending):
            raise RuntimeError("wrapped backend returned an invalid score count")
        for (index, past), request, scores in zip(metadata, pending, scored, strict=True):
            if len(scores) != len(request.continuations[0]):
                raise RuntimeError("wrapped backend returned an invalid score shape")
            results[index] = tuple(scores) + (float("-inf"),) * past
        if any(value is None for value in results):
            raise RuntimeError("incomplete stopped-sequence score batch")
        return [value for value in results if value is not None]


class StoppedGenerationSession:
    """The same scope law as sample_batch, with early boundary detection."""

    def __init__(self, backend: StoppedSequenceBackend) -> None:
        self.backend = backend
        self.inner = getattr(backend.backend, "open_generation_session")()
        self.requests: dict[str, GenerationRequest] = {}
        self.tokens: dict[str, list[int]] = {}
        self.routes: dict[str, tuple[str, GenerationRequest]] = {}
        self.current: dict[str, str] = {}
        self.phases: dict[str, str] = {}
        self.checkpoints: dict[str, int | None] = {}
        self.generated: dict[str, TokenSequence] = {}
        self.slots: dict[str, int | None] = {}

    def submit(self, requests: Sequence[GenerationRequest], *, phase: str = "rollout", checkpoint: int | None = None) -> None:
        pending = []
        for request in requests:
            if request.request_id in self.requests:
                raise ValueError("duplicate scoped request id")
            if request.draft is not None or request.log_weight_stop is not None:
                raise ValueError("streamed scope does not support draft replay or log-weight stopping")
            generated = self.backend._generated(request.prefix)
            if self.backend._stop_end(generated) is not None:
                raise ValueError("the prefix already ended at a stop boundary")
            self.requests[request.request_id] = request
            self.tokens[request.request_id] = []
            self.slots[request.request_id] = 0
            self.phases[request.request_id] = phase
            self.checkpoints[request.request_id] = checkpoint
            self.generated[request.request_id] = generated
            pending.append(self._continuation(request.request_id))
        self.inner.submit(pending, phase=phase, checkpoint=checkpoint)

    def _continuation(self, key: str) -> GenerationRequest:
        request, tokens = self.requests[key], self.tokens[key]
        offset = len(tokens)
        inner = replace(request, prefix=request.prefix + tuple(tokens), max_new_tokens=request.max_new_tokens - offset,
                        sampling=self.backend._inner_policy(request.sampling), request_id=f"{key}:stop:{offset}",
                        stop_sequences=self.backend.stop_sequences, uniform_offset=request.uniform_offset + offset,
                        # Single-token markers stop natively; longer markers need this wrapper's deltas.
                        intermediate_events=request.intermediate_events or not getattr(
                            self.backend.backend, "supports_native_token_stops", False) or any(
                            len(marker) > 1 for marker in self.backend.stop_sequences))
        self.routes[inner.request_id] = key, inner
        self.current[key] = inner.request_id
        return inner

    def next_event(self):
        from inference_scaling.arllm.backends.scheduling import GenerationEvent
        event = self.inner.next_event()
        sample = event.delta
        key, inner_request = self.routes[sample.request_id]
        request, tokens = self.requests[key], self.tokens[key]
        if (sample.prefix != inner_request.prefix or sample.model_id != self.backend.backend.model_id
                or sample.policy_id != inner_request.sampling.policy_id):
            raise RuntimeError("wrapped stream violated request identity")
        generated = self.generated[key]
        # Parse only when a new EOS/close marker appears, avoiding a full-prefix
        # parse on every decode step (including markers split between deltas).
        width = max((len(marker) for marker in self.backend.stop_sequences), default=1)
        tail = tuple(tokens[-width:]) if len(tokens) >= width else (generated + tuple(tokens))[-width:]
        recent = tail + sample.token_ids
        possible = self.backend.eos_token_id in sample.token_ids or any(
            recent[index:index + len(marker)] == marker
            for marker in self.backend.stop_sequences for index in range(max(0, len(tail) - len(marker) + 1), len(recent)))
        end = self.backend._stop_end(generated + tuple(tokens) + sample.token_ids) if possible else None
        keep = len(sample.token_ids) if end is None else end - len(generated) - len(tokens)
        if not 0 <= keep <= len(sample.token_ids):
            raise RuntimeError("scope boundary moved behind streamed tokens")
        tokens.extend(sample.token_ids[:keep])
        if len(tokens) > request.max_new_tokens:
            raise RuntimeError("scoped stream exceeded its generation limit")
        finished = end is not None or len(tokens) == request.max_new_tokens
        reason = "stop" if end is not None else "length"
        if event.finished:
            self.slots[key] = (None if self.slots[key] is None or sample.forward_token_slots is None
                               else self.slots[key] + sample.forward_token_slots)
        if end is not None and not event.finished:
            self.slots[key] = None
            self.inner.stop(sample.request_id)
        if event.finished and not finished:
            if sample.finish_reason != "stop" or not sample.token_ids:
                raise RuntimeError("generation ended without EOS or a thinking boundary")
            # A native marker may be an empty/invalid thinking close. Match the
            # batch wrapper's continuation seed and uniform offset exactly.
            checkpoint = self.checkpoints[key]
            checkpoint = None if checkpoint is None or checkpoint <= len(tokens) else checkpoint - len(tokens)
            self.inner.submit([self._continuation(key)], phase=self.phases[key], checkpoint=checkpoint)
        return GenerationEvent(replace(
            sample, prefix=request.prefix, request_id=key, policy_id=request.sampling.policy_id,
            model_id=self.backend.model_id, token_ids=sample.token_ids[:keep],
            token_logprobs=sample.token_logprobs[:keep], finish_reason=reason,
            reference_token_logprobs=None if sample.reference_policy_id != inner_request.reference_policy.policy_id else (
                sample.reference_token_logprobs or ())[:keep],
            reference_policy_id=None if sample.reference_policy_id != inner_request.reference_policy.policy_id else request.reference_policy.policy_id,
            token_cdf_bounds=None if sample.token_cdf_bounds is None else sample.token_cdf_bounds[:keep],
            token_statistics=None if sample.token_statistics is None else sample.token_statistics[:keep],
            forward_token_slots=self.slots[key] if finished else None), finished)

    def seal(self) -> None:
        self.inner.seal()

    def stop(self, request_id: str) -> None:
        current = self.current.pop(request_id, None)
        if current is not None:
            self.inner.stop(current)
        for records in (self.requests, self.tokens, self.phases, self.checkpoints, self.generated, self.slots):
            records.pop(request_id, None)
        for key, (logical, _) in tuple(self.routes.items()):
            if logical == request_id:
                del self.routes[key]

    def close(self) -> None:
        self.inner.close()


__all__ = ["StoppedSequenceBackend", "StoppedGenerationSession"]
