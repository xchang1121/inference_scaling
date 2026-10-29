"""A thinking limit shared by generation, scoring and streamed requests.

At the budget boundary an unfinished thought receives its closing marker with
probability one. Natural closes and EOS are unchanged. The wrapper counts from
the original prompt, including an IS/MH prefix, rather than restarting the
budget on every rollout. It does not extend the request's total token allowance.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from inference_scaling.arllm.backends.scheduling import GenerationEvent, SampleBuffer
from inference_scaling.arllm.backends.streaming import BatchGenerationSession
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest, SequenceSample


def budgeted_backend(backend, prompt, parser, budget):
    """Install a budget only for a recognized, prefilled thinking phase."""
    limit = budget.get("effective_thinking_budget")
    if limit is None or not budget.get("thinking_enabled", True):
        budget["thinking_budget_enforcement"] = "disabled"
        return backend
    if limit >= budget["effective_max_new_tokens"]:
        budget["thinking_budget_enforcement"] = "total_limit"
        return backend
    formats = getattr(parser, "formats", ())
    # Parsing a hypothetical close also checks that the prompt has an open,
    # leading thinking phase, rather than a delimiter quoted in earlier text.
    opened = [format_ for format_ in formats if format_.prompt_is_thinking(prompt)]
    if not opened:
        budget["thinking_budget_enforcement"] = "unrecognized_open_thinking"
        raise ValueError("thinking_budget requires a recognized prefilled thinking block; "
                         "configure its markers/starts_in_thinking or set ar.thinking=null")
    marker = opened[-1].end_token_ids
    budget["thinking_budget_enforcement"] = "close_then_content"
    return ThinkingBudgetBackend(backend, tuple(prompt), parser, limit, marker)


class ThinkingBudgetBackend:
    def __init__(self, backend, prompt, parser, limit, marker):
        self.backend, self.prompt, self.parser = backend, prompt, parser
        self.limit, self.marker = limit, tuple(marker)
        self.model_id = f"{backend.model_id}|thinking_budget={limit};close={self.marker}"

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def _plan(self, prefix, maximum):
        if prefix[:len(self.prompt)] != self.prompt:
            raise ValueError("thinking-budget prefix does not start with the original prompt")
        generated = prefix[len(self.prompt):]
        if len(generated) < self.limit:
            return min(maximum, self.limit - len(generated)), ()
        head = generated[:self.limit]
        eos = self.tokenizer.eos_token_id
        # Empty/malformed blocks retain the established full-sequence fallback;
        # in particular, content after an empty close is not thinking.
        if eos in head or self.parser.split(self.prompt, head, eos_token_id=eos).status != "incomplete":
            return maximum, ()
        offset = len(generated) - self.limit
        if generated[self.limit:self.limit + len(self.marker)] != self.marker[:min(offset, len(self.marker))]:
            raise ValueError("prefix violates the thinking-budget closing marker")
        forced = self.marker[offset:offset + maximum]
        return (0, forced) if forced else (maximum, ())

    def sample_batch(self, requests):
        if not requests:
            return []
        session = BudgetSession(self, BatchGenerationSession(self.backend))
        buffers = {request.request_id: SampleBuffer() for request in requests}
        completed: set[str] = set()
        try:
            session.submit(requests)
            session.seal()
            while len(completed) < len(requests):
                event = session.next_event()
                buffers[event.delta.request_id].append(event.delta)
                if event.finished:
                    completed.add(event.delta.request_id)
            return [buffers[request.request_id].sample() for request in requests]
        finally:
            session.close()

    def open_generation_session(self):
        return BudgetSession(self, self.backend.open_generation_session())

    def score_batch(self, requests):
        rows, pending, routes = [], [], []
        for request in requests:
            for continuation in request.continuations:
                row = [float("-inf")] * len(continuation)
                rows.append(row)
                offset = 0
                while offset < len(continuation):
                    try:
                        count, forced = self._plan(request.prefix + continuation[:offset], len(continuation) - offset)
                    except ValueError:
                        break
                    if forced:
                        for token in forced:
                            if continuation[offset] != token:
                                break
                            row[offset] = 0.0
                            offset += 1
                        else:
                            continue
                        break
                    pending.append(ScoreRequest(request.prefix + continuation[:offset],
                                                (continuation[offset:offset + count],), request.sampling))
                    routes.append((row, offset, count))
                    offset += count
        scored = self.backend.score_batch(pending) if pending else []
        if len(scored) != len(routes):
            raise RuntimeError("invalid thinking-budget score count")
        for (row, start, count), values in zip(routes, scored, strict=True):
            if len(values) != count:
                raise RuntimeError("invalid thinking-budget score length")
            row[start:start + count] = values
        return [tuple(row) for row in rows]

    def direct_generate(self, prefix, *, max_new_tokens, num_beams=1):
        if num_beams != 1:
            if not getattr(self.backend, "supports_constrained_beam_search", False):
                raise ValueError("this beam backend does not support a separate thinking budget; "
                                 "use transformers/asynchronous vllm or set ar.thinking=null")
            def constraint(tokens):
                _, forced = self._plan(tokens, 1)
                return forced[0] if forced else None
            return self.backend.direct_generate(prefix, max_new_tokens=max_new_tokens, num_beams=num_beams,
                                                token_constraint=constraint)
        tokens = ()
        self.direct_forward_token_slots = 0
        while len(tokens) < max_new_tokens:
            count, forced = self._plan(prefix + tokens, max_new_tokens - len(tokens))
            chunk = forced or tuple(self.backend.direct_generate(prefix + tokens, max_new_tokens=count, num_beams=1))
            if not chunk:
                raise RuntimeError("empty greedy generation")
            if not forced:
                self.direct_forward_token_slots += len(prefix) + len(tokens) + max(0, len(chunk) - 1)
            tokens += chunk
            if self.tokenizer.eos_token_id in chunk:
                break
        return tokens


class BudgetSession:
    """Continue after the budget boundary without waiting for sibling requests."""

    def __init__(self, backend, inner):
        self.backend, self.inner = backend, inner
        self.requests, self.buffers, self.routes, self.phases, self.slots = {}, {}, {}, {}, {}
        self.weights: dict[str, float] = {}
        self.ready: deque[GenerationEvent] = deque()

    def submit(self, requests, *, phase="rollout", checkpoint=None):
        pending: list[GenerationRequest] = []
        for request in requests:
            key = request.request_id
            if key in self.requests:
                raise ValueError("duplicate thinking-budget request id")
            self.requests[key], self.buffers[key] = request, []
            self.phases[key], self.slots[key] = (phase, checkpoint), 0
            if request.log_weight_stop is not None:
                self.weights[key] = request.log_weight_stop.start
            self._advance(key, pending)
        if pending:
            self.inner.submit(pending, phase=phase, checkpoint=checkpoint)

    def _advance(self, key, pending=None):
        request, tokens = self.requests[key], self.buffers[key]
        prefix = request.prefix + tuple(tokens)
        remaining = request.max_new_tokens - len(tokens)
        count, forced = self.backend._plan(prefix, remaining)
        if forced:
            tokens.extend(forced)
            stopped = any((request.prefix + tuple(tokens))[-len(marker):] == marker for marker in request.stop_sequences)
            finished = stopped or len(tokens) == request.max_new_tokens
            self.ready.append(GenerationEvent(SequenceSample(
                prefix=request.prefix, token_ids=forced, token_logprobs=(0.0,) * len(forced),
                policy_id=request.sampling.policy_id, model_id=self.backend.model_id, request_id=key,
                finish_reason="stop" if stopped else "length",
                reference_policy_id=request.reference_policy.policy_id,
                reference_token_logprobs=(0.0,) * len(forced), token_cdf_bounds=((-1.0, 1.0),) * len(forced),
                # The delimiter is not a model confidence observation. Thinking
                # rewards exclude it; full-sequence rewards can rescore it.
                token_statistics=(float("nan"),) * len(forced) if request.statistic is not None else None,
                forward_token_slots=self.slots[key] if finished else None), finished))
            if not finished:
                self._advance(key, pending)
            return
        offset = len(tokens)
        draft = request.draft
        stop = request.log_weight_stop
        if offset and stop is not None:
            # A resumed early-rejection request needs the already accumulated
            # weight. Ordinary streaming requests do not use this optimization.
            stop = replace(stop, start=self.weights[key])
        inner = replace(request, prefix=prefix, max_new_tokens=count,
                        request_id=f"{key}:budget:{offset}", uniform_offset=request.uniform_offset + offset,
                        draft=draft.after(offset) if draft and tuple(tokens) == draft.token_ids[:offset] else None,
                        log_weight_stop=stop)
        self.routes[inner.request_id] = (key, inner)
        phase, checkpoint = self.phases[key]
        if pending is not None:
            pending.append(inner)
        else:
            checkpoint = None if checkpoint is None or checkpoint <= offset else checkpoint - offset
            self.inner.submit([inner], phase=phase, checkpoint=checkpoint)

    def next_event(self):
        if self.ready:
            return self.ready.popleft()
        event = self.inner.next_event()
        sample = event.delta
        key, inner = self.routes[sample.request_id]
        request, tokens = self.requests[key], self.buffers[key]
        if (sample.prefix != inner.prefix or sample.model_id != self.backend.backend.model_id
                or sample.policy_id != request.sampling.policy_id):
            raise RuntimeError("thinking-budget stream changed request identity")
        tokens.extend(sample.token_ids)
        if len(tokens) > request.max_new_tokens:
            raise RuntimeError("thinking-budget stream exceeded total length")
        if request.log_weight_stop is not None:
            previous = self.weights.get(key, request.log_weight_stop.start)
            self.weights[key] = replace(request.log_weight_stop, start=previous).weight(
                sample.reference_token_logprobs or (), sample.token_logprobs)
        if event.finished:
            self.slots[key] = (None if self.slots[key] is None or sample.forward_token_slots is None
                               else self.slots[key] + sample.forward_token_slots)
        finished = event.finished and (len(tokens) == request.max_new_tokens or sample.finish_reason != "length")
        if event.finished and not finished:
            self._advance(key)
        return GenerationEvent(replace(sample, prefix=request.prefix, request_id=key,
                                       model_id=self.backend.model_id,
                                       forward_token_slots=self.slots[key] if finished else None), finished)

    def seal(self):
        self.inner.seal()

    def stop(self, request_id):
        for inner_id, (key, _) in tuple(self.routes.items()):
            if key == request_id:
                self.inner.stop(inner_id)
                del self.routes[inner_id]
        self.ready = deque(event for event in self.ready if event.delta.request_id != request_id)
        for records in (self.requests, self.buffers, self.phases, self.slots, self.weights):
            records.pop(request_id, None)

    def close(self):
        self.inner.close()
        self.ready.clear()
        for records in (self.requests, self.buffers, self.routes, self.phases, self.slots, self.weights):
            records.clear()
