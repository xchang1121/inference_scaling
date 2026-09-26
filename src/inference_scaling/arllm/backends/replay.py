"""Draft replay for backends that sample by inverse CDF from a request's uniform stream.

A draft token is kept exactly when the request's own uniform falls in the token's
CDF interval, which is when plain generation would draw it; the first token that
fails is left to the backend, which continues the same uniform stream. The output
is therefore the plain output, and the kept tokens cost no model call.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

from inference_scaling.arllm.types import GenerationRequest, SequenceSample
from inference_scaling.shared.rng import uniform_stream
from inference_scaling.shared.types import TokenSequence


def _end(request: GenerationRequest, generated: TokenSequence, honors_stops: bool) -> str | None:
    """Why plain generation ends right after ``generated``: EOS, then a stop sequence, then the length limit."""

    if generated[-1] == request.sampling.eos_token_id:
        return "eos"
    if honors_stops and any(generated[-len(stop):] == stop for stop in request.stop_sequences if len(generated) >= len(stop)):
        return "stop"
    return "length" if len(generated) == request.max_new_tokens else None


def sample_with_drafts(
    requests: Sequence[GenerationRequest],
    generate: Callable[[list[GenerationRequest]], list[SequenceSample]],
    *,
    model_id: str,
    honors_stops: bool,
) -> tuple[list[SequenceSample], int]:
    """Samples of ``requests`` and the number of replayed tokens; ``generate`` samples what the drafts leave."""

    kept: list[int] = []
    ends: list[str | None] = []
    tails: list[GenerationRequest] = []
    for request in requests:
        draft, count, end = request.draft, 0, None
        if draft is not None:
            limit = min(len(draft.token_ids), request.max_new_tokens)
            uniforms = uniform_stream(request.seed, request.uniform_offset, limit)
            for uniform, (below, through) in zip(uniforms, draft.token_cdf_bounds[:limit], strict=True):
                if not below < uniform <= through:
                    break
                count += 1
                if (end := _end(request, draft.token_ids[:count], honors_stops)) is not None:
                    break
        kept.append(count)
        ends.append(end)
        if end is None:
            tails.append(replace(request, prefix=request.prefix + (draft.token_ids[:count] if draft else ()),
                                 max_new_tokens=request.max_new_tokens - count,
                                 uniform_offset=request.uniform_offset + count, draft=None))
    generated = iter(generate(tails) if tails else ())
    outputs: list[SequenceSample] = []
    for request, count, end in zip(requests, kept, ends, strict=True):
        tail = next(generated) if end is None else None
        if not count:
            assert tail is not None
            outputs.append(replace(tail, prefix=request.prefix))
            continue
        head = request.draft
        assert head is not None
        references = None if tail is not None and tail.reference_token_logprobs is None else (
            head.reference_token_logprobs[:count] + (() if tail is None else tail.reference_token_logprobs or ()))
        bounds = None if tail is not None and tail.token_cdf_bounds is None else (
            head.token_cdf_bounds[:count] + (() if tail is None else tail.token_cdf_bounds or ()))
        outputs.append(SequenceSample(
            prefix=request.prefix, token_ids=head.token_ids[:count] + (() if tail is None else tail.token_ids),
            token_logprobs=head.token_logprobs[:count] + (() if tail is None else tail.token_logprobs),
            policy_id=request.sampling.policy_id, model_id=model_id, request_id=request.request_id,
            finish_reason=end if tail is None else tail.finish_reason, reference_token_logprobs=references,
            reference_policy_id=None if references is None else request.reference_policy.policy_id,
            token_cdf_bounds=bounds,
        ))
    return outputs, sum(kept)


__all__ = ["sample_with_drafts"]
