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

    def score_statistics_batch(self, requests, *, confidence_top_k):
        # Statistic spans are chosen explicitly by the reward; these remain original model statistics.
        return getattr(self.backend, "score_statistics_batch")(requests, confidence_top_k=confidence_top_k)

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
        bounds: list[list[tuple[float, float]] | None] = [[] for _ in requests]
        done = [False] * len(requests)
        for request in requests:
            if self._stop_end(self._generated(request.prefix)) is not None:
                raise ValueError("the prefix already ended at a stop boundary")
        while True:
            pending, indices = list[GenerationRequest](), list[int]()
            for index, request in enumerate(requests):
                offset = len(tokens[index])
                if done[index] or offset >= request.max_new_tokens:
                    continue
                pending.append(replace(
                    request, prefix=request.prefix + tuple(tokens[index]), max_new_tokens=request.max_new_tokens - offset,
                    sampling=self._inner_policy(request.sampling), request_id=f"{request.request_id}:stop:{offset}",
                    stop_sequences=self.stop_sequences, uniform_offset=request.uniform_offset + offset,
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
                reference = references[index]
                if reference is not None and sample.reference_policy_id == inner_request.reference_policy.policy_id:
                    reference.extend((sample.reference_token_logprobs or ())[:keep])
                else:
                    references[index] = None
                bound = bounds[index]
                if bound is not None and sample.token_cdf_bounds is not None:
                    bound.extend(sample.token_cdf_bounds[:keep])
                else:
                    bounds[index] = None
                if end is not None:
                    done[index] = True
                elif sample.finish_reason != "stop" and len(sample.token_ids) != inner_request.max_new_tokens:
                    raise RuntimeError("generation ended without EOS or a thinking boundary")
        return [
            SequenceSample(
                prefix=request.prefix, token_ids=tuple(tokens[index]), token_logprobs=tuple(logs[index]),
                model_id=self.model_id, policy_id=request.sampling.policy_id, request_id=request.request_id,
                finish_reason="stop" if done[index] else "length",
                reference_token_logprobs=None if references[index] is None else tuple(references[index] or ()),
                reference_policy_id=None if references[index] is None else request.reference_policy.policy_id,
                token_cdf_bounds=None if bounds[index] is None else tuple(bounds[index] or ()),
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


__all__ = ["StoppedSequenceBackend"]
