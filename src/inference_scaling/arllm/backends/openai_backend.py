"""OpenAI-compatible HTTP inference backend adapter.

Adapts any external OpenAI-compatible completion server (``vllm serve``,
SGLang, MindIE, TGI, etc.) to the :class:`AutoregressiveBackend` protocol
using only the Python standard library, keeping the algorithm and serve layers
decoupled from any single local engine runtime.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
from math import fsum
import threading
from typing import Any, Mapping, Sequence
from urllib.request import Request, urlopen

from inference_scaling.arllm.backends.capabilities import EngineCapabilities
from inference_scaling.arllm.config import SamplingConfig, TokenPenalty
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest, SequenceSample, TokenSequence, TokenStatistic
from inference_scaling.shared.compute import dense_forward_flops
from inference_scaling.shared.rng import SeedStream


@dataclass(frozen=True, slots=True)
class OpenAIBackendSnapshot:
    sample_calls: int
    score_calls: int
    sampled_sequences: int
    generated_tokens: int
    prefill_tokens: int
    scored_tokens: int
    generation_forward_token_slots: int
    score_forward_token_slots: int
    estimated_dense_forward_flops: int
    engine_requests: int


class OpenAIBackend:
    """Thin adapter for remote OpenAI-compatible ``/v1/completions`` servers."""

    supports_native_continuous_batching = True
    supports_native_token_stops = True

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        tokenizer: Any,
        parameter_count: int = 1,
        api_key: str | None = None,
        timeout_seconds: float = 600.0,
        max_workers: int = 16,
        token_penalty: TokenPenalty | None = None,
        extra_body: Mapping[str, Any] | None = None,
        transport: Any | None = None,
    ) -> None:
        if parameter_count <= 0:
            raise ValueError("parameter_count must be positive")
        if timeout_seconds <= 0 or max_workers <= 0:
            raise ValueError("timeout_seconds and max_workers must be positive")
        self._base_url = base_url.rstrip("/")
        self._model = str(model)
        self.tokenizer = tokenizer
        self._parameter_count = int(parameter_count)
        self._api_key = api_key
        self._timeout = float(timeout_seconds)
        self._penalty = token_penalty
        self._extra_body = dict(extra_body or {})
        self._transport = transport
        self._model_id = (
            f"openai:{self._base_url}:{self._model}"
            if token_penalty is None
            else f"openai:{self._base_url}:{self._model}|{token_penalty.penalty_id}"
        )
        self._pool = ThreadPoolExecutor(max_workers=int(max_workers), thread_name_prefix="openai-backend")
        self._lock = threading.Lock()
        self._closed = False
        self._sample_calls = 0
        self._score_calls = 0
        self._sampled_sequences = 0
        self._generated_tokens = 0
        self._prefill_tokens = 0
        self._scored_tokens = 0
        self._generation_forward_token_slots = 0
        self._score_forward_token_slots = 0
        self._estimated_dense_forward_flops = 0
        self._engine_requests = 0

    @classmethod
    def from_config(
        cls,
        *,
        base_url: str,
        model: str,
        tokenizer_name_or_path: str,
        tokenizer_revision: str | None = None,
        tokenizer_kwargs: Mapping[str, Any] | None = None,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        parameter_count: int | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 600.0,
        max_workers: int = 16,
        token_penalty: Mapping[str, Any] | None = None,
        extra_body: Mapping[str, Any] | None = None,
    ) -> "OpenAIBackend":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            revision=tokenizer_revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            **dict(tokenizer_kwargs or {}),
        )
        penalty = (
            None
            if token_penalty is None
            else TokenPenalty.from_words(tokenizer, token_penalty["words"], token_penalty["strength"])
        )
        return cls(
            base_url=base_url,
            model=model,
            tokenizer=tokenizer,
            parameter_count=parameter_count or 1,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_workers=max_workers,
            token_penalty=penalty,
            extra_body=extra_body,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            backend="openai",
            device="remote",
            streaming=False,
            native_continuous_batching=True,
            native_token_stops=True,
            scalar_self_certainty=False,
            pre_mask_statistics=False,
            exact_scoring=False,
            constrained_beam_search=False,
            sampling_constraints=False,
        )

    def execution_config(self) -> dict[str, Any]:
        return {"backend": "openai", "base_url": self._base_url, "model": self._model}

    def encode(self, text: str, *, add_special_tokens: bool = True) -> TokenSequence:
        return tuple(int(token) for token in self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def decode(self, tokens: TokenSequence, *, skip_special_tokens: bool = True) -> str:
        return str(self.tokenizer.decode(list(tokens), skip_special_tokens=skip_special_tokens))

    def records(self, statistic: TokenStatistic, sampling: SamplingConfig) -> bool:
        return statistic.top_k is not None and statistic.matches(sampling)

    def _post(self, path: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("OpenAI backend is closed")
        if self._transport is not None:
            return dict(self._transport(path, payload))
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        req = Request(
            f"{self._base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(req, timeout=self._timeout) as response:
            return dict(json.loads(response.read().decode("utf-8")))

    def _logit_bias(self) -> dict[str, float] | None:
        if self._penalty is None:
            return None
        return {str(token): -float(self._penalty.strength) for token in self._penalty.token_ids}

    def _sample_one(self, request: GenerationRequest) -> tuple[SequenceSample, int, int]:
        policy = request.sampling
        seed = (
            request.seed
            if not request.uniform_offset
            else SeedStream(request.seed).derive("uniform-offset", request.uniform_offset)
        )
        statistic = request.statistic if (request.statistic is not None and self.records(request.statistic, policy)) else None
        stops = [int(stop[0]) for stop in request.stop_sequences if len(stop) == 1]
        if policy.eos_token_id is not None:
            stops = [int(policy.eos_token_id), *stops]
        payload: dict[str, Any] = {
            **self._extra_body,
            "model": self._model,
            "prompt": list(request.prefix),
            "max_tokens": int(request.max_new_tokens),
            "temperature": float(policy.temperature),
            "top_p": float(policy.top_p),
            "seed": int(seed),
            "logprobs": int(statistic.top_k) if statistic is not None and statistic.top_k is not None else 1,
            "return_tokens_as_token_ids": True,
            "ignore_eos": True,
        }
        if policy.top_k is not None:
            payload["top_k"] = int(policy.top_k)
        if stops:
            payload["stop_token_ids"] = stops
        if (bias := self._logit_bias()) is not None:
            payload["logit_bias"] = bias
        response = self._post("/v1/completions", payload)
        choices = response.get("choices") or []
        if len(choices) != 1:
            raise RuntimeError("OpenAI backend expected exactly one choice per request")
        choice = choices[0]
        logprob_info = choice.get("logprobs") or {}
        raw_tokens = choice.get("token_ids") or logprob_info.get("tokens")
        raw_lps = logprob_info.get("token_logprobs")
        top_logprobs = logprob_info.get("top_logprobs") or []
        if raw_tokens is None or raw_lps is None or len(raw_tokens) != len(raw_lps):
            raise RuntimeError("OpenAI backend requires token IDs and per-token logprobs from the server")
        tokens = tuple(int(str(tok).removeprefix("token_id:")) for tok in raw_tokens)
        token_lps = tuple(float(lp) for lp in raw_lps)
        stats: tuple[float, ...] | None = None
        if statistic is not None and statistic.top_k is not None:
            if len(top_logprobs) != len(tokens):
                raise RuntimeError("OpenAI backend omitted top_logprobs required by the statistic")
            k = statistic.top_k
            computed = []
            for row in top_logprobs:
                vals = sorted((float(v) for v in row.values()), reverse=True)
                if len(vals) < k:
                    raise RuntimeError("OpenAI backend returned fewer top_logprobs than requested")
                computed.append(-fsum(vals[:k]) / k)
            stats = tuple(computed)
        finish_reason = str(choice.get("finish_reason") or "length")
        if policy.eos_token_id is not None and tokens and tokens[-1] == policy.eos_token_id:
            finish_reason = "eos"
        reference_lps = token_lps if policy == request.reference_policy else None
        slots = len(request.prefix) + max(0, len(tokens) - 1)
        sample = SequenceSample(
            prefix=request.prefix,
            token_ids=tokens,
            token_logprobs=token_lps,
            policy_id=policy.policy_id,
            model_id=self.model_id,
            request_id=request.request_id,
            finish_reason=finish_reason,
            reference_token_logprobs=reference_lps,
            reference_policy_id=None if reference_lps is None else request.reference_policy.policy_id,
            token_statistics=stats,
            forward_token_slots=slots,
        )
        return sample, len(request.prefix), slots

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        if not requests:
            return []
        if len(requests) == 1:
            results = [self._sample_one(requests[0])]
        else:
            results = list(self._pool.map(self._sample_one, requests))
        samples = [item[0] for item in results]
        prefill = sum(item[1] for item in results)
        slots = sum(item[2] for item in results)
        with self._lock:
            self._sample_calls += 1
            self._sampled_sequences += len(samples)
            self._generated_tokens += sum(len(s.token_ids) for s in samples)
            self._prefill_tokens += prefill
            self._generation_forward_token_slots += slots
            self._estimated_dense_forward_flops += dense_forward_flops(self._parameter_count, slots)
            self._engine_requests += len(samples)
        return samples

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        raise NotImplementedError("OpenAIBackend does not implement exact prompt rescoring; use on-policy IS or local scoring")

    def direct_generate(self, prefix: TokenSequence, *, max_new_tokens: int, num_beams: int = 1) -> TokenSequence:
        if num_beams != 1:
            raise ValueError("OpenAIBackend direct_generate supports greedy (num_beams=1) only")
        payload: dict[str, Any] = {
            **self._extra_body,
            "model": self._model,
            "prompt": list(prefix),
            "max_tokens": int(max_new_tokens),
            "temperature": 0.0,
            "return_tokens_as_token_ids": True,
        }
        if (bias := self._logit_bias()) is not None:
            payload["logit_bias"] = bias
        response = self._post("/v1/completions", payload)
        choice = response["choices"][0]
        raw_tokens = choice.get("token_ids") or (choice.get("logprobs") or {}).get("tokens")
        if raw_tokens is not None:
            return tuple(int(str(tok).removeprefix("token_id:")) for tok in raw_tokens)
        return self.encode(str(choice.get("text", "")), add_special_tokens=False)

    def snapshot(self) -> OpenAIBackendSnapshot:
        with self._lock:
            return OpenAIBackendSnapshot(
                sample_calls=self._sample_calls,
                score_calls=self._score_calls,
                sampled_sequences=self._sampled_sequences,
                generated_tokens=self._generated_tokens,
                prefill_tokens=self._prefill_tokens,
                scored_tokens=self._scored_tokens,
                generation_forward_token_slots=self._generation_forward_token_slots,
                score_forward_token_slots=self._score_forward_token_slots,
                estimated_dense_forward_flops=self._estimated_dense_forward_flops,
                engine_requests=self._engine_requests,
            )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._pool.shutdown(wait=False)


__all__ = ["OpenAIBackend", "OpenAIBackendSnapshot"]
