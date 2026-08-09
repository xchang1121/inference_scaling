"""vLLM backends with exact sampling-policy accounting.

The synchronous backend deliberately submits one vLLM request per framework
``GenerationRequest``.  This retains request-local seeds while still allowing
vLLM to batch requests and reuse their common prefixes.  Collapsing repeated
prompts into ``SamplingParams(n=...)`` would replace those independent seeds by
one group seed and make results depend on how callers happened to be batched.

vLLM can return processed log-probabilities for generated tokens, but prompt
log-probabilities are always raw model probabilities.  Native continuation
scoring is therefore exact for the full-support temperature-one policy.  Other
policies are delegated to an optional exact scoring backend instead of silently
using incorrect importance ratios.
"""

from __future__ import annotations

import inspect
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference_scaling.compute import dense_forward_flops
from inference_scaling.config import SamplingConfig
from inference_scaling.types import (
    AutoregressiveBackend,
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
    TokenSequence,
)


@dataclass(frozen=True, slots=True)
class VLLMBackendSnapshot:
    sample_calls: int
    score_calls: int
    sampled_sequences: int
    generated_tokens: int
    prefill_tokens: int
    shared_prefill_tokens_saved: int
    scored_tokens: int
    generation_forward_token_slots: int
    score_forward_token_slots: int
    estimated_dense_forward_flops: int
    engine_requests: int
    native_score_sequences: int
    delegated_score_sequences: int


def _checkpoint_parameter_count(model_name_or_path: str) -> int | None:
    """Read local safetensor shapes without materializing model weights."""

    root = Path(model_name_or_path)
    if not root.exists():
        return None
    files = [root] if root.is_file() and root.suffix == ".safetensors" else []
    if root.is_dir():
        files = sorted(root.glob("*.safetensors"))
    if not files:
        return None
    try:
        from safetensors import safe_open
    except ImportError:
        return None

    total = 0
    names: set[str] = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in names:
                    raise ValueError(f"duplicate tensor {name!r} across checkpoint shards")
                names.add(name)
                size = 1
                for dimension in handle.get_slice(name).get_shape():
                    size *= int(dimension)
                total += size
    return total


def _logprob_value(position: Any, token_id: int) -> float:
    if position is None:
        raise RuntimeError("vLLM omitted a required token log-probability")
    try:
        value = position[int(token_id)]
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"vLLM did not return the chosen token {token_id} in its log-probabilities"
        ) from error
    return float(getattr(value, "logprob", value))


class VLLMBackend:
    """Synchronous offline vLLM implementation of ``AutoregressiveBackend``.

    The supplied engine must use ``logprobs_mode='processed_logprobs'`` and
    ``generation_config='vllm'``.  Prefer :meth:`from_pretrained`, which enforces
    both settings and explicitly enables automatic prefix caching.
    """

    supports_native_continuous_batching = False

    def __init__(
        self,
        engine: Any,
        tokenizer: Any,
        *,
        model_id: str,
        parameter_count: int,
        sampling_params_factory: Callable[..., Any],
        tokens_prompt_factory: Callable[..., Any] | None = None,
        scoring_backend: AutoregressiveBackend | None = None,
        lora_request: Any | None = None,
    ) -> None:
        if parameter_count <= 0:
            raise ValueError("parameter_count must be positive")
        if scoring_backend is not None and scoring_backend.model_id != model_id:
            raise ValueError("the exact scoring backend must use the same model_id")
        self._engine = engine
        self.tokenizer = tokenizer
        self._model_id = str(model_id)
        self._parameter_count = int(parameter_count)
        self._sampling_params_factory = sampling_params_factory
        self._tokens_prompt_factory = tokens_prompt_factory
        self._scoring_backend = scoring_backend
        self._lora_request = lora_request
        pad = getattr(tokenizer, "pad_token_id", None)
        eos = getattr(tokenizer, "eos_token_id", None)
        if pad is None and eos is not None:
            tokenizer.pad_token_id = eos
        self.bos_token_id = (
            None
            if getattr(tokenizer, "bos_token_id", None) is None
            else int(tokenizer.bos_token_id)
        )
        self._engine_lock = threading.RLock()
        self._statistics_lock = threading.Lock()
        self._closed = False
        self._sample_calls = 0
        self._score_calls = 0
        self._sampled_sequences = 0
        self._generated_tokens = 0
        self._prefill_tokens = 0
        self._shared_prefill_tokens_saved = 0
        self._scored_tokens = 0
        self._generation_forward_token_slots = 0
        self._score_forward_token_slots = 0
        self._estimated_dense_forward_flops = 0
        self._engine_requests = 0
        self._native_score_sequences = 0
        self._delegated_score_sequences = 0

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        adapter_name_or_path: str | None = None,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        data_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        max_num_batched_tokens: int | None = None,
        quantization: str | None = None,
        enforce_eager: bool = False,
        trust_remote_code: bool = False,
        revision: str | None = None,
        download_dir: str | None = None,
        seed: int = 0,
        parameter_count: int | None = None,
        scoring_backend: AutoregressiveBackend | None = None,
        enable_prefix_caching: bool = True,
        max_lora_rank: int = 16,
        engine_kwargs: dict[str, Any] | None = None,
    ) -> "VLLMBackend":
        try:
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams, TokensPrompt
        except ImportError as error:  # pragma: no cover - optional GPU installation
            raise ModuleNotFoundError(
                "VLLMBackend.from_pretrained requires the project's vllm extra"
            ) from error

        base_model = model_name_or_path
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            local_files_only=Path(base_model).exists(),
            trust_remote_code=trust_remote_code,
            revision=revision,
            cache_dir=download_dir,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        kwargs: dict[str, Any] = {
            "model": base_model,
            "dtype": dtype,
            "tensor_parallel_size": int(tensor_parallel_size),
            "data_parallel_size": int(data_parallel_size),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "quantization": quantization,
            "enforce_eager": bool(enforce_eager),
            "trust_remote_code": bool(trust_remote_code),
            "revision": revision,
            "download_dir": download_dir,
            "seed": int(seed),
            "enable_prefix_caching": bool(enable_prefix_caching),
            "generation_config": "vllm",
            "logprobs_mode": "processed_logprobs",
            "enable_lora": adapter_name_or_path is not None,
            "max_lora_rank": int(max_lora_rank),
        }
        optional = {
            "max_model_len": max_model_len,
            "max_num_seqs": max_num_seqs,
            "max_num_batched_tokens": max_num_batched_tokens,
        }
        kwargs.update({name: value for name, value in optional.items() if value is not None})
        if engine_kwargs:
            protected = {
                "model",
                "generation_config",
                "logprobs_mode",
                "enable_prefix_caching",
            }
            overlap = protected.intersection(engine_kwargs)
            if overlap:
                raise ValueError(
                    "engine_kwargs cannot override correctness-critical settings: "
                    + ", ".join(sorted(overlap))
                )
            kwargs.update(engine_kwargs)
        engine = LLM(**kwargs)

        lora_request = None
        if adapter_name_or_path is not None:
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest("inference-scaling", 1, adapter_name_or_path)
        model_id = (
            base_model
            if adapter_name_or_path is None
            else f"{base_model}+adapter:{adapter_name_or_path}"
        )
        counted = parameter_count or _checkpoint_parameter_count(base_model)
        if counted is None:
            raise ValueError(
                "parameter_count could not be read from a local safetensors checkpoint; "
                "pass parameter_count explicitly"
            )
        return cls(
            engine,
            tokenizer,
            model_id=model_id,
            parameter_count=counted,
            sampling_params_factory=SamplingParams,
            tokens_prompt_factory=TokensPrompt,
            scoring_backend=scoring_backend,
            lora_request=lora_request,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def parameter_count(self) -> int:
        return self._parameter_count

    def _model_prefix(self, prefix: TokenSequence) -> TokenSequence:
        if prefix:
            return prefix
        if self.bos_token_id is None:
            raise ValueError("an empty prefix requires a tokenizer bos_token_id")
        return (self.bos_token_id,)

    def _prompt(self, prefix: TokenSequence) -> Any:
        token_ids = list(self._model_prefix(prefix))
        if self._tokens_prompt_factory is None:
            return {"prompt_token_ids": token_ids}
        return self._tokens_prompt_factory(prompt_token_ids=token_ids)

    def _sampling_params(self, request: GenerationRequest) -> Any:
        policy = request.sampling
        return self._sampling_params_factory(
            max_tokens=int(request.max_new_tokens),
            temperature=float(policy.temperature),
            top_p=float(policy.top_p),
            top_k=0 if policy.top_k is None else int(policy.top_k),
            seed=int(request.seed),
            logprobs=0,
            flat_logprobs=False,
            ignore_eos=True,
            stop_token_ids=(
                [] if policy.eos_token_id is None else [int(policy.eos_token_id)]
            ),
            detokenize=False,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )

    def _score_params(self) -> Any:
        return self._sampling_params_factory(
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            seed=0,
            prompt_logprobs=0,
            ignore_eos=True,
            detokenize=False,
            skip_special_tokens=False,
            spaces_between_special_tokens=False,
        )

    def _generate(self, prompts: Sequence[Any], params: Any) -> list[Any]:
        if self._closed:
            raise RuntimeError("vLLM backend is closed")
        kwargs = {
            "sampling_params": params,
            "use_tqdm": False,
        }
        if self._lora_request is not None:
            kwargs["lora_request"] = self._lora_request
        with self._engine_lock:
            outputs = self._engine.generate(list(prompts), **kwargs)
        return list(outputs)

    @staticmethod
    def _completion(output: Any) -> Any:
        completions = getattr(output, "outputs", None)
        if not completions or len(completions) != 1:
            raise RuntimeError("vLLM must return exactly one completion per request")
        return completions[0]

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        if not requests:
            return []
        outputs = self._generate(
            [self._prompt(request.prefix) for request in requests],
            [self._sampling_params(request) for request in requests],
        )
        if len(outputs) != len(requests):
            raise RuntimeError("vLLM returned an invalid number of request outputs")

        samples: list[SequenceSample] = []
        prefill_tokens = 0
        cached_tokens = 0
        forward_slots = 0
        for request, output in zip(requests, outputs, strict=True):
            completion = self._completion(output)
            tokens = tuple(int(token) for token in completion.token_ids)
            positions = completion.logprobs
            if positions is None or len(positions) != len(tokens):
                raise RuntimeError("vLLM returned an invalid generated log-probability shape")
            token_logprobs = tuple(
                _logprob_value(position, token)
                for position, token in zip(positions, tokens, strict=True)
            )
            finish_reason = str(getattr(completion, "finish_reason", "length") or "length")
            eos = request.sampling.eos_token_id
            if eos is not None and tokens and tokens[-1] == eos:
                finish_reason = "eos"
            samples.append(
                SequenceSample(
                    prefix=request.prefix,
                    token_ids=tokens,
                    token_logprobs=token_logprobs,
                    policy_id=request.sampling.policy_id,
                    model_id=self.model_id,
                    request_id=request.request_id,
                    finish_reason=finish_reason,
                )
            )
            prompt_length = len(self._model_prefix(request.prefix))
            cached = min(prompt_length, int(getattr(output, "num_cached_tokens", 0) or 0))
            prefill_tokens += prompt_length - cached
            cached_tokens += cached
            forward_slots += prompt_length - cached + max(0, len(tokens) - 1)

        with self._statistics_lock:
            self._sample_calls += 1
            self._sampled_sequences += len(samples)
            self._generated_tokens += sum(len(sample.token_ids) for sample in samples)
            self._prefill_tokens += prefill_tokens
            self._shared_prefill_tokens_saved += cached_tokens
            self._generation_forward_token_slots += forward_slots
            self._estimated_dense_forward_flops += dense_forward_flops(
                self.parameter_count, forward_slots
            )
            self._engine_requests += len(requests)
        return samples

    @staticmethod
    def _supports_native_score(sampling: SamplingConfig | None) -> bool:
        policy = sampling or SamplingConfig()
        return policy.temperature == 1 and policy.top_p == 1 and policy.top_k is None

    def _score_native(
        self,
        items: Sequence[tuple[int, ScoreRequest, TokenSequence]],
        results: list[tuple[float, ...]],
    ) -> tuple[int, int]:
        if not items:
            return 0, 0
        prompts = [
            self._prompt(self._model_prefix(request.prefix) + continuation)
            for _, request, continuation in items
        ]
        outputs = self._generate(prompts, self._score_params())
        if len(outputs) != len(items):
            raise RuntimeError("vLLM returned an invalid number of scoring outputs")
        forward_slots = 0
        for (index, request, continuation), output in zip(items, outputs, strict=True):
            prompt_logprobs = getattr(output, "prompt_logprobs", None)
            prefix = self._model_prefix(request.prefix)
            if prompt_logprobs is None or len(prompt_logprobs) != len(prefix) + len(continuation):
                raise RuntimeError("vLLM returned an invalid prompt log-probability shape")
            positions = prompt_logprobs[len(prefix) :]
            results[index] = tuple(
                _logprob_value(position, token)
                for position, token in zip(positions, continuation, strict=True)
            )
            forward_slots += len(prefix) + len(continuation)
        return len(items), forward_slots

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        flattened = [
            (request, continuation)
            for request in requests
            for continuation in request.continuations
        ]
        results: list[tuple[float, ...]] = [()] * len(flattened)
        native: list[tuple[int, ScoreRequest, TokenSequence]] = []
        delegated: list[tuple[int, ScoreRequest, TokenSequence]] = []
        for index, (request, continuation) in enumerate(flattened):
            if not continuation:
                continue
            item = (index, request, continuation)
            if self._supports_native_score(request.sampling):
                native.append(item)
            else:
                delegated.append(item)

        native_count, score_slots = self._score_native(native, results)
        if delegated:
            if self._scoring_backend is None:
                policies = sorted({item[1].sampling.policy_id for item in delegated if item[1].sampling})
                raise ValueError(
                    "vLLM prompt log-probabilities cannot exactly score temperature/top-k/top-p "
                    "policies; configure an exact scoring_backend for: " + ", ".join(policies)
                )
            delegated_requests = [
                ScoreRequest(request.prefix, (continuation,), request.sampling)
                for _, request, continuation in delegated
            ]
            delegated_outputs = self._scoring_backend.score_batch(delegated_requests)
            if len(delegated_outputs) != len(delegated):
                raise RuntimeError("exact scoring backend returned an invalid result count")
            for (index, _, continuation), scores in zip(
                delegated, delegated_outputs, strict=True
            ):
                if len(scores) != len(continuation):
                    raise RuntimeError("exact scoring backend returned an invalid score shape")
                results[index] = scores

        with self._statistics_lock:
            self._score_calls += 1
            self._scored_tokens += sum(len(continuation) for _, continuation in flattened)
            self._score_forward_token_slots += score_slots
            self._estimated_dense_forward_flops += dense_forward_flops(
                self.parameter_count, score_slots
            )
            self._engine_requests += native_count
            self._native_score_sequences += native_count
            self._delegated_score_sequences += len(delegated)
        return results

    def snapshot(self) -> VLLMBackendSnapshot:
        with self._statistics_lock:
            return VLLMBackendSnapshot(
                sample_calls=self._sample_calls,
                score_calls=self._score_calls,
                sampled_sequences=self._sampled_sequences,
                generated_tokens=self._generated_tokens,
                prefill_tokens=self._prefill_tokens,
                shared_prefill_tokens_saved=self._shared_prefill_tokens_saved,
                scored_tokens=self._scored_tokens,
                generation_forward_token_slots=self._generation_forward_token_slots,
                score_forward_token_slots=self._score_forward_token_slots,
                estimated_dense_forward_flops=self._estimated_dense_forward_flops,
                engine_requests=self._engine_requests,
                native_score_sequences=self._native_score_sequences,
                delegated_score_sequences=self._delegated_score_sequences,
            )

    def encode(self, text: str, *, add_special_tokens: bool = True) -> TokenSequence:
        return tuple(
            int(token)
            for token in self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        )

    def decode(self, tokens: TokenSequence, *, skip_special_tokens: bool = True) -> str:
        return str(
            self.tokenizer.decode(list(tokens), skip_special_tokens=skip_special_tokens)
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        shutdown = getattr(self._engine, "shutdown", None)
        if shutdown is None:
            shutdown = getattr(getattr(self._engine, "llm_engine", None), "shutdown", None)
        if shutdown is not None:
            result = shutdown()
            if inspect.isawaitable(result):
                raise RuntimeError("an asynchronous vLLM engine requires AsyncVLLMBackend")

    def __enter__(self) -> "VLLMBackend":
        return self

    def __exit__(self, _type, _value, _traceback) -> None:
        self.close()

