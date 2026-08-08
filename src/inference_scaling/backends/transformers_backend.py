"""Batched causal-LM backend with exact policy probabilities and KV decoding."""

from __future__ import annotations

import threading
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from inference_scaling.config import SamplingConfig
from inference_scaling.types import (
    GenerationRequest,
    ScoreRequest,
    SequenceSample,
    TokenSequence,
)

try:
    import torch
except ImportError:  # pragma: no cover - exercised in dependency-free installations
    torch = None  # type: ignore[assignment]


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError(
            "TransformersBackend requires the optional GPU dependencies; "
            "install the project's gpu extra first"
        )
    return torch


@dataclass(frozen=True, slots=True)
class TransformersBackendSnapshot:
    sample_calls: int
    score_calls: int
    sampled_sequences: int
    generated_tokens: int
    prefill_tokens: int
    shared_prefill_tokens_saved: int
    scored_tokens: int


class TransformersBackend:
    """Manual batched decoding with request-local, scheduling-independent RNG."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        model_id: str | None = None,
        device: str | Any | None = None,
    ) -> None:
        torch_module = _require_torch()
        self.model = model
        self.tokenizer = tokenizer
        self._model_id = model_id or str(
            getattr(getattr(model, "config", None), "_name_or_path", "transformers-model")
        )
        inferred_device = getattr(model, "device", None)
        self.device = torch_module.device(
            device or inferred_device or ("cuda" if torch_module.cuda.is_available() else "cpu")
        )
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        self.pad_token_id = int(pad_token_id)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        self.bos_token_id = None if bos_token_id is None else int(bos_token_id)
        self.model.eval()
        self._model_lock = threading.RLock()
        self._statistics_lock = threading.Lock()
        self._sample_calls = 0
        self._score_calls = 0
        self._sampled_sequences = 0
        self._generated_tokens = 0
        self._prefill_tokens = 0
        self._shared_prefill_tokens_saved = 0
        self._scored_tokens = 0

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device: str = "cuda",
        dtype: str = "float32",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = False,
    ) -> "TransformersBackend":
        torch_module = _require_torch()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - depends on optional install
            raise ModuleNotFoundError(
                "TransformersBackend.from_pretrained requires transformers"
            ) from error
        try:
            torch_dtype = getattr(torch_module, dtype)
        except AttributeError as error:
            raise ValueError(f"unknown torch dtype {dtype!r}") from error
        if torch_dtype != torch_module.float32:
            warnings.warn(
                "Reduced-precision logits can depend noticeably on batch shape. "
                "Use dtype='float32' when importance weights must match later rescoring.",
                RuntimeWarning,
                stacklevel=2,
            )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
        )
        model.to(torch_module.device(device))
        return cls(model, tokenizer, model_id=model_name_or_path, device=device)

    @property
    def model_id(self) -> str:
        return self._model_id

    def _model_prefix(self, prefix: TokenSequence) -> TokenSequence:
        if prefix:
            return prefix
        if self.bos_token_id is None:
            raise ValueError("an empty prefix requires a tokenizer bos_token_id")
        return (self.bos_token_id,)

    @staticmethod
    def _position_ids(attention_mask):
        position_ids = attention_mask.to(dtype=torch.long).cumsum(dim=-1) - 1
        return position_ids.masked_fill(attention_mask == 0, 0)

    @staticmethod
    def _policy_log_probs(logits, sampling: SamplingConfig | None):
        torch_module = _require_torch()
        policy = sampling or SamplingConfig()
        transformed = logits.to(dtype=torch_module.float32) / policy.temperature
        vocabulary_size = transformed.shape[-1]
        if policy.top_k is not None and policy.top_k < vocabulary_size:
            threshold = torch_module.topk(
                transformed, policy.top_k, dim=-1
            ).values[..., -1, None]
            transformed = transformed.masked_fill(transformed < threshold, float("-inf"))
        if policy.top_p < 1:
            sorted_logits, sorted_indices = torch_module.sort(
                transformed, descending=True, dim=-1
            )
            sorted_probabilities = torch_module.softmax(sorted_logits, dim=-1)
            remove = sorted_probabilities.cumsum(dim=-1) > policy.top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            remove_original = torch_module.zeros_like(remove).scatter(
                -1, sorted_indices, remove
            )
            transformed = transformed.masked_fill(remove_original, float("-inf"))
        return torch_module.log_softmax(transformed, dim=-1)

    def _padded_inputs(self, prefixes: Sequence[TokenSequence]):
        torch_module = _require_torch()
        maximum = max(len(prefix) for prefix in prefixes)
        input_ids = torch_module.full(
            (len(prefixes), maximum),
            self.pad_token_id,
            dtype=torch_module.long,
            device=self.device,
        )
        attention_mask = torch_module.zeros_like(input_ids)
        for index, prefix in enumerate(prefixes):
            values = torch_module.tensor(prefix, dtype=torch_module.long, device=self.device)
            input_ids[index, maximum - len(prefix) :] = values
            attention_mask[index, maximum - len(prefix) :] = 1
        return input_ids, attention_mask

    @staticmethod
    def _repeat_cache(cache, repeats: int):
        if cache is None:
            return None
        repeat_method = getattr(cache, "batch_repeat_interleave", None)
        if callable(repeat_method):
            repeat_method(repeats)
            return cache
        if isinstance(cache, (tuple, list)):
            repeated_layers = []
            for layer in cache:
                if not isinstance(layer, (tuple, list)):
                    return None
                repeated_layers.append(
                    tuple(value.repeat_interleave(repeats, dim=0) for value in layer)
                )
            return tuple(repeated_layers)
        return None

    def _sample_same_policy(
        self,
        indexed_requests: Sequence[tuple[int, GenerationRequest]],
    ) -> list[tuple[int, SequenceSample]]:
        torch_module = _require_torch()
        requests = [request for _, request in indexed_requests]
        sampling = requests[0].sampling
        prefixes = [self._model_prefix(request.prefix) for request in requests]
        uniforms = [
            np.random.default_rng(request.seed).random(request.max_new_tokens)
            for request in requests
        ]
        token_lists: list[list[int]] = [[] for _ in requests]
        logprob_lists: list[list[float]] = [[] for _ in requests]
        active = torch_module.ones(len(requests), dtype=torch_module.bool, device=self.device)
        finish_reasons = ["length"] * len(requests)
        maximum_new_tokens = max(request.max_new_tokens for request in requests)
        prefill_tokens = sum(len(prefix) for prefix in prefixes)
        shared_prefill_tokens_saved = 0

        with self._model_lock, torch_module.inference_mode():
            shared_prefix = len(requests) > 1 and all(
                prefix == prefixes[0] for prefix in prefixes[1:]
            )
            cache = None
            if shared_prefix:
                shared_input_ids, shared_attention_mask = self._padded_inputs([prefixes[0]])
                shared_outputs = self.model(
                    input_ids=shared_input_ids,
                    attention_mask=shared_attention_mask,
                    position_ids=self._position_ids(shared_attention_mask),
                    use_cache=True,
                    return_dict=True,
                )
                cache = self._repeat_cache(
                    getattr(shared_outputs, "past_key_values", None), len(requests)
                )
                if cache is not None:
                    attention_mask = shared_attention_mask.expand(len(requests), -1).clone()
                    logits = shared_outputs.logits[:, -1, :].expand(len(requests), -1)
                    prefill_tokens = len(prefixes[0])
                    shared_prefill_tokens_saved = (len(requests) - 1) * len(prefixes[0])
            if cache is None:
                input_ids, attention_mask = self._padded_inputs(prefixes)
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=self._position_ids(attention_mask),
                    use_cache=True,
                    return_dict=True,
                )
                logits = outputs.logits[:, -1, :]
                cache = getattr(outputs, "past_key_values", None)
            for step in range(maximum_new_tokens):
                permitted = torch_module.tensor(
                    [step < request.max_new_tokens for request in requests],
                    dtype=torch_module.bool,
                    device=self.device,
                )
                step_active = active & permitted
                if not bool(step_active.any()):
                    break
                log_probs = self._policy_log_probs(logits, sampling)
                probabilities = log_probs.exp()
                random_values = torch_module.tensor(
                    [
                        uniforms[index][step]
                        if step < len(uniforms[index])
                        else 0.0
                        for index in range(len(requests))
                    ],
                    dtype=probabilities.dtype,
                    device=self.device,
                )
                cumulative = probabilities.cumsum(dim=-1)
                sampled_tokens = (cumulative < random_values[:, None]).sum(dim=-1)
                sampled_tokens = sampled_tokens.clamp_max(probabilities.shape[-1] - 1)
                sampled_logprobs = log_probs.gather(-1, sampled_tokens[:, None]).squeeze(-1)

                sampled_cpu = sampled_tokens.detach().cpu().tolist()
                logprobs_cpu = sampled_logprobs.detach().cpu().tolist()
                for index, is_active in enumerate(step_active.detach().cpu().tolist()):
                    if not is_active:
                        continue
                    token = int(sampled_cpu[index])
                    token_lists[index].append(token)
                    logprob_lists[index].append(float(logprobs_cpu[index]))
                    if sampling.eos_token_id is not None and token == sampling.eos_token_id:
                        finish_reasons[index] = "eos"

                eos_finished = torch_module.tensor(
                    [
                        sampling.eos_token_id is not None
                        and int(sampled_cpu[index]) == sampling.eos_token_id
                        for index in range(len(requests))
                    ],
                    dtype=torch_module.bool,
                    device=self.device,
                )
                active_after = step_active & ~eos_finished
                active_after &= torch_module.tensor(
                    [step + 1 < request.max_new_tokens for request in requests],
                    dtype=torch_module.bool,
                    device=self.device,
                )
                if not bool(active_after.any()):
                    break
                next_tokens = torch_module.where(
                    step_active,
                    sampled_tokens,
                    torch_module.full_like(sampled_tokens, self.pad_token_id),
                )
                next_positions = attention_mask.sum(dim=-1, dtype=torch_module.long)
                attention_mask = torch_module.cat(
                    [attention_mask, step_active.to(dtype=attention_mask.dtype)[:, None]],
                    dim=-1,
                )
                outputs = self.model(
                    input_ids=next_tokens[:, None],
                    attention_mask=attention_mask,
                    position_ids=next_positions[:, None],
                    past_key_values=cache,
                    use_cache=True,
                    return_dict=True,
                )
                logits = outputs.logits[:, -1, :]
                cache = getattr(outputs, "past_key_values", None)
                active = active_after

        results: list[tuple[int, SequenceSample]] = []
        for (original_index, request), tokens, token_logprobs, finish_reason in zip(
            indexed_requests,
            token_lists,
            logprob_lists,
            finish_reasons,
            strict=True,
        ):
            results.append(
                (
                    original_index,
                    SequenceSample(
                        prefix=request.prefix,
                        token_ids=tuple(tokens),
                        token_logprobs=tuple(token_logprobs),
                        policy_id=request.sampling.policy_id,
                        model_id=self.model_id,
                        request_id=request.request_id,
                        finish_reason=finish_reason,
                    ),
                )
            )
        with self._statistics_lock:
            self._prefill_tokens += prefill_tokens
            self._shared_prefill_tokens_saved += shared_prefill_tokens_saved
        return results

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        if not requests:
            return []
        grouped: OrderedDict[
            SamplingConfig, list[tuple[int, GenerationRequest]]
        ] = OrderedDict()
        for index, request in enumerate(requests):
            grouped.setdefault(request.sampling, []).append((index, request))
        indexed_outputs: list[tuple[int, SequenceSample]] = []
        for group in grouped.values():
            indexed_outputs.extend(self._sample_same_policy(group))
        indexed_outputs.sort(key=lambda item: item[0])
        outputs = [sample for _, sample in indexed_outputs]
        with self._statistics_lock:
            self._sample_calls += 1
            self._sampled_sequences += len(outputs)
            self._generated_tokens += sum(len(output.token_ids) for output in outputs)
        return outputs

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        torch_module = _require_torch()
        flattened: list[tuple[ScoreRequest, TokenSequence]] = [
            (request, continuation)
            for request in requests
            for continuation in request.continuations
        ]
        results: list[tuple[float, ...]] = [()] * len(flattened)
        nonempty: list[tuple[int, ScoreRequest, TokenSequence, TokenSequence]] = []
        for index, (request, continuation) in enumerate(flattened):
            if continuation:
                nonempty.append(
                    (index, request, continuation, self._model_prefix(request.prefix))
                )
        if nonempty:
            sequences = [prefix + continuation for _, _, continuation, prefix in nonempty]
            input_ids, attention_mask = self._padded_inputs(sequences)
            with self._model_lock, torch_module.inference_mode():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=self._position_ids(attention_mask),
                    use_cache=False,
                    return_dict=True,
                )
            padded_length = input_ids.shape[1]
            for row, (flat_index, request, continuation, prefix) in enumerate(nonempty):
                padding = padded_length - len(prefix) - len(continuation)
                predictor_positions = torch_module.arange(
                    padding + len(prefix) - 1,
                    padding + len(prefix) + len(continuation) - 1,
                    device=self.device,
                )
                token_logits = outputs.logits[row].index_select(0, predictor_positions)
                log_probs = self._policy_log_probs(token_logits, request.sampling)
                targets = torch_module.tensor(
                    continuation, dtype=torch_module.long, device=self.device
                )
                selected = log_probs.gather(-1, targets[:, None]).squeeze(-1)
                results[flat_index] = tuple(float(value) for value in selected.cpu().tolist())
        with self._statistics_lock:
            self._score_calls += 1
            self._scored_tokens += sum(len(continuation) for _, continuation in flattened)
        return results

    def snapshot(self) -> TransformersBackendSnapshot:
        with self._statistics_lock:
            return TransformersBackendSnapshot(
                sample_calls=self._sample_calls,
                score_calls=self._score_calls,
                sampled_sequences=self._sampled_sequences,
                generated_tokens=self._generated_tokens,
                prefill_tokens=self._prefill_tokens,
                shared_prefill_tokens_saved=self._shared_prefill_tokens_saved,
                scored_tokens=self._scored_tokens,
            )

    def encode(self, text: str, *, add_special_tokens: bool = True) -> TokenSequence:
        return tuple(
            int(token)
            for token in self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        )

    def decode(self, tokens: TokenSequence, *, skip_special_tokens: bool = True) -> str:
        return str(
            self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
        )
