"""Batched causal-LM backend with exact policy probabilities and KV decoding."""

from __future__ import annotations

import inspect
import threading
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from inference_scaling.arllm.config import SamplingConfig, TokenPenalty
from inference_scaling.shared.compute import dense_forward_flops
from inference_scaling.shared.model.loading import model_identity
from inference_scaling.shared.rng import uniform_stream
from inference_scaling.arllm.backends.causal_scoring import run_causal_chunks
from inference_scaling.arllm.backends.kv_cache import DynamicCache, GrowingCache, PrefixStore, StaticDecoder, cache_layers
from inference_scaling.arllm.backends.replay import sample_with_drafts
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest, SequenceSample, TokenSequence

try:
    import torch
except ImportError:  # pragma: no cover - exercised in dependency-free installations
    torch = None  # type: ignore[assignment]


def _require_torch():
    if torch is None:
        raise ModuleNotFoundError("TransformersBackend requires the optional GPU dependencies; "
                                  "install the project's gpu extra first")
    return torch


@dataclass(frozen=True, slots=True)
class TransformersBackendSnapshot:
    sample_calls: int
    score_calls: int
    sampled_sequences: int
    generated_tokens: int
    # Draft tokens kept without a model call.
    replayed_tokens: int
    prefill_tokens: int
    shared_prefill_tokens_saved: int
    scored_tokens: int
    generation_forward_token_slots: int
    score_forward_token_slots: int
    estimated_dense_forward_flops: int


@dataclass(frozen=True, slots=True)
class SequenceScoreStatistics:
    """Per-token top-K confidence of one scored continuation (the Consilience statistic)."""

    token_topk_confidences: tuple[float, ...]


class TransformersBackend:
    """Manual batched decoding with request-local, scheduling-independent RNG."""

    def __init__(self, model: Any, tokenizer: Any, *, model_id: str | None = None, device: str | Any | None = None,
                 max_score_batch_size: int, score_chunk_size: int, token_penalty: TokenPenalty | None = None,
                 prefix_cache_bytes: int = 0, in_place_kv: bool = False, cuda_graphs: bool = False) -> None:
        torch_module = _require_torch()
        self.model = model
        self.tokenizer = tokenizer
        self._model_id = model_id or str(getattr(getattr(model, "config", None), "_name_or_path", "transformers-model"))
        self.device = torch_module.device(device or getattr(model, "device", None) or (
            "cuda" if torch_module.cuda.is_available() else "cpu"))
        # A token penalty makes the penalized model the model, so it is part of the identity.
        self._penalty = token_penalty
        if token_penalty is not None:
            self._model_id += f"|{token_penalty.penalty_id}"
            self._penalty_index = torch_module.tensor(token_penalty.token_ids, device=self.device)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id or eos_token_id")
        self.pad_token_id = int(pad_token_id)
        if max_score_batch_size <= 0:
            raise ValueError("max_score_batch_size must be positive")
        self.max_score_batch_size = int(max_score_batch_size)
        if score_chunk_size <= 0:
            raise ValueError("score_chunk_size must be positive")
        self.score_chunk_size = int(score_chunk_size)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        self.bos_token_id = None if bos_token_id is None else int(bos_token_id)
        self.model.eval()
        inspected_model = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
        forward_parameters = inspect.signature(inspected_model.forward).parameters
        self._supports_logits_to_keep = "logits_to_keep" in forward_parameters
        self._model_lock = threading.RLock()
        if prefix_cache_bytes < 0:
            raise ValueError("prefix_cache_bytes must be non-negative")
        # KV states of finished rows, which later requests resume over their longest stored prefix.
        self._store = PrefixStore(prefix_cache_bytes) if prefix_cache_bytes else None
        self._cache_class = GrowingCache if in_place_kv else DynamicCache
        # Decode steps at fixed shapes, captured as CUDA graphs on a CUDA device.
        if cuda_graphs and (len(set(getattr(model, "hf_device_map", {}).values())) > 1 or getattr(
                getattr(model, "config", None), "_attn_implementation", "sdpa") not in {"sdpa", "eager"}):
            raise ValueError("ar.engine.transformers.cuda_graphs needs the model on one device with sdpa or eager attention")
        self._static = StaticDecoder(model, self._supports_logits_to_keep, self.device.type == "cuda") if cuda_graphs else None
        self._statistics_lock = threading.Lock()
        for name in TransformersBackendSnapshot.__dataclass_fields__:
            setattr(self, "_" + name, 0)
        count_parameters = getattr(model, "num_parameters", None)
        self._parameter_count = int(count_parameters()) if callable(count_parameters) else sum(
            parameter.numel() for parameter in model.parameters())

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, *, adapter_name_or_path: str | None = None, device: str, dtype: str,
        cache_dir: str | None = None, revision: str | None = None, tokenizer_name_or_path: str | None = None,
        tokenizer_revision: str | None = None, adapter_revision: str | None = None,
        device_map: str | dict[str, Any] | None = None, attn_implementation: str | None = None,
        model_kwargs: Mapping[str, Any] | None = None, tokenizer_kwargs: Mapping[str, Any] | None = None,
        local_files_only: bool = False, trust_remote_code: bool = False, max_score_batch_size: int,
        score_chunk_size: int, token_penalty: Mapping[str, Any] | None, prefix_cache_mib: int, in_place_kv: bool,
        cuda_graphs: bool,
    ) -> "TransformersBackend":
        torch_module = _require_torch()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - depends on optional install
            raise ModuleNotFoundError("TransformersBackend.from_pretrained requires transformers") from error
        try:
            torch_dtype = "auto" if dtype == "auto" else getattr(torch_module, dtype)
        except AttributeError as error:
            raise ValueError(f"unknown torch dtype {dtype!r}") from error
        if dtype not in {"auto", "float32", "float16", "bfloat16", "float64"}:
            raise ValueError(f"unsupported floating-point dtype {dtype!r}")
        if torch_dtype != torch_module.float32:
            warnings.warn("Reduced-precision logits can depend noticeably on batch shape. Use dtype='float32' when "
                          "importance weights must match later rescoring.", RuntimeWarning, stacklevel=2)
        common = {"cache_dir": cache_dir, "local_files_only": local_files_only, "trust_remote_code": trust_remote_code}
        extra_model, extra_tokenizer = dict(model_kwargs or {}), dict(tokenizer_kwargs or {})
        reserved = {"dtype", "torch_dtype", "cache_dir", "revision", "local_files_only",
                    "trust_remote_code", "device_map", "attn_implementation", "token"}
        collision = reserved.intersection(extra_model.keys() | extra_tokenizer.keys())
        if collision:
            raise ValueError("loading kwargs duplicate explicit options or contain credentials: " + ", ".join(sorted(collision)))
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path or model_name_or_path, revision=(
            tokenizer_revision if tokenizer_revision is not None else (revision if tokenizer_name_or_path is None else None)),
            **common, **extra_tokenizer)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        model_load_kwargs = {**common, "revision": revision, **extra_model}
        if device_map is not None:
            model_load_kwargs["device_map"] = device_map
        if attn_implementation is not None:
            model_load_kwargs["attn_implementation"] = attn_implementation
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, dtype=torch_dtype, **model_load_kwargs)
        except TypeError as error:
            if "unexpected keyword argument 'dtype'" not in str(error):
                raise
            # Transformers 4.x names this argument ``torch_dtype``; 5.x uses ``dtype``.  Supporting both keeps the
            # AR backend usable from the dLLM-compatible test environment without changing model values.
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype, **model_load_kwargs)
        if adapter_name_or_path is not None:
            try:
                from peft import PeftModel
            except ImportError as error:  # pragma: no cover - optional training extra
                raise ModuleNotFoundError("Loading a GRPO adapter requires the project's training extra") from error
            model = PeftModel.from_pretrained(model, adapter_name_or_path, local_files_only=local_files_only,
                                              revision=adapter_revision, cache_dir=cache_dir)
        if device_map is None and not getattr(model, "is_quantized", False):
            model.to(torch_module.device(device))
        input_device = device
        if device_map is not None:
            input_device = model.get_input_embeddings().weight.device
            if str(input_device) == "meta":
                raise ValueError("input embeddings require a concrete device for manual decoding")
        return cls(model, tokenizer, model_id=model_identity(
            model_name_or_path, adapter_name_or_path, revision=revision, adapter_revision=adapter_revision,
            tokenizer=tokenizer_name_or_path, tokenizer_revision=tokenizer_revision), device=input_device,
            max_score_batch_size=max_score_batch_size, score_chunk_size=score_chunk_size,
            token_penalty=None if token_penalty is None else TokenPenalty.from_words(
                tokenizer, token_penalty["words"], token_penalty["strength"]),
            prefix_cache_bytes=int(prefix_cache_mib) * 2**20, in_place_kv=in_place_kv, cuda_graphs=cuda_graphs)

    @property
    def model_id(self) -> str:
        return self._model_id

    def close(self) -> None:
        """Release the model reference after all dispatchers have stopped."""
        with self._model_lock:
            self.model = self._static = None
            if self._store is not None:
                self._store.clear()

    @property
    def parameter_count(self) -> int:
        """Number of model parameters used by the dominant-matmul FLOP estimate."""

        return self._parameter_count

    def _dense_forward_flops(self, token_slots: int) -> int:
        """The dominant dense matmuls, ``2 * parameters * tokens``; attention's length term and elementwise work are excluded."""

        return dense_forward_flops(self._parameter_count, token_slots)

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

    def _policy_log_probs(self, logits, sampling: SamplingConfig | None):
        """Log-probabilities of ``sampling`` over the (penalized) model's logits; sampling and scoring share them."""

        torch_module = _require_torch()
        policy = sampling or SamplingConfig()
        transformed = logits.to(dtype=torch_module.float32) / policy.temperature
        if self._penalty is not None:
            # The penalty belongs to the model's logits, so the temperature scales it too.
            transformed[..., self._penalty_index.to(transformed.device)] -= self._penalty.strength / policy.temperature
        if policy.top_k is not None and policy.top_k < transformed.shape[-1]:
            threshold = torch_module.topk(transformed, policy.top_k, dim=-1).values[..., -1, None]
            transformed = transformed.masked_fill(transformed < threshold, float("-inf"))
        if policy.top_p < 1:
            sorted_logits, sorted_indices = torch_module.sort(transformed, descending=True, dim=-1)
            remove = torch_module.softmax(sorted_logits, dim=-1).cumsum(dim=-1) > policy.top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            transformed = transformed.masked_fill(torch_module.zeros_like(remove).scatter(-1, sorted_indices, remove),
                                                  float("-inf"))
        return torch_module.log_softmax(transformed, dim=-1)

    def _padded_inputs(self, prefixes: Sequence[TokenSequence]):
        torch_module = _require_torch()
        maximum = max(len(prefix) for prefix in prefixes)
        input_ids = torch_module.full((len(prefixes), maximum), self.pad_token_id, dtype=torch_module.long,
                                      device=self.device)
        attention_mask = torch_module.zeros_like(input_ids)
        for index, prefix in enumerate(prefixes):
            input_ids[index, maximum - len(prefix) :] = torch_module.tensor(prefix, dtype=torch_module.long,
                                                                            device=self.device)
            attention_mask[index, maximum - len(prefix) :] = 1
        return input_ids, attention_mask

    @staticmethod
    def _select_rows(cache, rows):
        """Keep, reorder or repeat batch rows of a KV cache."""

        select = getattr(cache, "batch_select_indices", None)
        if callable(select):
            select(rows)
            return cache
        return tuple(tuple(tensor.index_select(0, rows) for tensor in layer) for layer in cache)

    def _sample_same_policy(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        """Decode requests that share their policy, reference policy and stop sequences.

        Each distinct prefix resumes the stored KV state of its longest stored
        prefix, is prefilled once from there and its KV state copied to every row
        that repeats it; a row that ends leaves the batch and its KV state goes to
        the store. Tokens are drawn by inverse CDF from request-local uniforms, so a
        sample does not depend on the batch it runs in. Sampled values stay on the
        device until the batch is done.
        """

        torch_module = _require_torch()
        first = requests[0]
        sampling, reference_sampling = first.sampling, first.reference_policy
        prefixes = [self._model_prefix(request.prefix) for request in requests]
        positions = {prefix: index for index, prefix in enumerate(dict.fromkeys(prefixes))}
        unique = list(positions)
        steps = max(request.max_new_tokens for request in requests)
        uniforms = np.zeros((len(requests), steps))
        for row, request in enumerate(requests):
            uniforms[row, : request.max_new_tokens] = uniform_stream(request.seed, request.uniform_offset,
                                                                     request.max_new_tokens)
        device_uniforms = torch_module.from_numpy(uniforms).to(self.device)
        limits = torch_module.tensor([request.max_new_tokens for request in requests], device=self.device)
        markers = [torch_module.tensor(stop, device=self.device) for stop in first.stop_sequences]
        tokens = torch_module.zeros((len(requests), steps), dtype=torch_module.long, device=self.device)
        logprobs = torch_module.zeros((len(requests), steps), dtype=torch_module.float32, device=self.device)
        references = torch_module.zeros_like(logprobs)
        bounds = torch_module.zeros((len(requests), steps, 2), dtype=torch_module.float64, device=self.device)
        lengths = torch_module.zeros(len(requests), dtype=torch_module.long, device=self.device)
        # Why each row ended: 0 at its token limit, 1 at EOS, 2 after a stop sequence, 3 rejected by its log-weight.
        reasons = torch_module.zeros_like(lengths)
        stops = [request.log_weight_stop for request in requests]
        # Scale, current weight, threshold and running log-weight of each row; -inf never rejects.
        weights = torch_module.tensor([(stop.scale, stop.current, stop.threshold, stop.start) if stop else
                                       (0.0, 0.0, float("-inf"), 0.0) for stop in stops],
                                      dtype=torch_module.float64, device=self.device)
        with self._model_lock, torch_module.inference_mode():
            matches = [self._store.match(prefix) if self._store else (0, []) for prefix in unique]
            # The stored positions of each prefix, right-aligned; its last position is fed again for its logits.
            cached = [min(length, len(prefix) - 1) for (length, _), prefix in zip(matches, unique, strict=True)]
            width = max(cached)
            layers = []
            for layer, (key, value) in enumerate(next((entry for (_, entry), length in zip(matches, cached) if length), [])):
                keys, values = key.new_zeros((len(unique), *key.shape[1:2], width, key.shape[-1])), value.new_zeros(
                    (len(unique), *value.shape[1:2], width, value.shape[-1]))
                for row, ((_, entry), length) in enumerate(zip(matches, cached, strict=True)):
                    if length:
                        keys[row, :, width - length :] = entry[layer][0][0, :, :length]
                        values[row, :, width - length :] = entry[layer][1][0, :, :length]
                layers.append((keys, values))
            input_ids, attention_mask = self._padded_inputs([prefix[length:] for prefix, length in zip(unique, cached)])
            stored = torch_module.zeros((len(unique), width), dtype=torch_module.long, device=self.device)
            for row, length in enumerate(cached):
                stored[row, width - length :] = 1
            attention_mask = torch_module.cat([stored, attention_mask], dim=-1)
            last: list[Any] = []
            cache = run_causal_chunks(
                self.model, input_ids, attention_mask, self._position_ids(attention_mask)[:, width:],
                chunk_size=self.score_chunk_size, first_needed=input_ids.shape[1] - 1,
                supports_logits_to_keep=self._supports_logits_to_keep,
                cache=self._cache_class(layers) if layers else self._cache_class(),
                on_logits=lambda _position, logits: last.append(logits[:, -1, :]),
            )
            slots = int(input_ids.numel())
            fan_out = torch_module.tensor([positions[prefix] for prefix in prefixes], device=self.device)
            logits = last[-1].index_select(0, fan_out)
            attention_mask = attention_mask.index_select(0, fan_out)
            static = self._static
            if static is not None:
                static.load(cache_layers(cache), fan_out, attention_mask, attention_mask.shape[1] + steps)
            elif len(unique) < len(requests):
                cache = self._select_rows(cache, fan_out)
            alive = torch_module.arange(len(requests), device=self.device)
            for step in range(steps):
                log_probs = self._policy_log_probs(logits, sampling)
                reference_log_probs = (log_probs if sampling == reference_sampling
                                       else self._policy_log_probs(logits, reference_sampling))
                # Inverse-CDF sampling is especially sensitive to accumulated
                # roundoff over a language model's large vocabulary.  A
                # float32 CDF can move a fixed request-local uniform across a
                # token boundary when otherwise equivalent requests are
                # decoded in different batch shapes.  Accumulating the same
                # policy probabilities in float64 preserves the categorical
                # policy while making request-local seeds robust to scheduling.
                cumulative = log_probs.exp().to(dtype=torch_module.float64).cumsum(dim=-1)
                cumulative[:, -1] = 1.0
                sampled = (cumulative < device_uniforms[alive, step, None]).sum(dim=-1).clamp_max(log_probs.shape[-1] - 1)
                tokens[alive, step] = sampled
                below = cumulative.gather(-1, (sampled - 1).clamp_min(0)[:, None]).squeeze(-1)
                bounds[alive, step] = torch_module.stack(
                    (below.masked_fill(sampled == 0, -1.0), cumulative.gather(-1, sampled[:, None]).squeeze(-1)), dim=-1)
                logprobs[alive, step] = log_probs.gather(-1, sampled[:, None]).squeeze(-1)
                references[alive, step] = reference_log_probs.gather(-1, sampled[:, None]).squeeze(-1)
                lengths[alive] = step + 1
                reason = torch_module.zeros_like(sampled)
                for marker in markers:
                    if step + 1 >= len(marker):
                        reason = reason.masked_fill(
                            (tokens[alive, step + 1 - len(marker) : step + 1] == marker).all(dim=-1), 2)
                if sampling.eos_token_id is not None:
                    reason = reason.masked_fill(sampled == sampling.eos_token_id, 1)
                if any(stops):
                    row = weights[alive]
                    row[:, 3] += (row[:, 0] * references[alive, step].double() - logprobs[alive, step].double()).clamp(max=0.0)
                    weights[alive] = row
                    reason = reason.masked_fill((reason == 0) & (row[:, 3] - row[:, 1] < row[:, 2]), 3)
                reasons[alive] = reason
                keep = (reason == 0) & (step + 1 < limits[alive])
                remaining = int(keep.sum())
                if self._store is not None and remaining < len(alive):
                    # A row that ends is stored with the tokens fed so far, all but its last.
                    kv = cache_layers(cache) if static is None else static.layers(len(alive), attention_mask.shape[1])
                    for position in (~keep).nonzero().squeeze(-1).tolist():
                        row, valid = int(alive[position]), attention_mask[position].bool()
                        self._store.add(prefixes[row] + tuple(tokens[row, :step].tolist()),
                                        [(key[position : position + 1, :, valid], value[position : position + 1, :, valid])
                                         for key, value in kv])
                if not remaining:
                    break
                if remaining < len(alive):
                    # Finished rows leave the batch.
                    kept = keep.nonzero().squeeze(-1)
                    alive, sampled, attention_mask = alive[kept], sampled[kept], attention_mask[kept]
                    if static is None:
                        cache = self._select_rows(cache, kept)
                    else:
                        static.select(kept, attention_mask.shape[1])
                next_positions = attention_mask.sum(dim=-1, dtype=torch_module.long)
                attention_mask = torch_module.cat([attention_mask, attention_mask.new_ones((len(alive), 1))], dim=-1)
                slots += len(alive)
                if static is not None:
                    logits = static.step(sampled, next_positions, attention_mask.shape[1] - 1)
                    continue
                outputs = self.model(input_ids=sampled[:, None], attention_mask=attention_mask,
                                     position_ids=next_positions[:, None], past_key_values=cache, use_cache=True,
                                     return_dict=True, **({"logits_to_keep": 1} if self._supports_logits_to_keep else {}))
                logits, cache = outputs.logits[:, -1, :], getattr(outputs, "past_key_values", None)
        rows = zip(requests, lengths.tolist(), reasons.tolist(), tokens.cpu().numpy(), logprobs.cpu().numpy(),
                   references.cpu().numpy(), bounds.cpu().numpy(), strict=True)
        samples = [SequenceSample(
            prefix=request.prefix, token_ids=tuple(row_tokens[:length].tolist()),
            token_logprobs=tuple(row_logprobs[:length].tolist()), policy_id=request.sampling.policy_id,
            model_id=self.model_id, request_id=request.request_id, finish_reason=("length", "eos", "stop", "rejected")[reason],
            reference_token_logprobs=tuple(row_references[:length].tolist()), reference_policy_id=reference_sampling.policy_id,
            token_cdf_bounds=tuple(map(tuple, row_bounds[:length].tolist())),
        ) for request, length, reason, row_tokens, row_logprobs, row_references, row_bounds in rows]
        prefill_tokens = sum(len(prefix) - length for prefix, length in zip(unique, cached, strict=True))
        with self._statistics_lock:
            self._prefill_tokens += prefill_tokens
            self._shared_prefill_tokens_saved += sum(map(len, prefixes)) - prefill_tokens
            self._generation_forward_token_slots += slots
            self._estimated_dense_forward_flops += self._dense_forward_flops(slots)
        return samples

    def _generate(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        grouped: dict[tuple[SamplingConfig, tuple[TokenSequence, ...], float], list[int]] = {}
        for index, request in enumerate(requests):
            grouped.setdefault((request.sampling, request.stop_sequences, request.reference_temperature), []).append(index)
        results: dict[int, SequenceSample] = {}
        for indices in grouped.values():
            results.update(zip(indices, self._sample_same_policy([requests[index] for index in indices]), strict=True))
        return [results[index] for index in range(len(requests))]

    def sample_batch(self, requests: Sequence[GenerationRequest]) -> list[SequenceSample]:
        if not requests:
            return []
        outputs, replayed = sample_with_drafts(requests, self._generate, model_id=self.model_id, honors_stops=True)
        with self._statistics_lock:
            self._sample_calls += 1
            self._sampled_sequences += len(outputs)
            self._generated_tokens += sum(len(output.token_ids) for output in outputs) - replayed
            self._replayed_tokens += replayed
        return outputs

    def _token_scores(self, logits, sampling, continuation: TokenSequence, confidence_top_k: int | None):
        """Log-probabilities of ``continuation`` and, when asked, its top-K confidences."""

        torch_module = _require_torch()
        log_probs = self._policy_log_probs(logits, sampling)
        targets = torch_module.tensor(continuation, dtype=torch_module.long, device=log_probs.device)
        selected = log_probs.gather(-1, targets[:, None]).squeeze(-1)
        confidences: list[float] = []
        if confidence_top_k is not None:
            top = log_probs.topk(min(confidence_top_k, log_probs.shape[-1]), dim=-1).values
            confidences = (-top.mean(dim=-1)).cpu().tolist()
        return [float(value) for value in selected.cpu().tolist()], [float(value) for value in confidences]

    def _score_batch(self, batch, confidence_top_k: int | None) -> tuple[list[tuple[list[float], list[float]]], int]:
        """Teacher-force one batch of (request, continuation, inputs) rows in chunks; returns scores and slots."""

        torch_module = _require_torch()
        input_ids, attention_mask = self._padded_inputs([inputs for _, _, inputs in batch])
        width = input_ids.shape[1]
        scores: list[tuple[list[float], list[float]]] = [([], []) for _ in batch]

        def consume(position: int, logits: Any) -> None:
            # Rows are left-padded: a continuation of n tokens is predicted by the last n positions.
            for row, (request, continuation, _) in enumerate(batch):
                first = width - len(continuation)
                low = max(position, first)
                if low < position + logits.shape[1]:
                    logs, confidences = self._token_scores(
                        logits[row, low - position :], request.sampling,
                        continuation[low - first : position + logits.shape[1] - first], confidence_top_k,
                    )
                    scores[row][0].extend(logs)
                    scores[row][1].extend(confidences)

        with self._model_lock, torch_module.inference_mode():
            run_causal_chunks(
                self.model, input_ids, attention_mask, self._position_ids(attention_mask),
                chunk_size=self.score_chunk_size, first_needed=width - max(len(item[1]) for item in batch),
                supports_logits_to_keep=self._supports_logits_to_keep, on_logits=consume, cache=self._cache_class(),
            )
        return scores, int(input_ids.numel())

    def _score_rows(self, requests: Sequence[ScoreRequest],
                    confidence_top_k: int | None) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
        """Token log-probabilities and (optionally) top-K confidences of every continuation.

        Continuations are teacher-forced in batches of similar length and in
        chunks of ``score_chunk_size`` positions over a KV cache, so a long
        sequence keeps its complete context. A batch holds at most
        ``max_score_batch_size`` rows and as many padded positions as that many
        chunks, which bounds its logits and KV state.
        """

        flattened = [(request, continuation) for request in requests for continuation in request.continuations]
        results: list[tuple[tuple[float, ...], tuple[float, ...]]] = [((), ())] * len(flattened)
        items = sorted(
            ((index, request, continuation, self._model_prefix(request.prefix) + tuple(continuation[:-1]))
             for index, (request, continuation) in enumerate(flattened) if continuation),
            key=lambda item: len(item[3]),
        )
        forwarded, capacity = 0, self.max_score_batch_size * self.score_chunk_size
        while items:
            size = 1
            while size < min(len(items), self.max_score_batch_size) and (size + 1) * len(items[size][3]) <= capacity:
                size += 1
            batch, items = items[:size], items[size:]
            scores, slots = self._score_batch([item[1:] for item in batch], confidence_top_k)
            forwarded += slots
            for (index, *_), (logs, confidences) in zip(batch, scores, strict=True):
                results[index] = (tuple(logs), tuple(confidences))
        with self._statistics_lock:
            self._score_calls += 1
            self._scored_tokens += sum(len(continuation) for _, continuation in flattened)
            self._score_forward_token_slots += forwarded
            self._estimated_dense_forward_flops += self._dense_forward_flops(forwarded)
        return results

    def score_batch(self, requests: Sequence[ScoreRequest]) -> list[tuple[float, ...]]:
        return [logs for logs, _ in self._score_rows(requests, None)]

    def score_statistics_batch(self, requests: Sequence[ScoreRequest], *,
                               confidence_top_k: int) -> list[SequenceScoreStatistics]:
        """Top-K confidence trajectories of the continuations.

        A truncated policy gives some of the top-K candidates probability zero,
        so only a full-support policy is accepted. The forward passes count as
        ordinary scoring.
        """

        if confidence_top_k <= 0:
            raise ValueError("confidence_top_k must be positive")
        for request in requests:
            policy = request.sampling or SamplingConfig()
            if policy.top_p < 1 or policy.top_k is not None:
                raise ValueError("top-K confidence requires a full-support policy")
            if not all(request.continuations):
                raise ValueError("confidence rewards require nonempty continuations")
        return [SequenceScoreStatistics(confidences) for _, confidences in self._score_rows(requests, confidence_top_k)]

    def snapshot(self) -> TransformersBackendSnapshot:
        with self._statistics_lock:
            return TransformersBackendSnapshot(**{name: getattr(self, "_" + name)
                                                  for name in TransformersBackendSnapshot.__dataclass_fields__})

    def encode(self, text: str, *, add_special_tokens: bool = True) -> TokenSequence:
        return tuple(int(token) for token in self.tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def decode(self, tokens: TokenSequence, *, skip_special_tokens: bool = True) -> str:
        return str(self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens))

    def direct_generate(self, prefix: TokenSequence, *, max_new_tokens: int, num_beams: int = 1) -> TokenSequence:
        """Greedy/beam baseline using Transformers' native generation path."""

        if max_new_tokens <= 0 or num_beams <= 0:
            raise ValueError("generation length and beam count must be positive")
        torch_module = _require_torch()
        input_ids = torch_module.tensor([prefix], dtype=torch_module.long, device=self.device)
        attention_mask = torch_module.ones_like(input_ids)
        # Beam search adds up the processed scores, so they are the penalized model's normalized log-probabilities.
        penalized = None if self._penalty is None else [lambda _ids, scores: self._policy_log_probs(scores, None)]
        with self._model_lock, torch_module.inference_mode():
            output = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=False,
                num_beams=num_beams, use_cache=True, pad_token_id=self.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id, logits_processor=penalized)
        return tuple(int(token) for token in output[0, input_ids.shape[1] :].tolist())
