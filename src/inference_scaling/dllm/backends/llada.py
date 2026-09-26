"""Batched Transformers backend for LLaDA-style masked diffusion models."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from math import lgamma
from threading import Lock
from time import perf_counter
from typing import Any

from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.types import (
    DiffusionGenerationRequest,
    DiffusionSample,
    DiffusionTraceStep,
)


@dataclass(frozen=True, slots=True)
class LLaDABackendSnapshot:
    sample_requests: int
    forward_calls: int
    model_sequences: int
    model_token_slots: int
    generated_tokens: int
    elapsed_seconds: float
    total_parameters: int
    active_parameters: int
    resident_parameters: int


def active_parameter_counts(model: Any) -> tuple[int, int]:
    """Total parameters and those active per token: routed experts count by their routing share."""

    config = getattr(model, "config", None)
    experts = int(getattr(config, "num_experts", 0) or 0)
    active_experts = int(getattr(config, "num_experts_per_tok", 0) or 0)
    share = active_experts / experts if 0 < active_experts <= experts else 1.0
    total, active = 0, 0.0
    for name, parameter in model.named_parameters():
        total += int(parameter.numel())
        active += parameter.numel() * (share if ".experts." in name else 1.0)
    return total, int(round(active))


class LLaDATransformersBackend:
    """Execute blockwise masked diffusion and record committed trajectories.

    The implementation follows the public LLaDA sampler: all masked positions
    are predicted in parallel and each reverse step commits a fixed number of
    positions.  ``low_confidence`` commits the most confident predictions.
    ``random`` commits a uniform subset independent of sampled token values, so
    its trajectories have tractable probabilities.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        mask_token_id: int,
        max_batch_size: int,
        model_id: str | None = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - exercised without GPU extras
            raise RuntimeError("LLaDA backend requires PyTorch") from exc

        self.model = model.eval()
        self.tokenizer = tokenizer
        self._torch = torch
        configured_name = getattr(getattr(model, "config", None), "_name_or_path", None)
        self._model_id = model_id or configured_name or model.__class__.__name__
        if mask_token_id < 0 or max_batch_size <= 0:
            raise ValueError("mask_token_id must be non-negative and max_batch_size positive")
        self._mask_token_id = mask_token_id
        self._eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self._device = self._infer_device()
        self._max_batch_size = max_batch_size
        self._resident_parameters = int(sum(parameter.numel() for parameter in model.parameters()))
        self._total_parameters, self._active_parameters = active_parameter_counts(model)
        self._lock = Lock()
        self._sample_requests = self._forward_calls = self._model_sequences = self._model_token_slots = 0
        self._generated_tokens, self._elapsed_seconds = 0, 0.0

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device: str,
        dtype: str,
        trust_remote_code: bool,
        mask_token_id: int,
        max_batch_size: int,
        **model_kwargs: Any,
    ) -> "LLaDATransformersBackend":
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise RuntimeError("install the dllm optional dependency set") from exc

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        if dtype not in dtype_map:
            raise ValueError(f"unsupported dtype {dtype!r}")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code,
                                          torch_dtype=dtype_map[dtype], low_cpu_mem_usage=True, **model_kwargs).to(device)
        return cls(model, tokenizer, model_id=model_name_or_path, mask_token_id=mask_token_id,
                   max_batch_size=max_batch_size)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id

    def _request_chunks(self, group: Sequence[Any]) -> Sequence[Sequence[Any]]:
        return tuple(group[start : start + self._max_batch_size] for start in range(0, len(group), self._max_batch_size))

    def encode_chat(self, user_text: str, *, system_text: str | None) -> tuple[int, ...]:
        messages = ([{"role": "system", "content": system_text}] if system_text is not None else []) + [
            {"role": "user", "content": user_text},
        ]
        encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        return tuple(int(token_id) for token_id in encoded)

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return str(self.tokenizer.decode(list(token_ids), skip_special_tokens=skip_special_tokens))

    def snapshot(self) -> LLaDABackendSnapshot:
        with self._lock:
            return LLaDABackendSnapshot(**{name: getattr(self, "_" + name)
                                           for name in LLaDABackendSnapshot.__dataclass_fields__})

    def sample_batch(
        self, requests: Sequence[DiffusionGenerationRequest]
    ) -> list[DiffusionSample]:
        if not requests:
            return []
        started = perf_counter()
        groups: dict[
            tuple[int, int, DiffusionSamplingConfig, float | None, bool], list[tuple[int, DiffusionGenerationRequest]]
        ] = defaultdict(list)
        for index, request in enumerate(requests):
            key = (len(request.prefix), request.generation_length, request.sampling, request.reference_temperature,
                   request.stop_at_eos)
            groups[key].append((index, request))
        outputs: list[DiffusionSample | None] = [None] * len(requests)
        with self._torch.inference_mode():
            for group in groups.values():
                for chunk in self._request_chunks(group):
                    indices = [item[0] for item in chunk]
                    samples = self._sample_group([item[1] for item in chunk])
                    for index, sample in zip(indices, samples, strict=True):
                        outputs[index] = sample
        elapsed = perf_counter() - started
        if any(output is None for output in outputs):
            raise RuntimeError("internal dLLM batch reordering failure")
        samples = [output for output in outputs if output is not None]
        with self._lock:
            self._sample_requests += len(requests)
            self._generated_tokens += sum(len(sample.token_ids) for sample in samples)
            self._elapsed_seconds += elapsed
        return samples

    def _infer_device(self) -> Any:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self._torch.device("cpu")

    def _record_forward(self, batch_size: int, sequence_length: int) -> None:
        with self._lock:
            self._forward_calls += 1
            self._model_sequences += batch_size
            self._model_token_slots += batch_size * sequence_length

    def _block_logits(self, tokens: Any, *, prompt_length: int, sampling: DiffusionSamplingConfig,
                      start: int, end: int) -> Any:
        """Policy logits of positions ``start:end``; the model still reads the whole canvas."""

        if sampling.cfg_scale > 0:
            unconditional = tokens.clone()
            unconditional[:, :prompt_length] = self._mask_token_id
            model_input = self._torch.cat((tokens, unconditional), dim=0)
            both = self.model(model_input).logits[:, start:end].float()
            self._record_forward(model_input.shape[0], model_input.shape[1])
            logits, unconditional_logits = self._torch.chunk(both, 2, dim=0)
            logits = unconditional_logits + (sampling.cfg_scale + 1.0) * (logits - unconditional_logits)
        else:
            logits = self.model(tokens).logits[:, start:end].to(self._torch.float32, copy=True)
            self._record_forward(tokens.shape[0], tokens.shape[1])
        # A committed mask would leave the state unchanged and violate the fixed
        # transfer schedule.  The reverse policy is therefore normalized over
        # ordinary vocabulary tokens only.
        logits[..., self._mask_token_id] = -self._torch.inf
        return self._filter_logits(logits, sampling)

    def _filter_logits(self, logits: Any, sampling: DiffusionSamplingConfig) -> Any:
        """Apply the normalized top-k/top-p policy used for sampling and scoring."""

        if sampling.top_k > 0:
            kept = min(sampling.top_k, logits.shape[-1])
            threshold = self._torch.topk(logits, kept, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, -self._torch.inf)
        if sampling.top_p < 1:
            sorted_logits, sorted_indices = self._torch.sort(logits, descending=True, dim=-1)
            sorted_remove = self._torch.cumsum(self._torch.softmax(sorted_logits.float(), dim=-1), dim=-1) > sampling.top_p
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove = self._torch.zeros_like(sorted_remove).scatter(-1, sorted_indices, sorted_remove)
            logits = logits.masked_fill(remove, -self._torch.inf)
        return logits

    def _logprob_at(self, logits: Any, tokens: Any) -> Any:
        return self._torch.log_softmax(logits, dim=-1).gather(-1, tokens.unsqueeze(-1)).squeeze(-1)

    def _draw_tokens(self, logits: Any, *, temperature: float, generators: Sequence[Any]) -> tuple[Any, Any | None]:
        if temperature == 0:
            return self._torch.argmax(logits, dim=-1), None
        scaled = logits / temperature
        # Gumbel-max noise from each row's own generator keeps a sample independent of its batch.
        noise = self._torch.stack([self._torch.empty_like(row).exponential_(1.0, generator=generator)
                                   for row, generator in zip(scaled, generators, strict=True)])
        sampled = self._torch.argmax(scaled - noise.log(), dim=-1)
        return sampled, self._logprob_at(scaled, sampled)

    @staticmethod
    def _subset_logprob(available: int, selected: int) -> float:
        if not 0 <= selected <= available:
            raise ValueError("invalid subset size")
        return -(lgamma(available + 1) - lgamma(selected + 1) - lgamma(available - selected + 1))

    @staticmethod
    def _transfer_schedule(block_length: int, steps: int) -> tuple[int, ...]:
        quotient, remainder = divmod(block_length, steps)
        return tuple(quotient + (index < remainder) for index in range(steps))

    def _sample_group(self, requests: Sequence[DiffusionGenerationRequest]) -> list[DiffusionSample]:
        torch = self._torch
        first = requests[0]
        prompt_length, generation_length, sampling = len(first.prefix), first.generation_length, first.sampling
        reference_temperature = first.reference_temperature
        exact = sampling.has_exact_trajectory_density
        if reference_temperature is not None and not exact:
            raise ValueError("a reference trajectory probability requires an exact diffusion policy")
        eos = self._eos_token_id if first.stop_at_eos else None
        tokens = torch.full((len(requests), prompt_length + generation_length), self._mask_token_id,
                            dtype=torch.long, device=self._device)
        tokens[:, :prompt_length] = torch.tensor([request.prefix for request in requests], device=self._device)
        generators = [torch.Generator(device=self._device).manual_seed(request.seed) for request in requests]
        # Original request index of every row still being denoised.
        rows = list(range(len(requests)))
        traces: list[list[DiffusionTraceStep]] = [[] for _ in requests]
        logprobs, references = [0.0] * len(requests), [0.0] * len(requests)
        outputs: list[tuple[int, ...]] = [() for _ in requests]
        finish = ["length"] * len(requests)
        schedule = self._transfer_schedule(sampling.block_length, sampling.steps_per_block)
        for block_index in range(generation_length // sampling.block_length):
            start = prompt_length + block_index * sampling.block_length
            end, available = start + sampling.block_length, sampling.block_length
            for step_index, count in enumerate(schedule):
                logits = self._block_logits(tokens, prompt_length=prompt_length, sampling=sampling, start=start,
                                            end=end)
                sampled, token_logprobs = self._draw_tokens(logits, temperature=sampling.temperature,
                                                            generators=generators)
                if sampling.remasking == "random":
                    priorities = torch.stack([torch.rand(sampling.block_length, device=self._device, generator=g)
                                              for g in generators])
                else:
                    priorities = self._logprob_at(logits, sampled)
                # Every row has the same number of masked positions; commit ``count`` of them.
                priorities = priorities.masked_fill(tokens[:, start:end] != self._mask_token_id, -torch.inf)
                selected = priorities.topk(count, dim=-1).indices.sort(dim=-1).values
                chosen = sampled.gather(-1, selected)
                tokens[:, start:end].scatter_(-1, selected, chosen)
                step_logprobs = reference_logprobs = None
                if exact:
                    assert token_logprobs is not None
                    # Exact policies remask a uniform subset of the masked positions.
                    subset = self._subset_logprob(available, count)
                    step_logprobs = (subset + token_logprobs.gather(-1, selected).sum(-1)).tolist()
                    if reference_temperature is not None:
                        reference = self._logprob_at(logits / reference_temperature, sampled)
                        reference_logprobs = (subset + reference.gather(-1, selected).sum(-1)).tolist()
                available -= count
                positions, values = (selected + (start - prompt_length)).tolist(), chosen.tolist()
                for row, original in enumerate(rows):
                    step = None if step_logprobs is None else float(step_logprobs[row])
                    if step is not None:
                        logprobs[original] += step
                    if reference_logprobs is not None:
                        references[original] += float(reference_logprobs[row])
                    traces[original].append(DiffusionTraceStep(block_index, step_index, tuple(positions[row]),
                                                               tuple(values[row]), step))
            if eos is not None:
                ended = (tokens[:, start:end] == eos).all(dim=-1).tolist()
                if any(ended):
                    for original, done, values in zip(rows, ended, tokens[:, prompt_length:end].tolist(), strict=True):
                        if done:
                            outputs[original], finish[original] = tuple(values), "eos"
                    keep = [row for row, done in enumerate(ended) if not done]
                    tokens, rows, generators = tokens[keep], [rows[row] for row in keep], [generators[row] for row in keep]
                    if not rows:
                        break
        for original, values in zip(rows, tokens[:, prompt_length:].tolist(), strict=True):
            outputs[original] = tuple(values)
        return [
            DiffusionSample(
                prefix=request.prefix, token_ids=outputs[index], trace=tuple(traces[index]),
                trajectory_logprob=logprobs[index] if exact else None, policy_id=sampling.policy_id,
                model_id=self.model_id, request_id=request.request_id, finish_reason=finish[index],
                reference_trajectory_logprob=None if reference_temperature is None else references[index],
            )
            for index, request in enumerate(requests)
        ]


__all__ = ["LLaDABackendSnapshot", "LLaDATransformersBackend", "active_parameter_counts"]
