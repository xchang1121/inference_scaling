"""Transformers backend for SDAR block-diffusion language models."""

from __future__ import annotations

from collections.abc import Sequence
import importlib.util
from importlib.machinery import ModuleSpec
import sys
from types import ModuleType
from typing import Any, Literal

from inference_scaling.dllm.backends.llada import LLaDATransformersBackend
from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.types import (
    DiffusionGenerationRequest,
    DiffusionSample,
    DiffusionTraceStep,
    DiffusionTrajectoryScoreRequest,
)


class SDARTransformersBackend(LLaDATransformersBackend):
    """Run SDAR with block-causal KV reuse and auditable reverse trajectories.

    SDAR differs from a full-sequence masked dLLM in two places: completed
    blocks are autoregressive context for later blocks, and only the current
    block is repeatedly denoised.  The implementation follows the official
    sampler while adding batching, exact-length output, per-request generators,
    compute accounting, and trajectory scoring for random/sequential remasking.
    """

    @staticmethod
    def install_portable_rms_norm() -> bool:
        """Provide the one FlashAttention symbol imported unconditionally upstream.

        The official SDAR modeling file uses SDPA for attention when requested,
        but imports FlashAttention's Triton RMSNorm at module import time.  That
        package has no supported Windows build.  This narrowly scoped fallback
        supplies the same RMSNorm operation with PyTorch and leaves the actual
        attention implementation set to SDPA.  ``True`` means the fallback was
        installed; ``False`` means a real ``flash_attn`` package is available.
        """

        if "flash_attn" in sys.modules or importlib.util.find_spec("flash_attn") is not None:
            return False
        import torch

        root = ModuleType("flash_attn")
        ops = ModuleType("flash_attn.ops")
        triton = ModuleType("flash_attn.ops.triton")
        layer_norm = ModuleType("flash_attn.ops.triton.layer_norm")
        bert_padding = ModuleType("flash_attn.bert_padding")
        for module in (root, ops, triton, layer_norm, bert_padding):
            module.__spec__ = ModuleSpec(module.__name__, loader=None)

        def rms_norm_fn(
            hidden_states: Any,
            *,
            weight: Any,
            bias: Any | None = None,
            eps: float = 1e-6,
            **_: Any,
        ) -> Any:
            input_dtype = hidden_states.dtype
            normalized = hidden_states.float()
            normalized = normalized * torch.rsqrt(
                normalized.square().mean(dim=-1, keepdim=True) + eps
            )
            output = normalized.to(input_dtype) * weight
            return output if bias is None else output + bias

        layer_norm.rms_norm_fn = rms_norm_fn  # type: ignore[attr-defined]

        def flash_attn_func(
            query: Any,
            key: Any,
            value: Any,
            *,
            causal: bool = False,
            softmax_scale: float | None = None,
            **_: Any,
        ) -> Any:
            output = torch.nn.functional.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                is_causal=causal,
                scale=softmax_scale,
                enable_gqa=True,
            )
            return output.transpose(1, 2).contiguous()

        def unsupported_varlen(*_: Any, **__: Any) -> Any:
            raise RuntimeError("the portable SDAR backend does not use varlen FlashAttention")

        root.flash_attn_func = flash_attn_func  # type: ignore[attr-defined]
        root.flash_attn_varlen_func = unsupported_varlen  # type: ignore[attr-defined]
        bert_padding.index_first_axis = unsupported_varlen  # type: ignore[attr-defined]
        bert_padding.pad_input = unsupported_varlen  # type: ignore[attr-defined]
        bert_padding.unpad_input = unsupported_varlen  # type: ignore[attr-defined]
        root.ops = ops  # type: ignore[attr-defined]
        ops.triton = triton  # type: ignore[attr-defined]
        triton.layer_norm = layer_norm  # type: ignore[attr-defined]
        sys.modules.update(
            {
                "flash_attn": root,
                "flash_attn.ops": ops,
                "flash_attn.ops.triton": triton,
                "flash_attn.ops.triton.layer_norm": layer_norm,
                "flash_attn.bert_padding": bert_padding,
            }
        )
        return True

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        mask_token_id: int | None = None,
        model_id: str | None = None,
        active_parameters: int | None = None,
        **model_kwargs: Any,
    ) -> "SDARTransformersBackend":
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional runtime
            raise RuntimeError("install the dllm optional dependency set") from exc

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype not in dtype_map:
            raise ValueError(f"unsupported dtype {dtype!r}")
        cls.install_portable_rms_norm()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=dtype_map[dtype],
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            **model_kwargs,
        ).to(device)
        return cls(
            model,
            tokenizer,
            model_id=model_id or model_name_or_path,
            mask_token_id=mask_token_id,
            active_parameters=active_parameters,
        )

    @staticmethod
    def _layout(
        prefix_length: int,
        generation_length: int,
        block_length: int,
    ) -> tuple[int, int, int, int]:
        first_block = prefix_length // block_length
        end = prefix_length + generation_length
        block_count = (end + block_length - 1) // block_length
        total_length = block_count * block_length
        return first_block, block_count, total_length, first_block * block_length

    @staticmethod
    def _available_schedule(available: int, configured_steps: int) -> tuple[int, ...]:
        if available <= 0:
            return ()
        steps = min(available, configured_steps)
        quotient, remainder = divmod(available, steps)
        return tuple(quotient + (index < remainder) for index in range(steps))

    def _new_cache(self) -> Any:
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError as exc:  # pragma: no cover - version guard
            raise RuntimeError("SDAR requires Transformers DynamicCache support") from exc
        return DynamicCache()

    def _block_attention_mask(self, block_count: int, block_length: int) -> Any:
        block_mask = self._torch.tril(
            self._torch.ones(
                block_count,
                block_count,
                dtype=self._torch.float32,
                device=self._device,
            )
        )
        return (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
        )

    def _sdar_logits(
        self,
        current: Any,
        *,
        attention_mask: Any,
        position_ids: Any,
        cache: Any,
        sampling: DiffusionSamplingConfig,
        mask_token_id: int,
        store_kv: bool,
        phase: Literal["sample", "score"],
    ) -> Any:
        output = self.model(
            current,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            store_kv=store_kv,
        )
        self._record_forward(
            current.shape[0], current.shape[1], phase=phase
        )
        logits = output.logits.clone()
        logits[..., mask_token_id] = -self._torch.inf
        return self._filter_logits(logits, sampling)

    def _initialize_group(
        self,
        requests: Sequence[DiffusionGenerationRequest],
        *,
        phase: Literal["sample", "score"],
    ) -> tuple[Any, Any, Any, Any, int, int, int, int, int]:
        first = requests[0]
        prefix_length = len(first.prefix)
        generation_length = first.generation_length
        sampling = first.sampling
        block_length = sampling.block_length
        first_block, block_count, total_length, prefill_length = self._layout(
            prefix_length, generation_length, block_length
        )
        mask_token_id = self._resolve_mask_token_id(sampling)
        tokens = self._torch.full(
            (len(requests), total_length),
            mask_token_id,
            dtype=self._torch.long,
            device=self._device,
        )
        for row, request in enumerate(requests):
            tokens[row, :prefix_length] = self._torch.tensor(
                request.prefix,
                dtype=self._torch.long,
                device=self._device,
            )
        attention = self._block_attention_mask(block_count, block_length)
        positions = self._torch.arange(total_length, device=self._device).unsqueeze(0)
        cache = self._new_cache()
        if prefill_length:
            self._sdar_logits(
                tokens[:, :prefill_length],
                attention_mask=attention[:, :prefill_length, :prefill_length],
                position_ids=positions[:, :prefill_length],
                cache=cache,
                sampling=sampling,
                mask_token_id=mask_token_id,
                store_kv=True,
                phase=phase,
            )
        return (
            tokens,
            attention,
            positions,
            cache,
            prefix_length,
            generation_length,
            first_block,
            block_count,
            mask_token_id,
        )

    def _sample_group(
        self,
        requests: Sequence[DiffusionGenerationRequest],
    ) -> list[DiffusionSample]:
        first = requests[0]
        sampling = first.sampling
        block_length = sampling.block_length
        (
            tokens,
            attention,
            positions,
            cache,
            prefix_length,
            generation_length,
            first_block,
            block_count,
            mask_token_id,
        ) = self._initialize_group(requests, phase="sample")
        generators = [
            self._torch.Generator(device=self._device).manual_seed(request.seed)
            for request in requests
        ]
        traces: list[list[DiffusionTraceStep]] = [[] for _ in requests]
        trajectory_logprobs = [0.0 for _ in requests]
        exact = sampling.has_exact_trajectory_density
        target_end = prefix_length + generation_length

        for absolute_block in range(first_block, block_count):
            relative_block = absolute_block - first_block
            block_start = absolute_block * block_length
            block_end = block_start + block_length
            valid_start = max(prefix_length, block_start)
            valid_end = min(target_end, block_end)
            available_initial = max(0, valid_end - valid_start)
            schedule = self._available_schedule(
                available_initial, sampling.steps_per_block
            )
            current = tokens[:, block_start:block_end].clone()
            current_attention = attention[:, block_start:block_end, :block_end]
            current_positions = positions[:, block_start:block_end]

            for step_index, scheduled_transfer in enumerate(schedule):
                active_rows = []
                for row in range(len(requests)):
                    local = current[row, valid_start - block_start : valid_end - block_start]
                    if bool((local == mask_token_id).any().item()):
                        active_rows.append(row)
                if not active_rows:
                    break
                logits = self._sdar_logits(
                    current,
                    attention_mask=current_attention,
                    position_ids=current_positions,
                    cache=cache,
                    sampling=sampling,
                    mask_token_id=mask_token_id,
                    store_kv=False,
                    phase="sample",
                )
                sampled, sampled_logprobs = self._draw_tokens(
                    logits,
                    temperature=sampling.temperature,
                    generators=generators,
                )
                if sampled_logprobs is None:
                    normalized = self._torch.log_softmax(logits.float(), dim=-1)
                    confidence_logprob = self._torch.gather(
                        normalized, -1, sampled.unsqueeze(-1)
                    ).squeeze(-1)
                else:
                    confidence_logprob = sampled_logprobs

                for row, generator in enumerate(generators):
                    local_start = valid_start - block_start
                    local_end = valid_end - block_start
                    masked_local = self._torch.nonzero(
                        current[row, local_start:local_end] == mask_token_id,
                        as_tuple=False,
                    ).flatten() + local_start
                    available = int(masked_local.numel())
                    if available == 0:
                        continue
                    transfer_count = min(scheduled_transfer, available)
                    if sampling.remasking == "low_confidence_dynamic":
                        high_confidence = confidence_logprob[row, masked_local].exp() > (
                            sampling.confidence_threshold
                        )
                        transfer_count = min(
                            available,
                            max(transfer_count, int(high_confidence.sum().item())),
                        )

                    if sampling.remasking == "random":
                        priorities = self._torch.rand(
                            available,
                            device=self._device,
                            generator=generator,
                        )
                    elif sampling.remasking == "sequential":
                        priorities = -self._torch.arange(
                            available,
                            device=self._device,
                            dtype=self._torch.float32,
                        )
                    else:
                        priorities = confidence_logprob[row, masked_local]
                    selected = self._torch.topk(
                        priorities,
                        k=transfer_count,
                        sorted=False,
                    ).indices
                    selected_local = masked_local[selected]
                    selected_absolute = selected_local + block_start
                    selected_relative = selected_absolute - prefix_length
                    order = self._torch.argsort(selected_relative)
                    selected_local = selected_local[order]
                    selected_absolute = selected_absolute[order]
                    selected_relative = selected_relative[order]
                    selected_tokens = sampled[row, selected_local]
                    current[row, selected_local] = selected_tokens
                    tokens[row, selected_absolute] = selected_tokens

                    step_logprob: float | None = None
                    if exact:
                        assert sampled_logprobs is not None
                        subset_logprob = (
                            self._subset_logprob(available, transfer_count)
                            if sampling.remasking == "random"
                            else 0.0
                        )
                        step_logprob = subset_logprob + float(
                            sampled_logprobs[row, selected_local].sum().item()
                        )
                        trajectory_logprobs[row] += step_logprob
                    traces[row].append(
                        DiffusionTraceStep(
                            block_index=relative_block,
                            step_index=step_index,
                            positions=tuple(
                                int(value) for value in selected_relative.tolist()
                            ),
                            token_ids=tuple(int(value) for value in selected_tokens.tolist()),
                            logprob=step_logprob,
                        )
                    )

            if bool(
                (
                    current[:, valid_start - block_start : valid_end - block_start]
                    == mask_token_id
                ).any().item()
            ):
                raise RuntimeError("SDAR denoising schedule left target positions masked")
            if block_end < target_end:
                self._sdar_logits(
                    current,
                    attention_mask=current_attention,
                    position_ids=current_positions,
                    cache=cache,
                    sampling=sampling,
                    mask_token_id=mask_token_id,
                    store_kv=True,
                    phase="sample",
                )

        outputs: list[DiffusionSample] = []
        eos_ids = getattr(getattr(self.model, "generation_config", None), "eos_token_id", None)
        if isinstance(eos_ids, int):
            eos_set = {eos_ids}
        elif eos_ids is None:
            eos_set = set()
        else:
            eos_set = {int(token_id) for token_id in eos_ids}
        for row, request in enumerate(requests):
            continuation = tuple(
                int(value)
                for value in tokens[row, prefix_length:target_end].tolist()
            )
            outputs.append(
                DiffusionSample(
                    prefix=request.prefix,
                    token_ids=continuation,
                    trace=tuple(traces[row]),
                    trajectory_logprob=(trajectory_logprobs[row] if exact else None),
                    policy_id=sampling.policy_id,
                    model_id=self.model_id,
                    request_id=request.request_id,
                    finish_reason=(
                        "eos" if any(token_id in eos_set for token_id in continuation) else "length"
                    ),
                )
            )
        return outputs

    def _score_group(
        self,
        requests: Sequence[DiffusionTrajectoryScoreRequest],
    ) -> list[float]:
        first = requests[0]
        sampling = first.sampling
        if not sampling.has_exact_trajectory_density:
            raise ValueError(
                "trajectory scoring requires random or sequential remasking and positive temperature"
            )
        generation_requests = [
            DiffusionGenerationRequest(
                prefix=request.sample.prefix,
                generation_length=len(request.sample.token_ids),
                sampling=sampling,
                seed=0,
                request_id=request.sample.request_id,
            )
            for request in requests
        ]
        (
            tokens,
            attention,
            positions,
            cache,
            prefix_length,
            generation_length,
            first_block,
            block_count,
            mask_token_id,
        ) = self._initialize_group(generation_requests, phase="score")
        block_length = sampling.block_length
        target_end = prefix_length + generation_length
        totals = [0.0 for _ in requests]
        trace_offsets = [0 for _ in requests]

        for absolute_block in range(first_block, block_count):
            relative_block = absolute_block - first_block
            block_start = absolute_block * block_length
            block_end = block_start + block_length
            valid_start = max(prefix_length, block_start)
            valid_end = min(target_end, block_end)
            available_initial = max(0, valid_end - valid_start)
            schedule = self._available_schedule(
                available_initial, sampling.steps_per_block
            )
            current = tokens[:, block_start:block_end].clone()
            current_attention = attention[:, block_start:block_end, :block_end]
            current_positions = positions[:, block_start:block_end]

            for step_index, transfer_count in enumerate(schedule):
                logits = self._sdar_logits(
                    current,
                    attention_mask=current_attention,
                    position_ids=current_positions,
                    cache=cache,
                    sampling=sampling,
                    mask_token_id=mask_token_id,
                    store_kv=False,
                    phase="score",
                ).float()
                scaled = logits / sampling.temperature
                log_normalizers = self._torch.logsumexp(scaled, dim=-1)
                for row, request in enumerate(requests):
                    if trace_offsets[row] >= len(request.sample.trace):
                        raise ValueError("sample trace ends before the SDAR schedule")
                    step = request.sample.trace[trace_offsets[row]]
                    trace_offsets[row] += 1
                    if step.block_index != relative_block or step.step_index != step_index:
                        raise ValueError("sample trace steps are out of SDAR schedule order")
                    if len(step.positions) != transfer_count:
                        raise ValueError("sample trace committed the wrong number of positions")
                    absolute_positions = self._torch.tensor(
                        [prefix_length + position for position in step.positions],
                        dtype=self._torch.long,
                        device=self._device,
                    )
                    if any(
                        not valid_start <= int(position) < valid_end
                        for position in absolute_positions.tolist()
                    ):
                        raise ValueError("sample trace commits a position outside its SDAR block")
                    local_positions = absolute_positions - block_start
                    if not bool(
                        self._torch.all(current[row, local_positions] == mask_token_id).item()
                    ):
                        raise ValueError("sample trace commits an already visible position")
                    if sampling.remasking == "sequential":
                        masked_local = self._torch.nonzero(
                            current[row, valid_start - block_start : valid_end - block_start]
                            == mask_token_id,
                            as_tuple=False,
                        ).flatten() + (valid_start - block_start)
                        expected = masked_local[:transfer_count]
                        if not bool(self._torch.equal(local_positions, expected)):
                            raise ValueError("sequential trajectory did not commit leftmost positions")
                    selected_tokens = self._torch.tensor(
                        step.token_ids,
                        dtype=self._torch.long,
                        device=self._device,
                    )
                    selected_logits = scaled[row, local_positions, selected_tokens]
                    selected_logprobs = selected_logits - log_normalizers[
                        row, local_positions
                    ]
                    available = int(
                        (
                            current[
                                row,
                                valid_start - block_start : valid_end - block_start,
                            ]
                            == mask_token_id
                        ).sum().item()
                    )
                    subset_logprob = (
                        self._subset_logprob(available, transfer_count)
                        if sampling.remasking == "random"
                        else 0.0
                    )
                    totals[row] += subset_logprob + float(selected_logprobs.sum().item())
                    current[row, local_positions] = selected_tokens
                    tokens[row, absolute_positions] = selected_tokens

            if block_end < target_end:
                self._sdar_logits(
                    current,
                    attention_mask=current_attention,
                    position_ids=current_positions,
                    cache=cache,
                    sampling=sampling,
                    mask_token_id=mask_token_id,
                    store_kv=True,
                    phase="score",
                )

        for row, request in enumerate(requests):
            if trace_offsets[row] != len(request.sample.trace):
                raise ValueError("sample trace contains steps beyond the SDAR schedule")
            final = tuple(
                int(value)
                for value in tokens[row, prefix_length:target_end].tolist()
            )
            if final != request.sample.token_ids:
                raise ValueError("sample trace does not reconstruct its final continuation")
        return totals


__all__ = ["SDARTransformersBackend"]
