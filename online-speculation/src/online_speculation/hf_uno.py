"""Checkpoint-level Uno linear decoding with a portable Hugging Face backend.

This backend reproduces the two-pass algorithm and reuses a DynamicCache, but
it does not reproduce Nano-vLLM's fused Triton/FlashAttention/CUDA-graph
runtime. Consequently its acceptance/TPF results are algorithmically useful;
its wall-clock numbers describe this fallback backend only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor

from .tokenwise_lora import TokenwiseLoraRouter
from .torch_sampling import (
    SamplingConfig,
    filtered_distribution,
    verify_linear_filtered,
    verify_linear_greedy,
)


BASE_WEIGHT_SHA256 = "6392cc67c8dcc7aef1575f94ecdf3c7113b7d0e8f4e7058c4c3c74d4d876c365"
ADAPTER_WEIGHT_SHA256 = "5a499229d19ef4a69eb0b21884819d1b67cd983ba02b7ee2031ba8567dedfe4e"
BASE_REVISION = "ee770e713760cf6350e4322cdbbff91a163b7d70"
ADAPTER_REVISION = "b0d8896a301a2f4bc755538b1234a35100da50d0"


@dataclass(frozen=True)
class RunMetrics:
    method: str
    block_size: int
    prompt_tokens: int
    output_tokens: int
    decoder_tokens: int
    decoder_forwards: int
    cycles: int
    committed_cycle_tokens: int
    accepted_spec_tokens: int
    attempted_spec_tokens: int
    lookaheads: int
    prefill_seconds: float
    decode_seconds: float
    end_to_end_seconds: float
    decode_tokens_per_second: float
    end_to_end_tokens_per_second: float
    decoder_tokens_per_forward: float
    mean_tokens_per_cycle: float
    spec_acceptance_rate: float
    runtime_peak_memory_delta_bytes: int
    peak_memory_allocated_bytes: int
    stopped: bool
    output_token_ids: tuple[int, ...]
    output_text: str


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _cache_length(cache: object) -> int:
    getter = getattr(cache, "get_seq_length", None)
    if not callable(getter):
        raise TypeError("model did not return a cache with get_seq_length().")
    return int(getter())


def _crop_cache_by(cache: object, tokens_to_remove: int) -> None:
    """Remove a suffix using the negative-crop convention shared by HF 4/5."""

    if tokens_to_remove < 0:
        raise ValueError("tokens_to_remove cannot be negative.")
    if tokens_to_remove == 0:
        return
    before = _cache_length(cache)
    crop = getattr(cache, "crop", None)
    if not callable(crop):
        raise TypeError("model cache does not expose crop().")
    crop(-tokens_to_remove)
    after = _cache_length(cache)
    expected = before - tokens_to_remove
    if after != expected:
        raise RuntimeError(f"cache crop invariant failed: {before} -> {after}, expected {expected}.")


def _first_stop_length(tokens: Iterable[int], stop_ids: set[int]) -> int:
    token_list = list(tokens)
    for index, token in enumerate(token_list):
        if int(token) in stop_ids:
            return index + 1
    return len(token_list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


class HfUnoRuntime:
    """Single-GPU, batch-one reference runtime for AR and linear Uno."""

    def __init__(
        self,
        *,
        model: object,
        tokenizer: object,
        router: TokenwiseLoraRouter,
        device: torch.device,
        sampling: SamplingConfig,
        mask_token_id: int,
        stop_token_ids: Iterable[int],
        ignore_stop: bool,
        adapter_load_report: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.router = router
        self.device = device
        self.sampling = sampling
        self.mask_token_id = int(mask_token_id)
        self.stop_token_ids = {int(token) for token in stop_token_ids}
        self.ignore_stop = bool(ignore_stop)
        self.adapter_load_report = adapter_load_report or {}
        self.vocab_size = int(model.config.vocab_size)
        if not 1 < self.mask_token_id <= self.vocab_size:
            raise ValueError(
                "uniform-noise upper bound must lie in [2, vocab_size], got "
                f"{self.mask_token_id} for vocabulary {self.vocab_size}."
            )

    def encode_prompt(self, prompt: str) -> Tensor:
        messages = [{"role": "user", "content": prompt}]
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except (AttributeError, ValueError):
            encoded = self.tokenizer(prompt, return_tensors="pt").input_ids
        if not isinstance(encoded, Tensor):
            try:
                encoded = encoded["input_ids"]
            except (KeyError, TypeError) as error:
                raise TypeError("chat template did not return input_ids.") from error
        if encoded.ndim != 2 or encoded.size(0) != 1 or encoded.size(1) < 1:
            raise ValueError("prompt tokenizer must return one non-empty sequence.")
        return encoded.to(self.device)

    def _sample_logits(self, logits: Tensor, generator: torch.Generator) -> Tensor:
        if self.sampling.temperature <= 0:
            return torch.argmax(logits, dim=-1)
        return filtered_distribution(logits, self.sampling).sample(generator)

    def _base_context(self):
        disable_adapter = getattr(self.model, "disable_adapter", None)
        return disable_adapter() if callable(disable_adapter) else nullcontext()

    def _prefill(
        self,
        input_ids: Tensor,
        generator: torch.Generator,
    ) -> tuple[object, int, float]:
        _sync(self.device)
        start = time.perf_counter()
        with torch.inference_mode(), self._base_context():
            output = self.model(input_ids=input_ids, use_cache=True)
        seed_token = int(self._sample_logits(output.logits[:, -1, :], generator).item())
        _sync(self.device)
        return output.past_key_values, seed_token, time.perf_counter() - start

    def generate_ar(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        seed: int,
    ) -> RunMetrics:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        generator = _generator(self.device, seed)
        base_memory = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        cache, current_token, prefill_seconds = self._prefill(input_ids, generator)
        output_tokens = [current_token]
        stopped = not self.ignore_stop and current_token in self.stop_token_ids
        forwards = 0

        _sync(self.device)
        decode_start = time.perf_counter()
        while len(output_tokens) < max_new_tokens and not stopped:
            token_tensor = torch.tensor([[current_token]], device=self.device, dtype=torch.long)
            with torch.inference_mode(), self._base_context():
                output = self.model(
                    input_ids=token_tensor,
                    past_key_values=cache,
                    use_cache=True,
                )
            cache = output.past_key_values
            current_token = int(self._sample_logits(output.logits[:, -1, :], generator).item())
            output_tokens.append(current_token)
            forwards += 1
            stopped = not self.ignore_stop and current_token in self.stop_token_ids
        _sync(self.device)
        decode_seconds = time.perf_counter() - decode_start

        return self._metrics(
            method="ar",
            block_size=1,
            input_ids=input_ids,
            output_tokens=output_tokens,
            forwards=forwards,
            cycles=forwards,
            committed_cycle_tokens=forwards,
            accepted_spec_tokens=0,
            attempted_spec_tokens=0,
            lookaheads=0,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            base_memory=base_memory,
            stopped=stopped,
        )

    def generate_uno(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int,
        block_size: int,
        seed: int,
    ) -> RunMetrics:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        if block_size < 1:
            raise ValueError("block_size must be positive.")
        generator = _generator(self.device, seed)
        base_memory = torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        cache, seed_token, prefill_seconds = self._prefill(input_ids, generator)
        output_tokens = [seed_token]
        stopped = not self.ignore_stop and seed_token in self.stop_token_ids
        cycles = 0
        committed_cycle_tokens = 0
        accepted_spec_tokens = 0
        attempted_spec_tokens = 0
        lookaheads = 0

        _sync(self.device)
        decode_start = time.perf_counter()
        while len(output_tokens) < max_new_tokens and not stopped:
            prefix_cache_length = _cache_length(cache)
            if block_size > 1:
                noise = torch.randint(
                    1,
                    self.mask_token_id,
                    (1, block_size - 1),
                    device=self.device,
                    dtype=torch.long,
                    generator=generator,
                )
                seed_tensor = torch.tensor([[seed_token]], device=self.device, dtype=torch.long)
                draft_input = torch.cat((seed_tensor, noise), dim=1)
            else:
                draft_input = torch.tensor([[seed_token]], device=self.device, dtype=torch.long)

            lora_mask = torch.ones((1, block_size), device=self.device, dtype=torch.float32)
            lora_mask[:, 0] = 0.0
            self.router.set_token_mask(lora_mask)
            with torch.inference_mode():
                draft_output = self.model(
                    input_ids=draft_input,
                    past_key_values=cache,
                    use_cache=True,
                )
            cache = draft_output.past_key_values
            if _cache_length(cache) != prefix_cache_length + block_size:
                raise RuntimeError("draft cache length did not advance by block_size.")
            _crop_cache_by(cache, block_size - 1)

            draft_logits = draft_output.logits[0]
            free_token = int(self._sample_logits(draft_logits[0:1], generator).item())
            if block_size > 1:
                if self.sampling.temperature <= 0:
                    spec_tokens = torch.argmax(draft_logits[1:], dim=-1)
                    draft_used = None
                else:
                    draft_used = filtered_distribution(draft_logits[1:], self.sampling)
                    spec_tokens = draft_used.sample(generator)
            else:
                spec_tokens = torch.empty((0,), device=self.device, dtype=torch.long)
                draft_used = None
            proposal = torch.cat(
                (
                    torch.tensor([free_token], device=self.device, dtype=torch.long),
                    spec_tokens,
                )
            ).unsqueeze(0)

            with torch.inference_mode(), self._base_context():
                verify_output = self.model(
                    input_ids=proposal,
                    past_key_values=cache,
                    use_cache=True,
                )
            cache = verify_output.past_key_values
            verify_logits = verify_output.logits[0]
            if block_size > 1:
                if self.sampling.temperature <= 0:
                    verification = verify_linear_greedy(
                        free_token=free_token,
                        spec_tokens=spec_tokens,
                        target_logits=verify_logits[:-1],
                        lookahead_logits=verify_logits[-1],
                    )
                else:
                    target = filtered_distribution(verify_logits[:-1], self.sampling)
                    lookahead = filtered_distribution(verify_logits[-1:], self.sampling)
                    verification = verify_linear_filtered(
                        free_token=free_token,
                        spec_tokens=spec_tokens,
                        target=target,
                        draft_used=draft_used,
                        lookahead=lookahead,
                        generator=generator,
                    )
            else:
                lookahead_token = int(self._sample_logits(verify_logits[-1:], generator).item())
                from .torch_sampling import VerificationResult

                verification = VerificationResult(
                    committed=(free_token, lookahead_token),
                    accepted_spec_tokens=0,
                    rejected_index=None,
                    used_lookahead=True,
                )

            committed = list(verification.committed)
            if not self.ignore_stop:
                committed = committed[: _first_stop_length(committed, self.stop_token_ids)]
            remaining = max_new_tokens - len(output_tokens)
            committed = committed[:remaining]
            if not committed:
                raise RuntimeError("Uno cycle committed no tokens.")

            # Verify produced B KV rows after the now-cached seed. Keep every
            # committed token except the final uncached tail for the next cycle.
            _crop_cache_by(cache, block_size + 1 - len(committed))
            expected_cache_length = prefix_cache_length + len(committed)
            if _cache_length(cache) != expected_cache_length:
                raise RuntimeError("post-verification cache frontier is inconsistent.")

            output_tokens.extend(committed)
            seed_token = committed[-1]
            cycles += 1
            committed_cycle_tokens += len(committed)
            visible_specs = min(verification.accepted_spec_tokens, max(0, len(committed) - 1))
            accepted_spec_tokens += visible_specs
            attempted_spec_tokens += min(block_size - 1, max(0, len(committed) - 1))
            lookaheads += int(verification.used_lookahead and len(committed) == block_size + 1)
            stopped = not self.ignore_stop and seed_token in self.stop_token_ids

        _sync(self.device)
        decode_seconds = time.perf_counter() - decode_start
        return self._metrics(
            method="uno_linear_hf_fallback",
            block_size=block_size,
            input_ids=input_ids,
            output_tokens=output_tokens,
            forwards=2 * cycles,
            cycles=cycles,
            committed_cycle_tokens=committed_cycle_tokens,
            accepted_spec_tokens=accepted_spec_tokens,
            attempted_spec_tokens=attempted_spec_tokens,
            lookaheads=lookaheads,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            base_memory=base_memory,
            stopped=stopped,
        )

    def _metrics(
        self,
        *,
        method: str,
        block_size: int,
        input_ids: Tensor,
        output_tokens: list[int],
        forwards: int,
        cycles: int,
        committed_cycle_tokens: int,
        accepted_spec_tokens: int,
        attempted_spec_tokens: int,
        lookaheads: int,
        prefill_seconds: float,
        decode_seconds: float,
        base_memory: int,
        stopped: bool,
    ) -> RunMetrics:
        decoder_tokens = max(0, len(output_tokens) - 1)
        end_to_end = prefill_seconds + decode_seconds
        peak = torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0
        decoded = self.tokenizer.decode(output_tokens, skip_special_tokens=False)
        return RunMetrics(
            method=method,
            block_size=block_size,
            prompt_tokens=int(input_ids.size(1)),
            output_tokens=len(output_tokens),
            decoder_tokens=decoder_tokens,
            decoder_forwards=forwards,
            cycles=cycles,
            committed_cycle_tokens=committed_cycle_tokens,
            accepted_spec_tokens=accepted_spec_tokens,
            attempted_spec_tokens=attempted_spec_tokens,
            lookaheads=lookaheads,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
            end_to_end_seconds=end_to_end,
            decode_tokens_per_second=decoder_tokens / decode_seconds if decode_seconds else 0.0,
            end_to_end_tokens_per_second=len(output_tokens) / end_to_end if end_to_end else 0.0,
            decoder_tokens_per_forward=(
                committed_cycle_tokens / forwards if forwards else 0.0
            ),
            mean_tokens_per_cycle=(
                committed_cycle_tokens / cycles if cycles else 0.0
            ),
            spec_acceptance_rate=(
                accepted_spec_tokens / attempted_spec_tokens
                if attempted_spec_tokens
                else 0.0
            ),
            runtime_peak_memory_delta_bytes=max(0, int(peak) - int(base_memory)),
            peak_memory_allocated_bytes=int(peak),
            stopped=stopped,
            output_token_ids=tuple(output_tokens),
            output_text=decoded,
        )

    def routing_probe(self, input_ids: Tensor, block_size: int, seed: int) -> dict[str, Any]:
        """Check seed isolation and non-zero LoRA effect on noise rows."""

        if block_size < 2:
            raise ValueError("routing probe needs block_size >= 2.")
        generator = _generator(self.device, seed)
        noise = torch.randint(
            1,
            self.mask_token_id,
            (1, block_size - 1),
            device=self.device,
            dtype=torch.long,
            generator=generator,
        )
        probe_ids = torch.cat((input_ids, noise), dim=1)
        mask = torch.zeros_like(probe_ids, dtype=torch.float32)
        mask[:, input_ids.size(1) :] = 1.0
        self.router.set_token_mask(mask)
        with torch.inference_mode():
            routed = self.model(input_ids=probe_ids, use_cache=False).logits.float()
        with torch.inference_mode(), self._base_context():
            base = self.model(input_ids=probe_ids, use_cache=False).logits.float()
        clean_diff = (routed[:, : input_ids.size(1)] - base[:, : input_ids.size(1)]).abs()
        noise_diff = (routed[:, input_ids.size(1) :] - base[:, input_ids.size(1) :]).abs()
        return {
            "hook_count": self.router.hook_count,
            "clean_rows_max_abs_difference": float(clean_diff.max().item()),
            "noise_rows_mean_abs_difference": float(noise_diff.mean().item()),
            "noise_rows_max_abs_difference": float(noise_diff.max().item()),
            "clean_rows_match": bool(clean_diff.max().item() <= 1e-5),
            "noise_rows_changed": bool(noise_diff.max().item() > 0),
        }


def load_runtime(
    *,
    model_path: Path,
    adapter_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    sampling: SamplingConfig,
    mask_token_id: int,
    stop_token_ids: Iterable[int],
    ignore_stop: bool,
) -> HfUnoRuntime:
    """Load the pinned K2 base and public Uno adapter."""

    from peft import LoraConfig, get_peft_model
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.to(device)
    lora_config = LoraConfig.from_pretrained(adapter_path)
    model = get_peft_model(model, lora_config)
    model.to(device)
    adapter_weight = adapter_path / "adapter_model.safetensors"
    expected_lora_names = {
        name
        for name, _ in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    loaded_names: set[str] = set()
    with safe_open(adapter_weight, framework="pt", device="cpu") as adapter_file:
        raw_keys = list(adapter_file.keys())
        for raw_name in raw_keys:
            parameter_name = _adapter_parameter_name(raw_name)
            if parameter_name not in expected_lora_names:
                raise RuntimeError(
                    f"adapter tensor {raw_name!r} maps to unknown parameter "
                    f"{parameter_name!r}."
                )
            parameter = model.get_parameter(parameter_name)
            tensor = adapter_file.get_tensor(raw_name)
            if tuple(tensor.shape) != tuple(parameter.shape):
                raise RuntimeError(
                    f"adapter shape mismatch for {raw_name}: "
                    f"{tuple(tensor.shape)} != {tuple(parameter.shape)}."
                )
            parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded_names.add(parameter_name)
    missing = sorted(expected_lora_names - loaded_names)
    if missing or len(loaded_names) != 392:
        raise RuntimeError(
            "Uno adapter did not map exactly 392 tensors: "
            f"loaded={len(loaded_names)}, missing={missing[:5]}."
        )
    adapter_load_report = {
        "raw_tensor_count": len(raw_keys),
        "expected_parameter_count": len(expected_lora_names),
        "loaded_parameter_count": len(loaded_names),
        "missing_parameter_count": len(missing),
        "all_shapes_match": True,
    }
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    router = TokenwiseLoraRouter(model)
    return HfUnoRuntime(
        model=model,
        tokenizer=tokenizer,
        router=router,
        device=device,
        sampling=sampling,
        mask_token_id=mask_token_id,
        stop_token_ids=stop_token_ids,
        ignore_stop=ignore_stop,
        adapter_load_report=adapter_load_report,
    )


def _adapter_parameter_name(raw_name: str) -> str:
    """Map the public conversion format to PEFT's nested parameter name."""

    if not raw_name.startswith("model.layers."):
        raise ValueError(f"unexpected public adapter key: {raw_name!r}.")
    if raw_name.endswith(".lora_A.weight"):
        suffix = raw_name.removesuffix(".lora_A.weight") + ".lora_A.default.weight"
    elif raw_name.endswith(".lora_B.weight"):
        suffix = raw_name.removesuffix(".lora_B.weight") + ".lora_B.default.weight"
    else:
        raise ValueError(f"unexpected public adapter key: {raw_name!r}.")
    return "base_model.model." + suffix


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(statistics.median(values)),
        "q1": float(np.percentile(array, 25)),
        "q3": float(np.percentile(array, 75)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["label"], []).append(run)
    result: dict[str, Any] = {}
    for label, items in grouped.items():
        result[label] = {
            "runs": len(items),
            "decode_tokens_per_second": _summary(
                [float(item["metrics"]["decode_tokens_per_second"]) for item in items]
            ),
            "end_to_end_tokens_per_second": _summary(
                [float(item["metrics"]["end_to_end_tokens_per_second"]) for item in items]
            ),
            "decoder_tokens_per_forward": _summary(
                [float(item["metrics"]["decoder_tokens_per_forward"]) for item in items]
            ),
            "spec_acceptance_rate": _summary(
                [float(item["metrics"]["spec_acceptance_rate"]) for item in items]
            ),
        }
    if "ar" in result:
        ar_tps = result["ar"]["decode_tokens_per_second"]["median"]
        for label, item in result.items():
            if label == "ar":
                continue
            item["median_decode_speedup_over_ar"] = (
                item["decode_tokens_per_second"]["median"] / ar_tps if ar_tps else 0.0
            )
    return result


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _parse_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--block-sizes", type=_parse_ints, default=[2, 4, 8, 16])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--mask-token-id", type=int, default=64256)
    parser.add_argument("--stop-token-ids", type=_parse_ints, default=[64019, 1])
    parser.add_argument("--ignore-stop", action="store_true")
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--skip-hash-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    model_path = args.model_path.resolve()
    adapter_path = args.adapter_path.resolve()
    base_weight = model_path / "model-00000-of-00001.safetensors"
    adapter_weight = adapter_path / "adapter_model.safetensors"
    for required in (base_weight, adapter_weight):
        if not required.is_file():
            raise FileNotFoundError(required)

    hashes = {
        "base_weight_sha256": _sha256(base_weight),
        "adapter_weight_sha256": _sha256(adapter_weight),
    }
    if not args.skip_hash_check:
        if hashes["base_weight_sha256"] != BASE_WEIGHT_SHA256:
            raise RuntimeError("base checkpoint SHA-256 does not match the pinned revision.")
        if hashes["adapter_weight_sha256"] != ADAPTER_WEIGHT_SHA256:
            raise RuntimeError("Uno adapter SHA-256 does not match the pinned revision.")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    sampling = SamplingConfig(
        temperature=float(args.temperature),
        top_k=int(args.top_k) if args.top_k > 0 else None,
        top_p=float(args.top_p),
    )
    runtime = load_runtime(
        model_path=model_path,
        adapter_path=adapter_path,
        device=device,
        dtype=_dtype(args.dtype),
        sampling=sampling,
        mask_token_id=args.mask_token_id,
        stop_token_ids=args.stop_token_ids,
        ignore_stop=args.ignore_stop,
    )
    prompts = args.prompt or [
        "Explain in three concise paragraphs why speculative decoding can be lossless."
    ]
    encoded_prompts = [runtime.encode_prompt(prompt) for prompt in prompts]
    routing_probe = runtime.routing_probe(
        encoded_prompts[0],
        block_size=max(2, min(max(args.block_sizes), 8)),
        seed=args.seed,
    )

    # Warm both code paths without including their timings in the formal runs.
    runtime.generate_ar(
        encoded_prompts[0],
        max_new_tokens=max(2, args.warmup_tokens),
        seed=args.seed - 1,
    )
    runtime.generate_uno(
        encoded_prompts[0],
        max_new_tokens=max(2, args.warmup_tokens),
        block_size=args.block_sizes[0],
        seed=args.seed - 1,
    )

    runs: list[dict[str, Any]] = []
    for repetition in range(args.repetitions):
        for prompt_index, input_ids in enumerate(encoded_prompts):
            run_seed = args.seed + 1000 * repetition + prompt_index
            methods: list[tuple[str, int]] = [("ar", 1)] + [
                (f"uno_b{block_size}", block_size) for block_size in args.block_sizes
            ]
            if repetition % 2:
                methods.reverse()
            for label, block_size in methods:
                if label == "ar":
                    metrics = runtime.generate_ar(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        seed=run_seed,
                    )
                else:
                    metrics = runtime.generate_uno(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        block_size=block_size,
                        seed=run_seed,
                    )
                runs.append(
                    {
                        "label": label,
                        "repetition": repetition,
                        "prompt_index": prompt_index,
                        "seed": run_seed,
                        "metrics": asdict(metrics),
                    }
                )
                print(
                    f"{label} rep={repetition} prompt={prompt_index} "
                    f"TPF={metrics.decoder_tokens_per_forward:.3f} "
                    f"decode_TPS={metrics.decode_tokens_per_second:.2f}",
                    flush=True,
                )

    result = {
        "schema_version": 1,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "execution_backend": "huggingface_pytorch_kv_cache_fallback",
        "claim_scope": {
            "algorithmic_fidelity": "linear Psi-Spec, gated LoRA, KV rollback, old-q verification",
            "performance_fidelity": (
                "Not the official Nano-vLLM backend; no Triton/FlashAttention paged KV or CUDA graphs."
            ),
        },
        "checkpoint": {
            "base_id": "IFM/K2-Horizon-0.9B",
            "base_revision": BASE_REVISION,
            "adapter_id": "IFM/K2-Horizon-0.9B-Uno",
            "adapter_revision": ADAPTER_REVISION,
            **hashes,
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": _package_version("transformers"),
            "peft": _package_version("peft"),
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_compute_capability": (
                list(torch.cuda.get_device_capability(device)) if device.type == "cuda" else None
            ),
        },
        "sampling": {
            **asdict(sampling),
            "mask_token_id": args.mask_token_id,
            "noise_support": [1, min(args.mask_token_id, runtime.vocab_size)],
            "stop_token_ids": args.stop_token_ids,
            "ignore_stop": args.ignore_stop,
        },
        "design": {
            "block_sizes": args.block_sizes,
            "max_new_tokens": args.max_new_tokens,
            "warmup_tokens": args.warmup_tokens,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "prompts": prompts,
        },
        "routing_probe": routing_probe,
        "adapter_load_report": runtime.adapter_load_report,
        "runs": runs,
        "summary": _summarize_runs(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
