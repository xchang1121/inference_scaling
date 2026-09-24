"""Per-problem helpers shared by the AR GSM8K experiment entry points.

Entry points own data loading, manifests and records. This module renders the
prompt, loads backends and draws
plain (non-search) samples, so every entry point executes them identically.
"""

from __future__ import annotations

import importlib.metadata
import time
from collections import Counter
from fractions import Fraction
from typing import Any, Callable, Sequence

import torch

from inference_scaling.arllm.backends import load_backend_from_config
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, TokenSequence
from inference_scaling.shared.evaluation import GSM8KProblem, gsm8k_prompt
from inference_scaling.shared.model.prompting import render_prompt


def installed_package_version(name: str) -> str | None:
    """Return an optional runtime dependency version without masking load errors."""

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def fraction_text(value: Fraction | None) -> str | None:
    if value is None:
        return None
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def answer_counts(values: Sequence[Fraction | None]) -> dict[str, int]:
    """Return JSON-stable diagnostic keys, including unparseable candidates."""

    keys = (
        fraction_text(value) if value is not None else "<unparseable>"
        for value in values
    )
    return dict(sorted(Counter(keys).items()))


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(call: Callable[[], Any]) -> tuple[Any, float]:
    cuda_sync()
    started = time.perf_counter()
    result = call()
    cuda_sync()
    return result, time.perf_counter() - started


def prompt_tokens(backend: Any, problem: GSM8KProblem, config: dict[str, Any] | None = None) -> TokenSequence:
    messages = [{"role": "user", "content": gsm8k_prompt(problem.question)}]
    rendered = render_prompt(backend.tokenizer, messages, config or {})
    return backend.encode(str(rendered), add_special_tokens=False)


def load_backend(
    path: str,
    config: dict[str, Any],
    *,
    adapter_base: str | None = None,
    role: str | None = None,
) -> Any:
    return load_backend_from_config(path, config, adapter_base=adapter_base, role=role)


def trim_eos(tokens: TokenSequence, eos_token_id: int | None) -> TokenSequence:
    if eos_token_id is None or eos_token_id not in tokens:
        return tokens
    return tokens[: tokens.index(eos_token_id) + 1]


def direct_generate(
    backend: Any,
    prompt: TokenSequence,
    *,
    max_new_tokens: int,
    num_beams: int,
) -> TokenSequence:
    generated = backend.direct_generate(
        prompt,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return trim_eos(generated, backend.tokenizer.eos_token_id)


def sample_one(
    backend: Any,
    prompt: TokenSequence,
    *,
    max_new_tokens: int,
    temperature: float,
    seed: int,
    request_id: str,
) -> TokenSequence:
    sample = backend.sample_batch(
        [
            GenerationRequest(
                prompt,
                max_new_tokens,
                SamplingConfig(
                    temperature=temperature,
                    eos_token_id=backend.tokenizer.eos_token_id,
                ),
                seed,
                request_id,
            )
        ]
    )[0]
    return sample.token_ids


__all__ = [
    "answer_counts",
    "cuda_sync",
    "direct_generate",
    "fraction_text",
    "installed_package_version",
    "load_backend",
    "prompt_tokens",
    "sample_one",
    "timed",
    "trim_eos",
]
