"""vLLM 0.26 worker adapter that reads the unmodified model in the sampling step.

The public vLLM output contains probabilities under the processed sampling
policy. This adapter takes two numbers per request and step from the same
logits that produced each token, under the model at temperature 1: the chosen
token's log-probability (the base probability that MH needs) and the
distribution's KL divergence from uniform over the vocabulary (Self-Certainty).
It exposes them through ``collective_rpc``.

The module is imported only by an opted-in vLLM worker.  Keeping it separate
prevents the base package from importing the optional vLLM and torch stacks.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import math
from collections.abc import Sequence
from typing import Any

import torch
from packaging.version import Version
from vllm.v1.outputs import SamplerOutput
from vllm.v1.worker.gpu_model_runner import GPUModelRunner as _GPUModelRunner
from vllm.v1.worker.gpu_worker import Worker as _GPUWorker

if not Version("0.26") <= Version(importlib.metadata.version("vllm")) < Version("0.27"):
    raise RuntimeError(f"FusedLogprobWorker requires vLLM >=0.26,<0.27; found {importlib.metadata.version('vllm')}")

# Per request, the base log-probabilities and Self-Certainty of its generated tokens.
FusedValues = tuple[tuple[float, ...], tuple[float, ...]]


class _FusedLogprobGPUModelRunner(_GPUModelRunner):
    """Keep two numbers of the raw model distribution per request at each decode step."""

    def __init__(self, vllm_config: Any, device: Any) -> None:
        super().__init__(vllm_config, device)
        if self.use_async_scheduling:
            raise RuntimeError("fused log-probabilities require vLLM async_scheduling=False")
        if self.speculative_config is not None:
            raise RuntimeError("fused log-probabilities do not yet support speculative decoding")
        self._fused_step: torch.Tensor | None = None
        self._fused_by_request: dict[str, list[tuple[float, float]]] = {}

    def _sample(self, logits: torch.Tensor | None, spec_decode_metadata: Any | None) -> SamplerOutput:
        if spec_decode_metadata is not None:
            raise RuntimeError("fused log-probabilities do not support speculative decoding")
        metadata = self.input_batch.sampling_metadata
        if logits is None or (metadata.max_num_logprobs is None and not metadata.logprob_token_ids):
            self._fused_step = None
            return super()._sample(logits, spec_decode_metadata)
        # The sampling call may transform its logits in place, so the base distribution comes first.
        raw_logprobs = logits.log_softmax(dim=-1, dtype=torch.float32)
        output = super()._sample(logits, spec_decode_metadata)
        selected = output.sampled_token_ids.to(dtype=torch.int64)
        if selected.ndim != 2 or selected.shape[1] != 1:
            raise RuntimeError("fused log-probabilities expected one sampled token per request")
        # The chosen token's log-probability, and -log V - mean log p, the divergence from uniform.
        self._fused_step = torch.cat((raw_logprobs.gather(-1, selected), -raw_logprobs.mean(dim=-1, keepdim=True)
                                      - math.log(raw_logprobs.shape[-1])), dim=-1)
        return output

    def _bookkeeping_sync(self, *args: Any, **kwargs: Any) -> Any:
        step, self._fused_step = self._fused_step, None
        result = super()._bookkeeping_sync(*args, **kwargs)
        if step is None:
            return result
        # vLLM has already synchronized the generated token ids; this adds two floats per active request.
        values = step.detach().to(dtype=torch.float32, device="cpu").tolist()
        valid_token_ids, request_ids = result[2], result[4]
        if len(values) > len(request_ids) or len(values) > len(valid_token_ids):
            raise RuntimeError("vLLM returned inconsistent fused bookkeeping shapes")
        for index, (reference, certainty) in enumerate(values):
            if not valid_token_ids[index]:
                continue
            if len(valid_token_ids[index]) != 1:
                raise RuntimeError("fused log-probabilities require one-token decode steps")
            self._fused_by_request.setdefault(str(request_ids[index]), []).append((float(reference), float(certainty)))
        return result

    def pop_fused_logprobs(self, request_ids: Sequence[str]) -> dict[str, FusedValues]:
        """Return and release finished requests' values."""

        released = {str(request_id): self._fused_by_request.pop(str(request_id), None) for request_id in request_ids}
        return {request_id: (tuple(value for value, _ in values), tuple(value for _, value in values))
                for request_id, values in released.items() if values is not None}


class FusedLogprobWorker(_GPUWorker):
    """Install the fused runner without replacing vLLM's scheduler or outputs."""

    def init_device(self) -> None:
        if self.use_v2_model_runner:
            raise RuntimeError("fused log-probabilities require VLLM_USE_V2_MODEL_RUNNER=0")
        runner_module = importlib.import_module("vllm.v1.worker.gpu_model_runner")
        original, runner_module.GPUModelRunner = runner_module.GPUModelRunner, _FusedLogprobGPUModelRunner
        try:
            super().init_device()
        finally:
            runner_module.GPUModelRunner = original
        if not isinstance(self.model_runner, _FusedLogprobGPUModelRunner):
            raise RuntimeError("failed to install the fused-logprob model runner")

    def pop_fused_logprobs(self, request_ids: Sequence[str]) -> dict[str, FusedValues]:
        if not isinstance(self.model_runner, _FusedLogprobGPUModelRunner):
            raise RuntimeError("the fused-logprob model runner is unavailable")
        return self.model_runner.pop_fused_logprobs(request_ids)
