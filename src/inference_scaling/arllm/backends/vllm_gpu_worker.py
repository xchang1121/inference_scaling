"""CUDA GPU worker adapter matching the Ascend scalar-statistic contract.

Supports both vLLM V1 (0.25--0.26) and vLLM GPU V2 (0.30+) samplers, enabling
``scalar_self_certainty`` (including pre-mask full-vocabulary Self-Certainty
and top-K Consilience under top-k/top-p sampling), asynchronous scheduling,
streaming generation, ``token_penalty``, and sampling constraints on NVIDIA
GPUs without falling back to the legacy synchronous fused-logprob worker.
"""

from __future__ import annotations

from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from inference_scaling.arllm.backends.vllm_sampling import SamplingConstraintsWorker
from inference_scaling.arllm.backends.vllm_statistics import SCALAR_STATISTICS_KEY, install_scalar_statistics

_SAMPLER_TYPES: list[type] = []
for _mod_path in ("vllm.v1.worker.gpu.sample.sampler", "vllm.v1.sample.sampler"):
    try:
        _mod = __import__(_mod_path, fromlist=["Sampler"])
        _cls = getattr(_mod, "Sampler", None)
        if isinstance(_cls, type):
            _SAMPLER_TYPES.append(_cls)
    except ImportError:
        pass
_SAMPLER_TUPLE = tuple(_SAMPLER_TYPES)


class GPULogprobWorker(SamplingConstraintsWorker, GPUWorker):
    def _configure_sampler(self) -> None:
        runner = getattr(self, "model_runner", None)
        sampler = getattr(runner, "sampler", None)
        if sampler is None:
            return
        if _SAMPLER_TUPLE and not isinstance(sampler, _SAMPLER_TUPLE):
            raise RuntimeError(
                f"GPU processed log-probabilities require a supported vLLM Sampler; got {type(sampler)!r}"
            )
        mode = self.model_config.logprobs_mode
        if mode != "processed_logprobs":
            raise ValueError("the inference-scaling backend requires processed_logprobs")
        sampler.logprobs_mode = mode
        topk_topp = getattr(sampler, "topk_topp_sampler", None)
        if topk_topp is not None and hasattr(topk_topp, "logprobs_mode"):
            topk_topp.logprobs_mode = mode
        if (self.vllm_config.additional_config or {}).get(SCALAR_STATISTICS_KEY, False):
            install_scalar_statistics(sampler, self.model_config.get_vocab_size())

    def init_device(self) -> None:
        super().init_device()
        self._configure_sampler()

    def load_model(self) -> None:
        super().load_model()
        self._configure_sampler()


__all__ = ["GPULogprobWorker"]
