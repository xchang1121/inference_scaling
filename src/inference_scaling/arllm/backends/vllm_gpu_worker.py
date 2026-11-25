"""CUDA GPU V1 worker adapter matching the Ascend scalar-statistic contract.

Enables ``scalar_self_certainty`` (including pre-mask full-vocabulary
Self-Certainty under top-k/top-p sampling), asynchronous scheduling, streaming
generation, ``token_penalty``, and worker-side sampling constraints on NVIDIA
GPUs without falling back to the legacy synchronous fused-logprob worker.
"""

from __future__ import annotations

from vllm.v1.sample.sampler import Sampler
from vllm.v1.worker.gpu_worker import Worker as GPUWorker

from inference_scaling.arllm.backends.vllm_sampling import SamplingConstraintsWorker
from inference_scaling.arllm.backends.vllm_statistics import SCALAR_STATISTICS_KEY, install_scalar_statistics


class GPULogprobWorker(SamplingConstraintsWorker, GPUWorker):
    def init_device(self) -> None:
        super().init_device()
        runner = self.model_runner
        sampler = getattr(runner, "sampler", None)
        if not isinstance(sampler, Sampler):
            raise RuntimeError(
                "GPU processed log-probabilities require the V1 Sampler "
                "(set VLLM_USE_V2_MODEL_RUNNER=0 if using an experimental runner)"
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


__all__ = ["GPULogprobWorker"]
