"""Ascend V1 compatibility for the backend's processed-log-probability contract.

Some releases construct AscendSampler() with its raw-log-probability default,
even when ModelConfig requests processed probabilities. Set both mode fields
before profiling or graph capture. Native sampling, RNG and logits processors
remain unchanged; worker extensions remain available to the caller.
"""

from vllm_ascend.sample.sampler import AscendSampler, AscendTopKTopPSampler
from vllm_ascend.worker.worker import NPUWorker

from inference_scaling.arllm.backends.vllm_sampling import SamplingConstraintsWorker


class AscendLogprobWorker(SamplingConstraintsWorker, NPUWorker):
    def init_device(self):
        super().init_device()
        runner = self.model_runner
        sampler = getattr(runner, "sampler", None)
        if (not isinstance(sampler, AscendSampler)
                or not isinstance(sampler.topk_topp_sampler, AscendTopKTopPSampler)):
            raise RuntimeError("Ascend processed log-probabilities require the V1 AscendSampler")
        if runner.ascend_config.enable_reduce_sample:
            raise ValueError("processed log-probabilities require Ascend enable_reduce_sample=false")
        mode = self.model_config.logprobs_mode
        if mode != "processed_logprobs":
            raise ValueError("the inference-scaling backend requires processed_logprobs")
        sampler.logprobs_mode = mode
        sampler.topk_topp_sampler.logprobs_mode = mode
