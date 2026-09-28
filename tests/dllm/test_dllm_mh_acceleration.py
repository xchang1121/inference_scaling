from __future__ import annotations

from dataclasses import replace
from math import log

import numpy as np
import pytest

from inference_scaling.dllm.algorithms.config import DiffusionMHConfig
from inference_scaling.dllm.algorithms.mh import run_diffusion_reward_mh
from inference_scaling.dllm.algorithms.mh_acceleration import run_diffusion_replay_mixture_mh
from inference_scaling.shared.types import pointwise
from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.types import DiffusionGenerationRequest, DiffusionSample, DiffusionTraceStep


class CountingCoinBackend:
    def __init__(self, probability_one: float = 0.6) -> None:
        self.model_id, self.probability_one, self.batch_calls = "coin", probability_one, 0

    def _logprob(self, token: int) -> float:
        return log(self.probability_one if token else 1 - self.probability_one)

    def sample_batch(self, requests):
        self.batch_calls += 1
        outputs = []
        for request in requests:
            rng = np.random.default_rng(request.seed)
            tokens = tuple(int(rng.random() < self.probability_one) for _ in range(request.generation_length))
            trace = tuple(DiffusionTraceStep(position, 0, (position,), (token,), self._logprob(token))
                          for position, token in enumerate(tokens))
            outputs.append(DiffusionSample(request.prefix, tokens, trace, sum(self._logprob(token) for token in tokens),
                                           request.sampling.policy_id, self.model_id, request.request_id))
        return outputs


EXACT = DiffusionSamplingConfig(block_length=1, steps_per_block=1, temperature=1.0, remasking="random", top_k=0,
                                top_p=1.0, cfg_scale=0.0)
CONFIG = DiffusionMHConfig(total_length=2, updates=6, reward_temperature=1.0)


def _zero_reward(_prompt, continuations):
    return [0.0 for _ in continuations]


def test_independence_mh_draws_every_proposal_in_one_batch():
    backend = CountingCoinBackend()
    result = run_diffusion_reward_mh(backend=backend, prompt=(9,), config=CONFIG, sampling=EXACT, seed=4,
                                     reward=pointwise(lambda _prompt, continuation: float(sum(continuation))))
    assert len(result.steps) == CONFIG.updates and backend.batch_calls == 1


def test_zero_history_weight_replay_mixture_reduces_to_base_independence_mh():
    backend = CountingCoinBackend()
    history = backend.sample_batch([DiffusionGenerationRequest((9,), 2, EXACT, 2, "history")])[0]
    result = run_diffusion_replay_mixture_mh(backend=backend, prompt=(9,), config=CONFIG, sampling=EXACT,
                                             history=(history,), history_probability=0.0, reward=_zero_reward, seed=5)
    assert result.history_draws == 0 and result.acceptance_rate == 1.0


def test_replay_mixture_rejects_a_cache_from_another_model():
    backend = CountingCoinBackend()
    history = backend.sample_batch([DiffusionGenerationRequest((9,), 2, EXACT, 2, "history")])[0]
    with pytest.raises(ValueError, match="match the prompt and exact policy"):
        run_diffusion_replay_mixture_mh(backend=backend, prompt=(9,), config=CONFIG, sampling=EXACT,
                                        history=(replace(history, model_id="another-model"),), history_probability=0.5,
                                        reward=_zero_reward, seed=5)
