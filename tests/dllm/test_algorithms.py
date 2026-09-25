from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inference_scaling.dllm.algorithms.is_sampling import run_conditional_diffusion_is
from inference_scaling.dllm.algorithms.mh import run_diffusion_reward_mh
from inference_scaling.dllm.backends.llada import LLaDATransformersBackend
from inference_scaling.dllm.algorithms.config import DiffusionISConfig, DiffusionMHConfig
from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.types import DiffusionSample


class TinyMaskedModel(torch.nn.Module):
    def __init__(self, bias, name):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        self.config = SimpleNamespace(_name_or_path=name, mask_token_id=3)

    def forward(self, token_ids):
        batch, length = token_ids.shape
        return SimpleNamespace(
            logits=self.bias.view(1, 1, -1).expand(batch, length, -1).clone()
        )


class TinyTokenizer:
    mask_token_id = 3


def _backend(bias, name):
    return LLaDATransformersBackend(TinyMaskedModel(bias, name), TinyTokenizer())


def test_conditional_is_decision_block_can_span_native_diffusion_blocks():
    base = _backend((0.0, 0.5, 1.0, -2.0), "base")
    sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=2,
        temperature=1.0,
        remasking="random",
    )

    result = run_conditional_diffusion_is(
        backend=base,
        prompt=(0,),
        config=DiffusionISConfig(
            candidate_count=2,
            rollout_count=1,
            block_size=4,
            total_length=8,
            reward_temperature=1.0,
        ),
        sampling=sampling,
        reward=lambda _prompt, continuation: float(sum(continuation)),
        seed=17,
    )

    assert len(result.steps) == 2
    assert all(len(candidate.token_ids) == 4 for step in result.steps for candidate in step.candidates)
    assert len(result.token_ids) == 8


def test_conditional_is_rejects_decision_block_that_splits_native_block():
    base = _backend((0.0, 0.5, 1.0, -2.0), "base")
    sampling = DiffusionSamplingConfig(
        block_length=4,
        steps_per_block=4,
        temperature=1.0,
        remasking="random",
    )

    with pytest.raises(ValueError, match="divisible by block_length"):
        run_conditional_diffusion_is(
            backend=base,
            prompt=(0,),
            config=DiffusionISConfig(
                candidate_count=2,
                rollout_count=1,
                block_size=6,
                total_length=12,
                reward_temperature=1.0,
            ),
            sampling=sampling,
            reward=lambda _prompt, continuation: float(sum(continuation)),
            seed=17,
        )


def _empty_sample(value: int, request_id: str) -> DiffusionSample:
    return DiffusionSample(
        prefix=(),
        token_ids=(value,),
        trace=(),
        trajectory_logprob=None,
        policy_id="uniform",
        model_id="coin",
        request_id=request_id,
    )


class CoinBackend:
    model_id = "coin"

    def sample_batch(self, requests):
        return [
            _empty_sample(int(np.random.default_rng(request.seed).integers(0, 2)), request.request_id)
            for request in requests
        ]


def test_independence_mh_approaches_base_times_reward_target_without_scores():
    ones = 0
    runs = 2500
    for seed in range(runs):
        result = run_diffusion_reward_mh(
            backend=CoinBackend(),
            prompt=(),
            config=DiffusionMHConfig(total_length=1, updates=8, reward_temperature=1.0),
            sampling=DiffusionSamplingConfig(
                block_length=1,
                steps_per_block=1,
                temperature=0.0,
            ),
            reward=lambda _prompt, continuation: float(continuation[0]),
            seed=seed,
        )
        ones += result.final.token_ids[0]
    expected = np.e / (1.0 + np.e)
    assert ones / runs == pytest.approx(expected, abs=0.03)
