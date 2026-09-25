from __future__ import annotations

from collections import Counter
from math import log
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from inference_scaling.dllm.algorithms.search import (
    run_diffusion_block_beam,
    run_diffusion_trajectory_power_mh,
)
from inference_scaling.dllm.algorithms.config import (
    DiffusionBlockBeamConfig,
    DiffusionPowerMHConfig,
)
from inference_scaling.dllm.backends.llada import LLaDATransformersBackend
from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.types import DiffusionSample, DiffusionTraceStep


class BinaryTrajectoryBackend:
    model_id = "binary"

    def sample_batch(self, requests):
        samples = []
        for request in requests:
            rng = np.random.default_rng(request.seed)
            token = int(rng.choice(2, p=(0.8, 0.2)))
            logprob = log((0.8, 0.2)[token])
            steps = tuple(
                DiffusionTraceStep(
                    block_index=index,
                    step_index=0,
                    positions=(index,),
                    token_ids=(token,),
                    logprob=logprob,
                )
                for index in range(request.generation_length)
            )
            samples.append(
                DiffusionSample(
                    prefix=request.prefix,
                    token_ids=(token,) * request.generation_length,
                    trace=steps,
                    trajectory_logprob=logprob * request.generation_length,
                    policy_id=request.sampling.policy_id,
                    model_id=self.model_id,
                    request_id=request.request_id,
                    # The binary law ignores temperature, so the base probability equals the proposal's.
                    reference_trajectory_logprob=(
                        None if request.reference_temperature is None else logprob * request.generation_length
                    ),
                )
            )
        return samples


EXACT = DiffusionSamplingConfig(
    block_length=1,
    steps_per_block=1,
    temperature=1.0,
    remasking="random",
)


def test_trajectory_power_mh_sharpens_the_exact_trajectory_distribution():
    zeroes = 0
    runs = 2000
    for seed in range(runs):
        result = run_diffusion_trajectory_power_mh(
            backend=BinaryTrajectoryBackend(),
            prompt=(),
            config=DiffusionPowerMHConfig(
                total_length=1,
                decision_block_size=1,
                updates_per_stage=8,
                alpha=2.0,
            ),
            sampling=EXACT,
            seed=seed,
        )
        zeroes += result.final.token_ids[0] == 0

    expected = 0.8**2 / (0.8**2 + 0.2**2)
    assert zeroes / runs == pytest.approx(expected, abs=0.025)


class ContextModel(torch.nn.Module):
    """Logits depend on the previous token and on how many masks the canvas holds."""

    def __init__(self) -> None:
        super().__init__()
        self.table = torch.nn.Parameter(torch.tensor(
            [[0.0, 0.6, -0.4, 0.0], [0.5, -0.3, 0.2, 0.0], [-0.2, 0.4, 0.3, 0.0], [0.1, 0.1, 0.1, 0.0]]
        ))
        self.config = SimpleNamespace(_name_or_path="context", mask_token_id=3)

    def forward(self, token_ids):
        previous = torch.cat([token_ids[:, :1], token_ids[:, :-1]], dim=1)
        masks = (token_ids == 3).sum(dim=1, keepdim=True).float()
        return SimpleNamespace(logits=self.table[previous] + 0.7 * masks[..., None] * torch.tensor([1.0, -1.0, 0.0, 0.0]))


def test_trajectory_power_mh_targets_the_blockwise_power_of_a_context_dependent_model():
    backend = LLaDATransformersBackend(ContextModel(), SimpleNamespace(mask_token_id=3))
    config = DiffusionPowerMHConfig(total_length=2, decision_block_size=1, updates_per_stage=6, alpha=2.0)
    finals = Counter(
        run_diffusion_trajectory_power_mh(backend=backend, prompt=(0,), config=config, sampling=EXACT, seed=seed)
        .final.token_ids
        for seed in range(600)
    )
    # Each block is drawn with the canvas ending at the block, so p is a product of per-block softmaxes.
    with torch.no_grad():
        logits = ContextModel().table + 0.7 * torch.tensor([1.0, -1.0, 0.0, 0.0])
        logits[:, 3] = -torch.inf
        block = torch.log_softmax(logits, dim=-1)
    weights = {(a, b): float(torch.exp(2.0 * (block[0, a] + block[a, b]))) for a in range(3) for b in range(3)}
    total = sum(weights.values())
    target = {key: value / total for key, value in weights.items()}
    empirical = {key: finals[key] / 600 for key in target}
    assert sum(abs(empirical[key] - target[key]) for key in target) / 2 < 0.07


def test_block_beam_retains_width_and_accumulates_stage_probabilities():
    result = run_diffusion_block_beam(
        backend=BinaryTrajectoryBackend(),
        prompt=(9,),
        config=DiffusionBlockBeamConfig(
            total_length=3,
            decision_block_size=1,
            width=4,
            branching_factor=2,
        ),
        sampling=EXACT,
        seed=4,
    )

    assert [stage.proposals for stage in result.stages] == [4, 8, 8]
    assert len(result.beams) == 4
    assert len(result.best.token_ids) == 3
    assert result.best.trajectory_logprob == sum(
        float(sample.trajectory_logprob) for sample in result.best.samples
    )


def test_search_algorithms_reject_intractable_remasking_policy():
    inexact = DiffusionSamplingConfig(
        block_length=1,
        steps_per_block=1,
        temperature=0.0,
        remasking="low_confidence",
    )
    with pytest.raises(ValueError, match="exact diffusion policy"):
        run_diffusion_trajectory_power_mh(
            backend=BinaryTrajectoryBackend(),
            prompt=(),
            config=DiffusionPowerMHConfig(
                total_length=1,
                decision_block_size=1,
                updates_per_stage=2,
                alpha=2.0,
            ),
            sampling=inexact,
        )
    with pytest.raises(ValueError, match="exact diffusion policy"):
        run_diffusion_block_beam(
            backend=BinaryTrajectoryBackend(),
            prompt=(),
            config=DiffusionBlockBeamConfig(
                total_length=1,
                decision_block_size=1,
                width=2,
                branching_factor=2,
            ),
            sampling=inexact,
        )
