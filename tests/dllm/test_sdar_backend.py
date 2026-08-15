from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from inference_scaling.dllm.backends import SDARTransformersBackend
from inference_scaling.dllm.algorithms import run_diffusion_trajectory_power_mh
from inference_scaling.dllm.config import (
    DiffusionPowerMHConfig,
    DiffusionSamplingConfig,
)
from inference_scaling.dllm.types import (
    DiffusionGenerationRequest,
    DiffusionTrajectoryScoreRequest,
)


class TinySDARModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(
            torch.tensor((0.0, 1.0, 5.0, -10.0), dtype=torch.float32)
        )
        self.config = SimpleNamespace(_name_or_path="tiny-sdar", mask_token_id=3)
        self.generation_config = SimpleNamespace(eos_token_id=[9])
        self.calls: list[tuple[int, int, bool]] = []

    def forward(self, input_ids, *, store_kv=False, **kwargs):
        del kwargs
        batch, length = input_ids.shape
        self.calls.append((batch, length, bool(store_kv)))
        logits = self.bias.view(1, 1, -1).expand(batch, length, -1).clone()
        return SimpleNamespace(logits=logits)


class TinyTokenizer:
    mask_token_id = 3

    def decode(self, token_ids, *, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def _backend() -> SDARTransformersBackend:
    return SDARTransformersBackend(TinySDARModel(), TinyTokenizer())


def test_random_sdar_trajectory_rescores_with_partial_boundary_blocks():
    backend = _backend()
    sampling = DiffusionSamplingConfig(
        block_length=4,
        steps_per_block=4,
        temperature=0.8,
        remasking="random",
    )
    requests = [
        DiffusionGenerationRequest((0, 1, 2, 0, 1), 8, sampling, seed, f"r-{seed}")
        for seed in (7, 11)
    ]

    samples = backend.sample_batch(requests)
    rescored = backend.score_trajectories(
        [DiffusionTrajectoryScoreRequest(sample, sampling) for sample in samples]
    )

    assert rescored == pytest.approx(
        [sample.trajectory_logprob for sample in samples], abs=1e-6
    )
    assert all(len(sample.token_ids) == 8 for sample in samples)
    assert all(
        sorted(position for step in sample.trace for position in step.positions)
        == list(range(8))
        for sample in samples
    )
    assert any(store_kv for _, _, store_kv in backend.model.calls)
    snapshot = backend.snapshot()
    assert snapshot.sample_requests == 2
    assert snapshot.score_requests == 2
    assert snapshot.generated_tokens == 16
    assert snapshot.sample_model_token_slots > 0
    assert snapshot.score_model_token_slots > 0
    assert snapshot.model_token_slots == (
        snapshot.sample_model_token_slots + snapshot.score_model_token_slots
    )


def test_dynamic_threshold_can_commit_multiple_sdar_tokens_per_step():
    backend = _backend()
    sampling = DiffusionSamplingConfig(
        block_length=4,
        steps_per_block=4,
        temperature=0.0,
        remasking="low_confidence_dynamic",
        confidence_threshold=0.8,
    )

    sample = backend.sample_batch(
        [DiffusionGenerationRequest((0, 1, 2, 0), 8, sampling, 3, "dynamic")]
    )[0]

    assert sample.token_ids == (2,) * 8
    assert len(sample.trace) == 2
    assert all(len(step.positions) == 4 for step in sample.trace)
    assert sample.trajectory_logprob is None


def test_sequential_sdar_policy_has_an_exact_trajectory_density():
    backend = _backend()
    sampling = DiffusionSamplingConfig(
        block_length=4,
        steps_per_block=2,
        temperature=1.0,
        remasking="sequential",
        top_k=3,
    )
    sample = backend.sample_batch(
        [DiffusionGenerationRequest((0, 1, 2, 0), 4, sampling, 23, "sequential")]
    )[0]

    score = backend.score_trajectories(
        [DiffusionTrajectoryScoreRequest(sample, sampling)]
    )[0]

    assert score == pytest.approx(sample.trajectory_logprob, abs=1e-6)
    assert sample.trace[0].positions == (0, 1)
    assert sample.trace[1].positions == (2, 3)


def test_absolute_alignment_matches_official_prompt_block_boundaries():
    backend = _backend()
    sampling = DiffusionSamplingConfig(
        block_length=4,
        steps_per_block=4,
        temperature=1.0,
        remasking="random",
        block_alignment="absolute",
    )
    sample = backend.sample_batch(
        [DiffusionGenerationRequest((0, 1, 2, 0, 1), 7, sampling, 13, "absolute")]
    )[0]

    score = backend.score_trajectories(
        [DiffusionTrajectoryScoreRequest(sample, sampling)]
    )[0]

    assert len(sample.trace) == 7
    assert score == pytest.approx(sample.trajectory_logprob, abs=1e-6)


def test_sdar_power_mh_resamples_only_complete_absolute_block_suffixes():
    backend = _backend()
    prompt = (0, 1, 2, 0, 1)
    sampling = DiffusionSamplingConfig(
        block_length=4,
        steps_per_block=4,
        temperature=1.0,
        remasking="random",
        block_alignment="absolute",
    )

    result = run_diffusion_trajectory_power_mh(
        backend=backend,
        prompt=prompt,
        config=DiffusionPowerMHConfig(
            total_length=7,
            decision_block_size=4,
            updates_per_stage=1,
            alpha=2.0,
        ),
        sampling=sampling,
        seed=29,
    )

    assert len(result.final.token_ids) == 7
    assert len(result.steps) == 2
    assert all(
        step.cut == 0 or (len(prompt) + step.cut) % sampling.block_length == 0
        for step in result.steps
    )
