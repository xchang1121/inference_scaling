from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from inference_scaling.dllm.backends import LLaDATransformersBackend
from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.types import (
    DiffusionGenerationRequest,
    DiffusionTrajectoryScoreRequest,
)


class TinyMaskedModel(torch.nn.Module):
    def __init__(self, bias: tuple[float, ...], name: str) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        self.config = SimpleNamespace(_name_or_path=name, mask_token_id=3)

    def forward(self, token_ids):
        batch, length = token_ids.shape
        logits = self.bias.view(1, 1, -1).expand(batch, length, -1).clone()
        return SimpleNamespace(logits=logits)


class TinyTokenizer:
    mask_token_id = 3

    def decode(self, token_ids, *, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def _backend(bias=(0.0, 0.5, 1.0, -2.0), name="tiny"):
    return LLaDATransformersBackend(TinyMaskedModel(bias, name), TinyTokenizer())


def test_random_remasking_trace_rescores_to_its_recorded_probability():
    backend = _backend()
    sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=2,
        temperature=0.8,
        remasking="random",
    )
    requests = [
        DiffusionGenerationRequest((0,), 4, sampling, seed, f"sample-{seed}")
        for seed in (7, 11, 19)
    ]

    samples = backend.sample_batch(requests)
    rescored = backend.score_trajectories(
        [DiffusionTrajectoryScoreRequest(sample, sampling) for sample in samples]
    )

    assert [score for score in rescored] == pytest.approx(
        [sample.trajectory_logprob for sample in samples], abs=1e-6
    )
    assert all(len(sample.trace) == 4 for sample in samples)
    assert all(sorted(position for step in sample.trace for position in step.positions) == [0, 1, 2, 3] for sample in samples)
    snapshot = backend.snapshot()
    assert snapshot.sample_requests == 3
    assert snapshot.score_requests == 3
    assert snapshot.generated_tokens == 12
    assert snapshot.model_token_slots > 0


def test_low_confidence_generation_is_not_mislabeled_as_exact_density():
    backend = _backend()
    sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=1,
        temperature=0.0,
        remasking="low_confidence",
    )
    sample = backend.sample_batch(
        [DiffusionGenerationRequest((0,), 2, sampling, 3, "greedy")]
    )[0]

    assert sample.token_ids == (2, 2)
    assert sample.trajectory_logprob is None
    with pytest.raises(ValueError, match="random or sequential remasking"):
        backend.score_trajectories([DiffusionTrajectoryScoreRequest(sample, sampling)])


def test_target_temperature_changes_the_same_trajectory_score():
    backend = _backend()
    proposal = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=2,
        temperature=1.5,
        remasking="random",
    )
    target = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=2,
        temperature=0.5,
        remasking="random",
    )
    sample = backend.sample_batch(
        [DiffusionGenerationRequest((0,), 2, proposal, 31, "proposal")]
    )[0]

    target_score = backend.score_trajectories(
        [DiffusionTrajectoryScoreRequest(sample, target)]
    )[0]

    assert target_score != pytest.approx(sample.trajectory_logprob)
