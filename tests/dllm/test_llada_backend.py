from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from inference_scaling.dllm.backends.llada import LLaDATransformersBackend
from inference_scaling.dllm.config import DiffusionSamplingConfig
from inference_scaling.dllm.types import DiffusionGenerationRequest


class TinyMaskedModel(torch.nn.Module):
    def __init__(self, bias: tuple[float, ...], name: str) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        self.config = SimpleNamespace(_name_or_path=name, mask_token_id=3)
        self.batch_sizes: list[int] = []

    def forward(self, token_ids):
        batch, length = token_ids.shape
        self.batch_sizes.append(batch)
        logits = self.bias.view(1, 1, -1).expand(batch, length, -1).clone()
        return SimpleNamespace(logits=logits)


class TinyTokenizer:
    mask_token_id = 3

    def decode(self, token_ids, *, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def _backend(bias=(0.0, 0.5, 1.0, -2.0), name="tiny", max_batch_size=64):
    return LLaDATransformersBackend(
        TinyMaskedModel(bias, name),
        TinyTokenizer(),
        max_batch_size=max_batch_size, mask_token_id=3,
    )


class TinyExpertLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = torch.nn.ModuleList(
            [torch.nn.Linear(1, 1, bias=False) for _ in range(2)]
        )
        self.dense = torch.nn.Parameter(torch.ones(1))


class TinyLayeredModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor((0.0, 0.5, 1.0, -2.0)))
        self.layers = torch.nn.ModuleList([TinyExpertLayer() for _ in range(4)])
        self.config = SimpleNamespace(
            _name_or_path="tiny-moe",
            mask_token_id=3,
            num_experts=2,
            num_experts_per_tok=1,
        )
        self.executed_layer_counts: list[int] = []

    def forward(self, token_ids):
        self.executed_layer_counts.append(len(self.layers))
        batch, length = token_ids.shape
        logits = self.bias.view(1, 1, -1).expand(batch, length, -1).clone()
        return SimpleNamespace(logits=logits)


def test_reference_temperature_reports_the_same_trajectory_at_the_base_temperature():
    backend = _backend()
    sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=2,
        temperature=0.8,
        remasking="random", top_k=0, top_p=1.0, cfg_scale=0.0,
    )
    requests = [
        DiffusionGenerationRequest((0,), 4, sampling, seed, f"sample-{seed}", reference_temperature=0.8)
        for seed in (7, 11, 19)
    ]

    samples = backend.sample_batch(requests)

    assert [sample.reference_trajectory_logprob for sample in samples] == pytest.approx(
        [sample.trajectory_logprob for sample in samples], abs=1e-6
    )
    assert all(len(sample.trace) == 4 for sample in samples)
    assert all(sorted(position for step in sample.trace for position in step.positions) == [0, 1, 2, 3] for sample in samples)
    snapshot = backend.snapshot()
    assert snapshot.sample_requests == 3
    assert snapshot.generated_tokens == 12
    assert snapshot.model_token_slots > 0
    other = backend.sample_batch([
        DiffusionGenerationRequest((0,), 2, replace(sampling, temperature=1.5), 31, "proposal", reference_temperature=0.5)
    ])[0]
    assert other.reference_trajectory_logprob != pytest.approx(other.trajectory_logprob)


def test_low_confidence_generation_is_not_mislabeled_as_exact_density():
    backend = _backend()
    sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=1,
        temperature=0.0,
        remasking="low_confidence", top_k=0, top_p=1.0, cfg_scale=0.0,
    )
    sample = backend.sample_batch(
        [DiffusionGenerationRequest((0,), 2, sampling, 3, "greedy")]
    )[0]

    assert sample.token_ids == (2, 2)
    assert sample.trajectory_logprob is None
    with pytest.raises(ValueError, match="exact diffusion policy"):
        backend.sample_batch([DiffusionGenerationRequest((0,), 2, sampling, 3, "greedy", reference_temperature=1.0)])


def test_active_parameters_count_the_routed_share_of_moe_experts():
    snapshot = LLaDATransformersBackend(TinyLayeredModel(), TinyTokenizer(), mask_token_id=3, max_batch_size=64).snapshot()
    assert (snapshot.total_parameters, snapshot.active_parameters, snapshot.resident_parameters) == (16, 12, 16)


def test_batch_limit_chunks_sampling_without_changing_results():
    sampling = DiffusionSamplingConfig(
        block_length=2,
        steps_per_block=1,
        temperature=0.8,
        remasking="random", top_k=0, top_p=1.0, cfg_scale=0.0,
    )
    requests = [
        DiffusionGenerationRequest((0,), 2, sampling, seed, f"sample-{seed}")
        for seed in range(5)
    ]
    limited = _backend(max_batch_size=2)
    unlimited = _backend()

    limited_samples = limited.sample_batch(requests)
    unlimited_samples = unlimited.sample_batch(requests)
    assert [sample.token_ids for sample in limited_samples] == [
        sample.token_ids for sample in unlimited_samples
    ]
    assert [sample.trajectory_logprob for sample in limited_samples] == pytest.approx(
        [sample.trajectory_logprob for sample in unlimited_samples]
    )
    assert limited.model.batch_sizes == [2, 2, 1]
    assert limited.snapshot().forward_calls == 3


class PromptModel(torch.nn.Module):
    """Every position prefers EOS (1) after prompt 0 and token 2 after any other prompt."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, token_ids):
        self.batch_sizes.append(token_ids.shape[0])
        preferred = (2 - (token_ids[:, :1] == 0).long()).expand_as(token_ids)
        return SimpleNamespace(logits=torch.nn.functional.one_hot(preferred, 4).float())


def test_stop_at_eos_ends_a_row_after_its_first_block_of_eos():
    model = PromptModel()
    backend = LLaDATransformersBackend(model, SimpleNamespace(eos_token_id=1), mask_token_id=3, max_batch_size=64)
    sampling = DiffusionSamplingConfig(block_length=2, steps_per_block=1, temperature=0.0, remasking="low_confidence",
                                       top_k=0, top_p=1.0, cfg_scale=0.0)
    ended, running, fixed = backend.sample_batch([
        DiffusionGenerationRequest((0,), 4, sampling, 0, "ended", stop_at_eos=True),
        DiffusionGenerationRequest((2,), 4, sampling, 0, "running", stop_at_eos=True),
        DiffusionGenerationRequest((0,), 4, sampling, 0, "fixed"),
    ])

    assert (ended.token_ids, ended.finish_reason) == ((1, 1), "eos")
    assert (running.token_ids, running.finish_reason) == ((2, 2, 2, 2), "length")
    assert (fixed.token_ids, fixed.finish_reason) == ((1, 1, 1, 1), "length")
    # The ended row leaves the batch before the second block.
    assert model.batch_sizes == [2, 1, 1, 1]
    assert backend.snapshot().generated_tokens == 10


def test_batch_limit_must_be_positive():
    with pytest.raises(ValueError, match="max_batch_size"):
        _backend(max_batch_size=0)
