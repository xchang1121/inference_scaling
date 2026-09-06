"""Tensor correction equivalence, captured sampling maps and online publication."""

import itertools

import pytest
import torch

from blockspec.calibration import OverlapMix
from blockspec.decoding import generate_ar, generate_speculative
from blockspec.model import Decoder, ModelConfig
from blockspec.sampling import SamplingConfig, probabilities, residual
from blockspec.sampling_execution import SamplingExecutor, exponential_choice, linear_correction


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph hardware required"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_acceptance_branches_match_scalar_residual(device, dtype):
    q = torch.tensor([[.6, .3, .1], [.5, .2, .3]], device=device, dtype=dtype)
    p = torch.tensor([[.2, .5, .3], [.25, .35, .4], [.1, .15, .75]], device=device, dtype=dtype)
    exponential = torch.tensor([.8, .2, 1.1], device=device, dtype=dtype)
    for candidates in itertools.product(range(3), repeat=2):
        ids = torch.tensor(candidates, device=device)
        for values in itertools.product([0., .25, .6, .99], repeat=2):
            uniforms = torch.tensor(values, device=device, dtype=dtype)
            count, tail, valid = linear_correction(ids, q, p, uniforms, exponential)
            expected = 0
            while expected < 2 and values[expected] < min(1., float(p[expected, ids[expected]] / q[expected, ids[expected]])):
                expected += 1
            law = residual(p[expected], q[expected])[0] if expected < 2 else p[-1]
            assert valid and int(count) == expected
            assert int(tail) == int(exponential_choice(law, exponential))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("top_p", [1., .95, .5])
@torch.no_grad()
def test_prepared_probability_maps_match_reference_and_own_snapshots(device, top_p):
    torch.manual_seed(616)
    sampling = SamplingConfig(1., 5, top_p)
    engine = SamplingExecutor(17, 4, sampling, device=device, use_cuda_graph=device == "cuda")
    logits = torch.randn(4, 17, device=device).round()
    baseline = probabilities(logits, sampling)
    mix = OverlapMix(4, 5, device=device, interval=1)
    gen = lambda: torch.Generator(device=device).manual_seed(17)
    tokens, q, _ = engine.draft(logits, gen())
    ids, original, feedback = engine.draft(logits, gen(), mix)
    torch.testing.assert_close(q, baseline, rtol=0, atol=0)
    torch.testing.assert_close(original, q, rtol=0, atol=0)
    assert torch.equal(tokens, ids)
    saved = q.clone()
    for _ in range(5):
        mix.observe(feedback, torch.randn(3, 17, device=device).softmax(-1))
    mixed_ids, changed, _ = engine.draft(logits, gen(), mix)
    expected, _ = mix.propose(baseline)
    torch.testing.assert_close(changed, expected, rtol=0, atol=0)
    torch.testing.assert_close(q, saved, rtol=0, atol=0)
    torch.testing.assert_close(feedback.baseline, feedback.mixed, rtol=0, atol=0)
    teacher = torch.randn(4, 17, device=device)
    verified, target = engine.verify(mixed_ids[1:], changed[1:], teacher, gen())
    torch.testing.assert_close(target, probabilities(teacher, sampling), rtol=0, atol=0)
    assert 1 <= len(verified.tokens) <= 4
    assert verified.tokens[:verified.accepted] == mixed_ids[1:1 + verified.accepted].tolist()
    target_copy = target.clone()
    engine.verify(mixed_ids[1:], changed[1:], teacher + 1., gen())
    torch.testing.assert_close(target, target_copy, rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("budget", [0, 1, 2, 9])
@torch.no_grad()
def test_captured_decoder_identity_and_frozen_online_stream(device, budget):
    torch.manual_seed(617)
    model = Decoder(ModelConfig(vocab_size=7, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                head_dim=8, adapter_rank=2)).to(device).eval().requires_grad_(False)
    prompt = torch.tensor([[1, 2, 3]], device=device)
    sampling = SamplingConfig(1., 5, .95)
    sampler = SamplingExecutor(7, 4, sampling, device=device, use_cuda_graph=device == "cuda")
    options = dict(block_size=4, sampling=sampling, sampler_executor=sampler)
    gen = lambda: torch.Generator(device=device).manual_seed(71)
    expected = generate_speculative(model, prompt, budget, generator=gen(), **options)
    identity = OverlapMix(4, 5, adaptive=False, device=device)
    actual = generate_speculative(model, prompt, budget, generator=gen(), calibrator=identity, **options)
    assert expected.tokens == actual.tokens
    assert expected.decode_forwards == actual.decode_forwards
    assert len(generate_ar(model, prompt, budget, sampling=sampling, sampler_executor=sampler).tokens) == budget
    mix = OverlapMix(4, 5, device=device, interval=1)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    result = generate_speculative(model, prompt, budget, generator=gen(), calibrator=mix, **options)
    assert len(result.tokens) == budget
    torch.testing.assert_close(state, model.state_dict(), rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_graph_boundary_rejects_nonfinite_logits_and_invalid_proposals(device):
    engine = SamplingExecutor(7, 4, SamplingConfig(1., 5), device=device, use_cuda_graph=device == "cuda")
    logits = torch.zeros(4, 7, device=device)
    tokens, q, _ = engine.draft(logits)
    bad = logits.clone()
    bad[1, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        engine.draft(bad)
    with pytest.raises(ValueError, match="finite"):
        engine.verify(tokens[1:], q[1:], bad)
    with pytest.raises(ValueError, match="valid"):
        engine.verify(tokens[1:], q[1:] * 0, logits)
    with pytest.raises(ValueError, match="valid"):
        engine.verify(tokens[1:] - 20, q[1:], logits)
