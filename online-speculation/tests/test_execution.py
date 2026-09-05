"""Padded execution is equivalent and never leaks mutable workspaces to replay."""

import copy

import pytest
import torch

from blockspec.decoding import generate_ar, generate_speculative
from blockspec.execution import FixedShapeExecutor
from blockspec.model import Decoder, ModelConfig, PackedCache, trim_cache
from blockspec.online import Feedback, OnlineConfig, OnlineLearner
from blockspec.tree import CandidateTree, compact_tree_cache, generate_tree


def example(device="cpu"):
    torch.manual_seed(135)
    model = Decoder(ModelConfig(vocab_size=13, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=3, num_attention_heads=2, num_key_value_heads=1,
                                head_dim=8, adapter_rank=2)).to(device).train_adapters_only()
    with torch.no_grad():
        for p in model.adapter_parameters():
            p.normal_(std=.1)
    engine = FixedShapeExecutor(model, capacity=48, max_query=5, use_cuda_graph=device == "cuda")
    engine.prepare([(i, False, None) for i in range(1, 6)] +
                   [(i, True, c) for i in range(2, 5) for c in (None, 2)])
    return model, engine


def close_cache(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph hardware required"))])
@torch.no_grad()
def test_padded_execution_prefill_cache_tree_and_owned_outputs(device):
    model, engine = example(device)
    ids = torch.tensor([[1, 4, 2, 9, 8, 7]], device=device)
    _, past = engine(ids, return_cache=True)  # long prefill is eager
    _, reference_past = model(ids, return_cache=True)
    seed_noise = ids[:, :4]
    mask = torch.tensor([[False, True, True, True]], device=device)
    actual, cache, boundary = engine(seed_noise, cache=past, adapter_mask=mask,
                                      return_cache=True, capture_layer=2)
    expected, expected_cache, expected_boundary = model(seed_noise, cache=reference_past,
                                                        adapter_mask=mask, return_cache=True, capture_layer=2)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    close_cache(cache, expected_cache)
    torch.testing.assert_close(boundary.hidden, expected_boundary.hidden, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(boundary.allowed, expected_boundary.allowed)
    kept = (actual.clone(), cache.packed.clone(), boundary.hidden.clone(), boundary.allowed.clone())
    # Reuse the SAME signature with a different history length and adapter.
    for p in model.adapter_parameters():
        p.add_(.03)
    engine(seed_noise.flip(-1), cache=trim_cache(cache, 3), adapter_mask=mask,
           return_cache=True, capture_layer=2)
    torch.testing.assert_close((actual, cache.packed, boundary.hidden, boundary.allowed), kept, atol=0, rtol=0)
    # Stale workspace entries beyond a shorter valid prefix are masked out.
    shorter = trim_cache(past, 2)
    result = engine(seed_noise, cache=shorter, adapter_mask=mask)
    torch.testing.assert_close(result, model(seed_noise, cache=shorter, adapter_mask=mask), rtol=2e-5, atol=2e-6)
    # Tree siblings occupy equal positions but cannot see each other.
    tree = CandidateTree([3, 4, 7, 8, 9], [-1, 0, 0, 1, 2], [0, 1, 1, 2, 2],
                         [0.] * 5, [{4: 1, 7: 2}, {8: 3}, {9: 4}, {}, {}])
    tokens, positions, allowed = tree.layout(2, device=device)
    actual, verified = engine(tokens, positions=positions, allowed=allowed, cache=shorter, return_cache=True)
    expected, expected_cache = model(tokens, positions=positions, allowed=allowed, cache=shorter, return_cache=True)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    compact = compact_tree_cache(verified, 2, [0, 2, 4])
    assert isinstance(compact, PackedCache) and isinstance(trim_cache(compact, 3), PackedCache)
    close_cache(compact, compact_tree_cache(expected_cache, 2, [0, 2, 4]))


@pytest.mark.parametrize("generate", [generate_speculative, generate_tree])
@pytest.mark.parametrize("last_layers", [None, 1])
def test_padded_online_is_real_continuation_with_immutable_feedback(generate, last_layers):
    model, engine = example()
    oracle = copy.deepcopy(model)
    options = dict(stride=1, replay_blocks=2, train_last_layers=last_layers, learning_rate=.001)
    learner, reference = OnlineLearner(model, OnlineConfig(**options)), OnlineLearner(oracle, OnlineConfig(**options))
    prompt = torch.tensor([[1, 3, 5]])
    kwargs = {"block_size": 4}
    if generate is generate_tree:
        kwargs.update(top_k=2, prefix_budget=5)
    expected = generate(oracle, prompt, 20, learner=reference,
                        generator=torch.Generator().manual_seed(27), **kwargs)
    actual = generate(model, prompt, 20, learner=learner, executor=engine,
                      generator=torch.Generator().manual_seed(27), **kwargs)
    assert actual.tokens == expected.tokens == generate_ar(model, prompt, 20, executor=engine).tokens
    assert actual.updates == expected.updates > 0
    for p, q in zip(model.parameters(), oracle.parameters()):
        torch.testing.assert_close(p, q, rtol=3e-4, atol=2e-6)
    assert not learner.replay


def test_feedback_survives_slot_reuse_before_a_later_backward():
    model, engine = example()
    learner = OnlineLearner(model, OnlineConfig(train_last_layers=1, stride=100))
    inputs = torch.tensor([[1, 2, 3, 4]])
    mask = torch.tensor([[False, True, True, True]])
    with torch.no_grad():
        _, past = engine(inputs, return_cache=True)
        logits, _, boundary = engine(inputs, cache=past, adapter_mask=mask,
                                     return_cache=True, capture_layer=2)
        learner.observe(Feedback(inputs, past, logits[0, 1:], 3, boundary), may_update=False)
        expected = model.forward_suffix(learner.replay[0].boundary, cache=learner.replay[0].cache).clone()
        engine(inputs.flip(-1), cache=past, adapter_mask=mask, return_cache=True, capture_layer=2)
        engine(inputs.flip(-1), return_cache=True)
        actual = model.forward_suffix(learner.replay[0].boundary, cache=learner.replay[0].cache)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert learner.update()["version"] == 1


def test_executor_contracts_and_storage_invalidation():
    model, engine = example()
    ids = torch.tensor([[1, 2]])
    with pytest.raises(RuntimeError, match="inference-only"):
        engine(ids)
    with torch.no_grad(), pytest.raises(ValueError, match="unprepared"):
        engine(ids, adapter_mask=torch.ones_like(ids), return_cache=True, capture_layer=1)
    with torch.no_grad(), pytest.raises(ValueError, match="layout"):
        engine(ids, positions=torch.tensor([[1]]))
    with torch.no_grad(), pytest.raises(ValueError, match="layout"):
        engine(ids, adapter_mask=torch.full_like(ids, .5, dtype=torch.float32))
    with torch.no_grad(), pytest.raises(ValueError, match="batch-one"):
        engine(ids.float())
    with pytest.raises(ValueError, match="share a model"):
        engine.validate(copy.deepcopy(model))
    with pytest.raises(ValueError, match="FP32 CUDA"):
        FixedShapeExecutor(model, capacity=4, max_query=2)
    p = model.adapter_parameters()[0]
    p.data = p.data.clone()
    with pytest.raises(RuntimeError, match="storage changed"):
        engine.validate(model)


def test_mutated_rotary_buffer_invalidates_captured_execution():
    model, engine = example()
    model.model.layers[0].self_attn.frequencies.add_(.001)
    with torch.no_grad(), pytest.raises(RuntimeError, match="buffers changed"):
        engine(torch.tensor([[1, 2]]))


def test_all_setup_cost_is_not_hidden_in_warm_throughput():
    from blockspec.benchmark import aggregate
    row = {"tokens": 30, "seconds": 2., "decode_forwards": 10, "rounds": 5,
           "accepted": 10, "proposed": 12, "updates": 1, "update_seconds": .1}
    result = aggregate([row], setup_seconds=.5, engine_setup_seconds=1.5)
    assert result["tps"] == 15 and result["tps_including_learner_setup"] == 12
    assert result["tps_including_all_setup"] == 7.5
