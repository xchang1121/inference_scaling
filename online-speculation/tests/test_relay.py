"""Conditional proposals, predictable stopping, gradients, and frozen-backbone replay."""

from itertools import product
from unittest.mock import patch

import pytest
import torch

from blockspec.decoding import generate_ar
from blockspec.execution import FixedShapeExecutor
from blockspec.model import Decoder, ModelConfig
from blockspec.relay import (RelayConfig, RelayFeedback, RelayHead, RelayLearner, generate_relay,
                             load_relay, relay_candidates, relay_loss, save_relay)
from blockspec.sampling import SamplingConfig, probabilities, verify_linear


def tiny():
    torch.manual_seed(21)
    model = Decoder(ModelConfig(vocab_size=7, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                head_dim=8, adapter_rank=2)).eval().requires_grad_(False)
    head = RelayHead(RelayConfig(7, 16, 3))
    return model, head


@pytest.mark.parametrize("sampling", [SamplingConfig(1.), SamplingConfig(.7, 3, .8), SamplingConfig()])
def test_zero_initialization_and_saved_conditional_rows(sampling):
    _, head = tiny()
    logits, hidden = torch.randn(4, 7), torch.randn(4, 16)
    draft = relay_candidates(head, logits, hidden, sampling=sampling)
    torch.testing.assert_close(draft.q, probabilities(logits[1:], sampling), rtol=0, atol=0)
    with torch.no_grad():
        head.projection.weight.normal_(0, .5)
    draft = relay_candidates(head, logits, hidden, sampling=sampling)
    expected = probabilities(head(logits[1:], draft.tokens[:-1]), sampling)
    torch.testing.assert_close(draft.q, expected)
    assert (draft.q.gather(1, draft.tokens[1:, None]) > 0).all()


def test_admission_occurs_before_current_draw_and_shrinks():
    _, head = tiny()
    logits, hidden = torch.randn(4, 7), torch.randn(4, 16)
    # Initial conditional confidence = 1/2: admit once (.5), then stop (.25 < .3).
    with patch("blockspec.relay.draw", wraps=__import__("blockspec.sampling", fromlist=["draw"]).draw) as draw:
        draft = relay_candidates(head, logits, hidden, sampling=SamplingConfig(1.), threshold=.3)
    assert len(draft.tokens) == 2 and draw.call_count == 1
    draft = relay_candidates(head, logits, hidden, threshold=1.)
    assert len(draft.tokens) == 1 and draft.q.shape == (0, 7)


@pytest.mark.parametrize("threshold", [0., .3, 1.])
@pytest.mark.parametrize("budget", [0, 1, 2, 3, 17])
def test_greedy_kv_budget_and_online_exactness(threshold, budget):
    model, head = tiny()
    with torch.no_grad():
        head.projection.weight.normal_(0, 1.)
    frozen = {k: v.clone() for k, v in model.state_dict().items()}
    learner = RelayLearner(head, interval=1)
    prompt = torch.tensor([[1, 4, 3]])
    expected = generate_ar(model, prompt, budget)
    result = generate_relay(model, head, prompt, budget, block_size=4, threshold=threshold, learner=learner)
    assert result.tokens == expected.tokens
    assert not learner.pending
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, frozen[key], atol=0, rtol=0)
    if threshold == 1.:
        assert result.proposed == 0
        assert result.verified_tokens == budget // 2


def test_terminal_capture_and_owned_executor_features():
    model, head = tiny()
    engine = FixedShapeExecutor(model, capacity=30, max_query=4, use_cuda_graph=False)
    engine.prepare([(n, False, None) for n in range(1, 5)] + [(n, True, 2) for n in range(2, 5)])
    inputs = torch.tensor([[1, 2, 3]])
    mask = torch.tensor([[False, True, True]])
    with torch.no_grad():
        actual = engine(inputs, adapter_mask=mask, return_cache=True, capture_layer=2)
        expected = model(inputs, adapter_mask=mask, return_cache=True, capture_layer=2)
        saved = actual[2].hidden.clone()
        torch.testing.assert_close(model._project(actual[2].hidden, None), actual[0])
        torch.testing.assert_close(actual[0], expected[0])
        engine(inputs.flip(-1), adapter_mask=mask, return_cache=True, capture_layer=2)
        torch.testing.assert_close(actual[2].hidden, saved, atol=0, rtol=0)
        result = generate_relay(model, head, inputs, 19, block_size=4, executor=engine, threshold=.3)
    assert result.tokens == generate_ar(model, inputs, 19).tokens


def test_head_gradients_teacher_detach_and_learning_transition():
    torch.manual_seed(9)
    head = RelayHead(RelayConfig(2, 3, 2)).double()
    previous = torch.tensor([0, 1] * 16)
    teacher = torch.tensor([[4., -4.], [-4., 4.]], dtype=torch.float64).repeat(16, 1).requires_grad_()
    logits = torch.zeros(32, 2, dtype=torch.float64, requires_grad=True)
    feedback = RelayFeedback(logits, torch.ones(32, 3, dtype=torch.float64), previous, teacher)
    initial = float(relay_loss(head, feedback)[1])
    learner = RelayLearner(head, lr=.05, interval=1)
    for _ in range(100):
        learner.observe(feedback)
    final = float(relay_loss(head, feedback)[1])
    assert initial > .49 and final < .02
    assert teacher.grad is None and logits.grad is None
    assert all(torch.isfinite(p).all() for p in head.parameters())
    assert learner.examples == 3200


def test_head_loss_numerical_gradient():
    _, head = tiny()
    head = head.double()
    with torch.no_grad():
        head.projection.weight.normal_(0, .02)
    previous = torch.tensor([0, 2, 4])
    feedback = RelayFeedback(torch.randn(3, 7).double(), torch.randn(3, 16).double(), previous,
                             torch.randn(3, 7).double())
    loss, _, _ = relay_loss(head, feedback, confidence_weight=0.)
    loss.backward()
    parameter = head.projection.weight
    expected = parameter.grad[0, 0].item()
    epsilon = 1e-5
    with torch.no_grad():
        original = parameter[0, 0].item()
        parameter[0, 0] = original + epsilon
        plus = relay_loss(head, feedback, confidence_weight=0.)[0].item()
        parameter[0, 0] = original - epsilon
        minus = relay_loss(head, feedback, confidence_weight=0.)[0].item()
        parameter[0, 0] = original
    assert abs(expected - (plus - minus) / (2 * epsilon)) < 1e-8


@pytest.mark.parametrize("scheduled", [False, True])
def test_exact_two_token_joint_with_conditional_proposal_and_stopping(scheduled):
    # Enumerate candidate draws, accept/reject branches, and correction draws.
    # If verification stops at one output, the next AR draw completes the pair.
    p0 = torch.tensor([.7, .3], dtype=torch.float64)
    p1 = torch.tensor([[.2, .8], [.6, .4]], dtype=torch.float64)
    q0 = torch.tensor([.4, .6], dtype=torch.float64)
    q1 = torch.tensor([[.85, .15], [.1, .9]], dtype=torch.float64)
    observed = torch.zeros(2, 2, dtype=torch.float64)
    for x in range(2):
        # Admission of position two can depend on x; position one's inclusion is fixed.
        admitted = 1 if scheduled and x == 1 else 2
        for y in range(2) if admitted == 2 else [0]:
            proposals = torch.tensor([x, y][:admitted])
            q = torch.stack([q0, q1[x]])[:admitted]
            p = torch.stack([p0, p1[x], p0])[:admitted + 1]
            mass = q0[x] * (q1[x, y] if admitted == 2 else 1.)
            alpha = torch.minimum(torch.ones(admitted, dtype=torch.float64),
                                  p[:admitted].gather(1, proposals[:, None]).flatten()
                                  / q.gather(1, proposals[:, None]).flatten())
            for flags in product([False, True], repeat=admitted):
                weight = mass.clone()
                uniforms = torch.empty(admitted, dtype=torch.float64)
                for i, accept in enumerate(flags):
                    weight *= alpha[i] if accept else 1 - alpha[i]
                    uniforms[i] = alpha[i] / 2 if accept else (1 + alpha[i]) / 2
                if weight == 0:
                    continue
                for z in range(2):
                    drawn = []
                    def categorical(distribution, generator=None, drawn=drawn, z=z):
                        drawn.append(distribution[z])
                        return torch.tensor(z)
                    with patch("blockspec.sampling.draw", side_effect=categorical):
                        result = verify_linear(proposals, q, p, acceptance_uniforms=uniforms)
                    branch_mass = weight * drawn[0]
                    if len(result.tokens) >= 2:
                        observed[result.tokens[0], result.tokens[1]] += branch_mass
                    else:
                        observed[result.tokens[0]] += branch_mass * p1[result.tokens[0]]
    torch.testing.assert_close(observed, p0[:, None] * p1, rtol=0, atol=1e-14)


def test_head_checkpoint_binding_and_exclusive_write(tmp_path):
    _, head = tiny()
    binding = {"base": "a" * 64, "adapter": "b" * 64}
    path = tmp_path / "relay.pt"
    save_relay(path, head, binding=binding, metadata={"test": True})
    restored, metadata = load_relay(path, binding=binding)
    assert metadata == {"test": True}
    for a, b in zip(head.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    with pytest.raises(FileExistsError):
        save_relay(path, head, binding=binding)
    with pytest.raises(ValueError, match="binding"):
        load_relay(path, binding={**binding, "adapter": "c" * 64})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph integration")
def test_cuda_graph_terminal_capture_online():
    model, head = tiny()
    model, head = model.cuda(), head.cuda()
    engine = FixedShapeExecutor(model, capacity=32, max_query=4)
    engine.prepare([(n, False, None) for n in range(1, 5)] + [(n, True, 2) for n in range(2, 5)])
    prompt = torch.tensor([[1, 2, 5]], device="cuda")
    learner = RelayLearner(head, interval=1)
    result = generate_relay(model, head, prompt, 21, block_size=4, executor=engine, learner=learner)
    assert result.tokens == generate_ar(model, prompt, 21, executor=engine).tokens
    assert result.updates > 0 and result.update_seconds > 0
