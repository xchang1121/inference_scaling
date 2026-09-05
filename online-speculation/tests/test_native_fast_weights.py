"""CPU math tests; CUDA kernel/integration checks run explicitly in WSL."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
spec = importlib.util.spec_from_file_location(
    "native_fast_weights", Path(__file__).resolve().parents[1] / "scripts/native_fast_weights.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def state():
    generator = torch.Generator().manual_seed(42)
    return module.FastWeights(12, 8, torch.ones(8),
                              torch.randn(20, 8, generator=generator), rank=2, max_block=8)


@pytest.mark.parametrize("committed,rows", [(1, 0), (2, 1), (3, 2), (8, 7), (9, 7)])
def test_alignment_and_rejected_tail_mask(committed, rows):
    assert module.eligible_rows(8, committed) == rows


@pytest.mark.parametrize("block,committed", [(1, 1), (8, 0), (8, 10)])
def test_bad_cycle_rejected(block, committed):
    with pytest.raises(ValueError):
        module.eligible_rows(block, committed)


def test_zero_init_and_feature_replay_gradient_and_reset():
    s = state()
    torch.manual_seed(314)
    s.features.normal_()
    s.before_delta.normal_()
    s.residual.normal_()
    frozen = (s.head_weight.clone(), s.norm_weight.clone())
    a, u, r = s.features[1:4], s.before_delta[1:4], s.residual[1:4]
    with torch.no_grad():
        start = s.logits(a, u, r)
        summed = u + r
        expected = (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-6)) @ s.head_weight.T
    torch.testing.assert_close(start, expected)
    # Teacher must disagree; no zero-A + zero-B dead initialization.
    s.last_teacher = torch.roll(start, 1, -1).clone()
    s.last_draft = start.clone()
    addresses = s.serve_left.data_ptr(), s.serve_right.data_ptr()
    s.update(3, audit=True)
    assert s.version == 1
    assert s.events[0]["grad_norm"] > 0
    assert s.events[0]["same_features_logit_change"] > 0
    assert s.events[0]["kl_after"] < s.events[0]["kl_before"]
    assert addresses == (s.serve_left.data_ptr(), s.serve_right.data_ptr())
    assert torch.equal(frozen[0], s.head_weight) and torch.equal(frozen[1], s.norm_weight)
    s.reset()
    assert s.version == 0 and not s.events
    assert not torch.count_nonzero(s.serve_right)
    torch.testing.assert_close(s.logits(a, u, r), start)


def test_replay_backprop_matches_direct_formula():
    torch.manual_seed(9)
    a, u, r = torch.randn(3, 12), torch.randn(3, 8), torch.randn(3, 8)
    left = torch.randn(2, 12, requires_grad=True)
    right = torch.randn(8, 2, requires_grad=True)
    actual = module.replay_hidden(a, u, r, left, right, torch.ones(8), 1e-6, 2)
    combined = (u + (a @ left.T) @ right.T + r).reshape(3, 2, 4)
    expected = (combined * torch.rsqrt(combined.square().mean(-1, keepdim=True) + 1e-6)).reshape(3, 8)
    grads = torch.autograd.grad(actual.square().sum(), (left, right), retain_graph=True)
    reference = torch.autograd.grad(expected.square().sum(), (left, right))
    torch.testing.assert_close(actual, expected)
    for grad, ref in zip(grads, reference):
        torch.testing.assert_close(grad, ref)


def test_serving_gate_keeps_seed_and_verifier_unchanged():
    s = state()
    down = torch.nn.Linear(12, 8, bias=False)
    down.requires_grad_(False)
    norm = SimpleNamespace(forward=lambda x, residual: (x + residual, residual))
    model = SimpleNamespace(model=SimpleNamespace(
        layers=[SimpleNamespace(mlp=SimpleNamespace(down_proj=down))], norm=norm))
    context = SimpleNamespace(lora_enabled=False, lora_mask=torch.tensor([0., 1., 1., 1.]))
    a, residual = torch.randn(4, 12), torch.randn(4, 8)
    reference = down(a).clone()
    frozen = down.weight.clone()
    s.attach(model, lambda: context)
    with torch.no_grad():
        s.serve_right.fill_(0.25)
        assert torch.equal(down(a), reference)  # OFF verifier
        context.lora_enabled = True
        draft = down(a)
        assert torch.equal(draft[0], reference[0])
        assert not torch.equal(draft[1:], reference[1:])
        norm.forward(draft, residual)
        assert torch.equal(s.features[:4], a)
        assert torch.equal(s.before_delta[:4], reference)
        assert torch.equal(s.residual[:4], residual)
        saved = s.features.clone()
        context.lora_enabled = False
        assert torch.equal(down(a * 2), reference * 2)
        assert torch.equal(s.features, saved)  # verify never overwrites draft cache
        assert torch.equal(down.weight, frozen)


def test_wrapper_publishes_only_after_commit_and_skips_final_update():
    events = []
    fast = SimpleNamespace(stride=1, serve_right=torch.zeros(8, 2))

    def reset():
        fast.version, fast.cycles = 0, 0

    def update(rows, **kwargs):
        assert events[-1][0] == "commit"
        fast.version += 1
        events.append(("update", rows))

    fast.reset, fast.update = reset, update
    fast.snapshot = lambda: {"version": fast.version}

    class Engine:
        def __init__(self):
            self.scheduler = SimpleNamespace(waiting=[], running=[])
            decoder = SimpleNamespace(run_block=lambda seqs, tokens, **kw: torch.zeros(1, 8, 20))
            self.model_runner = SimpleNamespace(fast_weights=fast, two_pass_decoder=decoder)

        def step(self):
            if self.scheduler.waiting:
                self.scheduler.waiting = []
                self.scheduler.running = [1]
                return [], 1
            version = fast.version
            decoder = self.model_runner.two_pass_decoder
            decoder.run_block([], None, lora_mask_batch=torch.ones(1, 8))
            decoder.run_block([], None)
            assert fast.version == version
            events.append(("commit", version))
            if fast.cycles == 2:
                self.scheduler.running = []
            return [], -3

        def generate(self, *args, **kwargs):
            self.scheduler.waiting = [1]
            self.step()
            self.step()
            self.step()
            return [{"token_ids": [1] * 7}]

    engine = Engine()
    original_step, original_block = engine.step, engine.model_runner.two_pass_decoder.run_block
    _, audit = module.generate_fast(engine, [0], SimpleNamespace(diffusion_block_size=8), 7)
    assert events == [("commit", 0), ("update", 2), ("commit", 1)]
    assert audit["version"] == 1
    assert engine.step == original_step and engine.model_runner.two_pass_decoder.run_block == original_block


@pytest.mark.skipif(sys.platform != "linux" or not torch.cuda.is_available(), reason="WSL CUDA/Triton test")
@pytest.mark.parametrize("rows,groups,has_residual", [(1, 1, False), (8, 1, True), (53, 3, True), (16, 2, False)])
def test_fused_norm_cast_points(rows, groups, has_residual):
    pytest.importorskip("triton")
    norm_spec = importlib.util.spec_from_file_location(
        "native_norm", Path(__file__).resolve().parents[1] / "scripts/native_norm.py")
    norm = importlib.util.module_from_spec(norm_spec)
    norm_spec.loader.exec_module(norm)
    torch.manual_seed(123)
    x = torch.randn(rows, 1536, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x) if has_residual else None
    layer = SimpleNamespace(hidden_size=1536, group_size=1536 // groups,
                            eps=1e-6, weight=torch.randn(1536, device="cuda", dtype=x.dtype))
    with torch.inference_mode():
        result = norm.fused_grouped_rms(layer, x, residual)
        summed = (x.float() + residual.float()).to(x.dtype) if has_residual else x
        grouped = summed.float().reshape(rows, groups, -1)
        expected = (grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6))
        expected = expected.reshape_as(x).to(x.dtype) * layer.weight
        actual = result[0] if has_residual else result
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=0.016)
        if has_residual:
            assert torch.equal(result[1], summed)
