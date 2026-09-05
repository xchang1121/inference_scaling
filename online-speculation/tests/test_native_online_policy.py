from __future__ import annotations

from collections import deque
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


MODULE = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "native_online_policy.py"))
Policy = MODULE["NativeWidthPolicy"]
generate_online = MODULE["generate_online"]


def test_two_cycles_per_epoch_and_actual_feedback_only():
    policy = Policy()
    selected = []
    for _ in range(6):
        width, reason = policy.choose()
        selected.append(width)
        assert reason == "initial_probe"
        policy.observe(width, 3, 0.01)
    assert selected == [8, 8, 4, 4, 16, 16]
    assert policy.counts == {4: 1, 8: 1, 16: 1}
    assert policy.completed_epochs == 3
    assert policy.choose() == (8, "guarded_exploit")


def test_learns_tokens_per_time_instead_of_acceptance_alone():
    policy = Policy(epoch_cycles=1)
    for _ in range(3):
        width, _ = policy.choose()
        tokens, seconds = {4: (3, 0.001), 8: (4, 0.002), 16: (9, 0.006)}[width]
        policy.observe(width, tokens, seconds)
    assert policy.choose() == (4, "guarded_exploit")


def test_margin_keeps_anchor_and_refreshes_are_bounded():
    policy = Policy(epoch_cycles=1, probe_every=4)
    reasons = []
    for _ in range(15):
        width, reason = policy.choose()
        reasons.append(reason)
        policy.observe(width, 2, 0.01 if width != 4 else 0.0099)
        if reason == "guarded_exploit":
            assert width == 8
    assert reasons.count("initial_probe") == 3
    assert reasons.count("refresh_probe") == 3


def test_pending_and_feedback_attribution_are_enforced():
    policy = Policy()
    width, _ = policy.choose()
    with pytest.raises(RuntimeError, match="previous"):
        policy.choose()
    with pytest.raises(RuntimeError, match="pending"):
        policy.observe(4, 3, 0.01)
    assert policy.pending == width
    policy.observe(width, 3, 0.01)
    with pytest.raises(RuntimeError, match="pending"):
        policy.observe(width, 3, 0.01)


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_bad_measurement_does_not_mutate_statistics(seconds):
    policy = Policy()
    width, _ = policy.choose()
    before = policy.snapshot()
    with pytest.raises(ValueError):
        policy.observe(width, 2, seconds)
    assert policy.snapshot() == before


def test_ema_has_convex_bounds_and_new_request_has_no_history():
    policy = Policy(widths=(8,), epoch_cycles=1)
    for tokens, seconds in [(1, 0.01), (9, 0.1), (4, 0.05)]:
        width, _ = policy.choose()
        policy.observe(width, tokens, seconds)
        assert 1 <= policy.mean_tokens[8] <= 9
        assert 0.01 <= policy.mean_seconds[8] <= 0.1
    assert Policy().completed_epochs == 0


class FakeEngine:
    def __init__(self, *, fail=False):
        self.config = SimpleNamespace(max_num_seqs=1, tree_verify_size=None,
                                      enforce_eager=False, tensor_parallel_size=1,
                                      cuda_graph_block_sizes=[1, 4, 8, 16], max_diffusion_block_size=16)
        self.scheduler = SimpleNamespace(waiting=deque())
        self.active = False
        self.calls = []
        self.fail = fail

    def is_finished(self):
        return not self.active

    def step(self):
        if self.scheduler.waiting:
            self.scheduler.waiting.clear()
            return [], 10
        self.calls.append(self.params.diffusion_block_size)
        if self.fail:
            raise RuntimeError("kernel failure")
        self.tokens.extend([len(self.tokens) + 1, len(self.tokens) + 2])
        self.active = len(self.tokens) < self.budget
        return [], -2

    def generate(self, prompts, params, *, use_tqdm, request_max_tokens):
        assert len(prompts) == 1 and use_tqdm is False
        self.params, self.tokens, self.budget = params, [], request_max_tokens[0]
        self.active = True
        self.scheduler.waiting.append(prompts[0])
        while self.active:
            self.step()
        return [{"token_ids": self.tokens.copy(), "stats": {}, "text": "official finalize"}]


def test_wrapper_preserves_generate_output_and_restores_state():
    engine = FakeEngine()
    params = SimpleNamespace(diffusion_block_size=8)
    output, diagnostics = generate_online(engine, [1, 2], params, 20, Policy())
    assert output == {"token_ids": list(range(1, 21)), "stats": {}, "text": "official finalize"}
    assert engine.calls[:6] == [8, 8, 4, 4, 16, 16]
    assert len(diagnostics["cycles"]) == 10  # prefill excluded
    assert sum(diagnostics["policy"]["total_tokens"].values()) == 20
    assert diagnostics["policy"]["pending"] is None
    assert params.diffusion_block_size == 8 and "step" not in engine.__dict__


def test_error_restores_original_step_and_width():
    engine = FakeEngine(fail=True)
    params = SimpleNamespace(diffusion_block_size=1)
    with pytest.raises(RuntimeError, match="kernel failure"):
        generate_online(engine, [1], params, 20, Policy())
    assert params.diffusion_block_size == 1 and "step" not in engine.__dict__


def test_uncaptured_shape_or_concurrent_request_is_rejected():
    engine = FakeEngine()
    params = SimpleNamespace(diffusion_block_size=8)
    engine.config.cuda_graph_block_sizes = [1, 8]
    with pytest.raises(ValueError, match="graph"):
        generate_online(engine, [1], params, 20, Policy())
    engine.active = True
    with pytest.raises(ValueError, match="idle"):
        generate_online(engine, [1], params, 20, Policy())


def test_shadow_single_width_policy_has_identical_decisions_to_static():
    params = SimpleNamespace(diffusion_block_size=8)
    static = FakeEngine().generate([[1]], params, use_tqdm=False, request_max_tokens=[20])[0]
    engine = FakeEngine()
    online, _ = generate_online(engine, [1], params, 20, Policy(widths=(8,)))
    assert static == online and engine.calls == [8] * 10
