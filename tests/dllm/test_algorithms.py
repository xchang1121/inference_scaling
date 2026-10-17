from __future__ import annotations

import itertools
import math
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
from inference_scaling.shared.sampling.conditional_is import RolloutEvaluation, select
from inference_scaling.shared.sampling.importance import normalize_log_weights
from inference_scaling.shared.sampling.mh import decide_metropolis_hastings
from inference_scaling.shared.types import pointwise

# (candidate_canvas, kept_sequence) of every IS variant.
MODES = [("block", False), ("full", False), ("full", True)]
SUM = pointwise(lambda _prompt, continuation: float(sum(continuation)))


class TinyMaskedModel(torch.nn.Module):
    def __init__(self, bias, name):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        self.config = SimpleNamespace(_name_or_path=name, mask_token_id=3)

    def forward(self, token_ids):
        batch, length = token_ids.shape
        return SimpleNamespace(logits=self.bias.view(1, 1, -1).expand(batch, length, -1).clone())


def _backend(bias, name, eos=None):
    return LLaDATransformersBackend(TinyMaskedModel(bias, name), SimpleNamespace(eos_token_id=eos), mask_token_id=3,
                                    max_batch_size=64)


def _sampling(block_length):
    return DiffusionSamplingConfig(block_length=block_length, steps_per_block=block_length, temperature=1.0,
                                   remasking="random", top_k=0, top_p=1.0)


def _config(mode, **counts):
    canvas, kept = mode
    return DiffusionISConfig(**{"reward_temperature": 1.0, **counts}, candidate_canvas=canvas, kept_sequence=kept)


@pytest.mark.parametrize("mode", MODES)
def test_conditional_is_decision_block_can_span_native_diffusion_blocks(mode):
    result = run_conditional_diffusion_is(
        backend=_backend((0.0, 0.5, 1.0, -2.0), "base"), prompt=(0,), sampling=_sampling(2), reward=SUM, seed=17,
        config=_config(mode, candidate_count=2, rollout_count=1, block_size=4, total_length=8))
    assert len(result.steps) == 2
    assert all(candidate.token_count == 4 for step in result.steps for candidate in step.candidates)
    assert len(result.token_ids) == 8


@pytest.mark.parametrize("mode", MODES)
def test_conditional_is_ends_when_it_selects_a_block_of_eos(mode):
    config = _config(mode, candidate_count=6, rollout_count=2, block_size=2, total_length=8)
    mixed = False
    for seed in range(8):
        # EOS (1) and 2 are equally likely, so a candidate block is all EOS with probability 1/4.
        result = run_conditional_diffusion_is(backend=_backend((-9.0, 0.0, 0.0, -2.0), "eos", eos=1), prompt=(0,),
                                              config=config, sampling=_sampling(2), reward=SUM, seed=seed)
        for step in result.steps:
            final = step.generated_length_before + 2 == config.total_length
            for candidate in step.candidates:
                # A finished candidate has one empty completion; the others have K non-empty ones.
                terminal = final or candidate.rollouts[0].token_count == 0
                assert [bool(item.token_count) for item in candidate.rollouts] == ([False] if terminal else [True, True])
            mixed |= len({len(candidate.rollouts) for candidate in step.candidates}) > 1
        assert all(result.token_ids[index:index + 2] != (1, 1) for index in range(0, len(result.token_ids) - 2, 2))
        assert result.steps[-1].selected.rollouts[0].reward == sum(result.token_ids)
    assert mixed


def test_conditional_is_rejects_decision_block_that_splits_native_block():
    with pytest.raises(ValueError, match="divisible by block_length"):
        run_conditional_diffusion_is(
            backend=_backend((0.0, 0.5, 1.0, -2.0), "base"), prompt=(0,), sampling=_sampling(4), reward=SUM, seed=17,
            config=_config(MODES[0], candidate_count=2, rollout_count=1, block_size=6, total_length=12))


def test_a_kept_sequence_needs_full_canvas_candidates():
    with pytest.raises(ValueError, match="candidate_canvas 'full'"):
        _config(("block", True), candidate_count=2, rollout_count=1, block_size=1, total_length=2)
    with pytest.raises(ValueError, match="candidate_canvas"):
        _config(("whole", False), candidate_count=2, rollout_count=1, block_size=1, total_length=2)


class CanvasBackend:
    """Tokens 0/1 from per-context tables: ``full`` when the canvas reaches ``end``, ``short`` otherwise.

    It stands for a dLLM that reads the masked positions after a block, so a block
    requested alone follows another distribution than inside a complete output.
    """

    model_id = "canvas"

    def __init__(self, full, short, prompt_length, end):
        self.full, self.short, self.prompt_length, self.end = full, short, prompt_length, end
        self.requests: list = []

    def sample_batch(self, requests):
        samples = []
        for request in requests:
            self.requests.append(request)
            rng = np.random.default_rng(request.seed)
            table = self.full if len(request.prefix) + request.generation_length == self.end else self.short
            tokens: tuple[int, ...] = ()
            for _ in range(request.generation_length):
                tokens += (int(rng.random() < table[request.prefix[self.prompt_length:] + tokens]),)
            samples.append(DiffusionSample(request.prefix, tokens, (), None, request.sampling.policy_id, self.model_id,
                                           request.request_id))
        return samples


def _tables(rng, length):
    contexts = [context for size in range(length) for context in itertools.product((0, 1), repeat=size)]
    full = {context: float(rng.uniform(0.15, 0.85)) for context in contexts}
    # Alone, a block leans the other way.
    return full, {context: 1 - value for context, value in full.items()}


def test_full_canvas_candidates_and_a_kept_sequence_leave_the_target_invariant(monkeypatch):
    states = list(itertools.product((0, 1), repeat=2))
    full, short = _tables(np.random.default_rng(4), 2)
    base = {y: math.prod(full[y[:i]] if token else 1 - full[y[:i]] for i, token in enumerate(y)) for y in states}
    target = np.asarray([base[y] * math.exp(sum(y)) for y in states])
    target /= target.sum()
    config = DiffusionISConfig(candidate_count=2, rollout_count=2, block_size=1, total_length=2,
                               reward_temperature=1, candidate_canvas="full", kept_sequence=True)
    # Inspect one actual step and integrate its two categorical selections exactly.
    monkeypatch.setattr("inference_scaling.dllm.algorithms.is_sampling.diffusion_decision_stage_lengths", lambda **_: (1,))
    captured = []
    def inspect(candidates, *args):
        captured[:] = candidates
        return select(candidates, *args)
    monkeypatch.setattr("inference_scaling.dllm.algorithms.is_sampling.select", inspect)

    class Draws(CanvasBackend):
        def sample_batch(self, requests):
            assert all(len(request.prefix) + request.generation_length == 3 for request in requests)
            return [DiffusionSample(request.prefix,
                                    (c, d) if request.request_id.endswith(":candidate:1") else
                                    (e if ":candidate:0:" in request.request_id else f,), (), None,
                                    request.sampling.policy_id, self.model_id, request.request_id) for request in requests]

    matrix = np.zeros((4, 4))
    for row, y in enumerate(states):
        for c, d, e, f in itertools.product((0, 1), repeat=4):
            run_conditional_diffusion_is(
                backend=Draws(full, short, 1, 3), prompt=(7,), config=config, sampling=_sampling(1), seed=1,
                reward=SUM, start=RolloutEvaluation(y, sum(y), sum(y)))
            density = base[c, d] * (full[y[:1]] if e else 1 - full[y[:1]]) * (full[c,] if f else 1 - full[c,])
            for candidate, probability in zip(captured, normalize_log_weights([item.log_weight for item in captured])):
                for leaf, conditional in zip(candidate.rollouts, normalize_log_weights([item.log_weight for item in candidate.rollouts])):
                    matrix[row, states.index(candidate.token_ids + leaf.token_ids)] += density * probability * conditional
    np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=1e-13, rtol=0)
    np.testing.assert_allclose(target @ matrix, target, atol=1e-13, rtol=0)


@pytest.mark.parametrize("mode", MODES)
def test_candidate_canvas_and_kept_sequence_set_the_requests_and_reuse_the_kept_reward(mode):
    full, short = _tables(np.random.default_rng(0), 3)
    backend = CanvasBackend(full, short, 1, 4)
    result = run_conditional_diffusion_is(backend=backend, prompt=(7,), sampling=_sampling(1), reward=SUM, seed=5,
                                          config=_config(mode, candidate_count=3, rollout_count=2, block_size=1,
                                                         total_length=3))
    candidates = [request for request in backend.requests if request.request_id.split(":")[-2] == "candidate"]
    rollouts = [request for request in backend.requests if "rollout" in request.request_id]
    canvas, kept = mode
    # A block alone, or a complete output of the remaining canvas cut at the block.
    assert {len(request.prefix) + request.generation_length for request in candidates} == (
        {2, 3, 4} if canvas == "block" else {4})
    # Three steps: a full-canvas candidate's rest is its first completion, and the kept sequence continues as candidate 0.
    assert len(candidates) == 9 - 2 * kept and len(rollouts) == 2 * 3 * (2 if canvas == "block" else 1)
    assert [step.rollout_evaluations_performed for step in result.steps] == (
        [6, 6, 3] if not kept else [6, 5, 2])
    assert [step.retained_candidate for step in result.steps] == [False, kept, kept]
    assert all((step.completion_index is not None) == kept for step in result.steps)


def test_independence_mh_preserves_base_times_reward_target_without_scores(monkeypatch):
    matrix = np.zeros((2, 2))
    decisions = []
    def inspect(**kwargs):
        decision = decide_metropolis_hastings(**kwargs)
        decisions.append(decision)
        return decision
    monkeypatch.setattr("inference_scaling.dllm.algorithms.mh.decide_metropolis_hastings", inspect)

    class Draws:
        model_id = "coin"
        def sample_batch(self, requests):
            return [DiffusionSample(request.prefix, (value,), (), None, request.sampling.policy_id,
                                    self.model_id, request.request_id)
                    for request, value in zip(requests, (current, proposed), strict=True)]

    for current, proposed in itertools.product((0, 1), repeat=2):
        run_diffusion_reward_mh(
            backend=Draws(), prompt=(), config=DiffusionMHConfig(total_length=1, updates=1, reward_temperature=1),
            sampling=_sampling(1), reward=SUM, seed=1)
        acceptance = math.exp(decisions[-1].log_acceptance)
        matrix[current, proposed] += .5 * acceptance
        matrix[current, current] += .5 * (1 - acceptance)
    target = np.asarray([1, np.e]) / (1 + np.e)
    np.testing.assert_allclose(matrix.sum(axis=1), 1, atol=1e-13, rtol=0)
    np.testing.assert_allclose(target @ matrix, target, atol=1e-13, rtol=0)
