from __future__ import annotations

import itertools
import math
from collections import Counter
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
from inference_scaling.shared.sampling.conditional_is import RolloutEvaluation
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
    assert all(len(candidate.token_ids) == 4 for step in result.steps for candidate in step.candidates)
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
                terminal = final or candidate.token_ids == (1, 1)
                assert [bool(item.token_ids) for item in candidate.rollouts] == ([False] if terminal else [True, True])
            mixed |= len({len(candidate.rollouts) for candidate in step.candidates}) > 1
        assert all(step.selected.token_ids != (1, 1) for step in result.steps[:-1])
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


def test_full_canvas_candidates_and_a_kept_sequence_leave_the_target_invariant():
    rng, length, runs = np.random.default_rng(4), 3, 3000
    full, short = _tables(rng, length)
    outcomes = list(itertools.product((0, 1), repeat=length))
    score = dict(zip(outcomes, rng.normal(0.0, 1.0, len(outcomes))))
    base = {y: math.prod(full[y[:i]] if token else 1 - full[y[:i]] for i, token in enumerate(y)) for y in outcomes}
    target = {y: base[y] * math.exp(score[y] / 0.5) for y in outcomes}
    target = {y: value / sum(target.values()) for y, value in target.items()}
    backend = CanvasBackend(full, short, 1, 1 + length)
    config = DiffusionISConfig(candidate_count=2, rollout_count=2, block_size=1, total_length=length,
                               reward_temperature=0.5, candidate_canvas="full", kept_sequence=True)

    def distance(start):
        counts: Counter = Counter()
        for seed in range(runs):
            y = outcomes[rng.choice(len(outcomes), p=[start[item] for item in outcomes])]
            counts[run_conditional_diffusion_is(
                backend=backend, prompt=(7,), config=config, sampling=_sampling(1), seed=seed,
                reward=pointwise(lambda _prompt, sequence: score[sequence]),
                start=RolloutEvaluation(y, score[y], score[y] / 0.5)).token_ids] += 1
        return 0.5 * sum(abs(counts[y] / runs - target[y]) for y in outcomes)

    # Started at the target, every step keeps it; started at the base policy, three steps do not reach it.
    assert distance(target) < 0.04 < 0.08 < distance(base)
    assert all(len(request.prefix) + request.generation_length == 1 + length for request in backend.requests)


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


def _empty_sample(value: int, request_id: str) -> DiffusionSample:
    return DiffusionSample(prefix=(), token_ids=(value,), trace=(), trajectory_logprob=None, policy_id="uniform",
                           model_id="coin", request_id=request_id)


class CoinBackend:
    model_id = "coin"

    def sample_batch(self, requests):
        return [_empty_sample(int(np.random.default_rng(request.seed).integers(0, 2)), request.request_id)
                for request in requests]


def test_independence_mh_approaches_base_times_reward_target_without_scores():
    ones, runs = 0, 2500
    for seed in range(runs):
        result = run_diffusion_reward_mh(
            backend=CoinBackend(), prompt=(), config=DiffusionMHConfig(total_length=1, updates=8, reward_temperature=1.0),
            sampling=DiffusionSamplingConfig(block_length=1, steps_per_block=1, temperature=0.0, top_k=0, top_p=1.0, remasking="low_confidence"),
            reward=pointwise(lambda _prompt, continuation: float(continuation[0])), seed=seed)
        ones += result.final.token_ids[0]
    assert ones / runs == pytest.approx(np.e / (1.0 + np.e), abs=0.03)
