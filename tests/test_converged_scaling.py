"""Tests for the converged engine/worker, reward refinements, adaptive IS, and serve layer."""

from __future__ import annotations

from collections import namedtuple
import copy
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from inference_scaling.app.settings import SettingsError, load_settings, validate_settings
from inference_scaling.arllm.algorithms.conditional_is import run_conditional_is
from inference_scaling.arllm.algorithms.config import ConditionalISConfig
from inference_scaling.arllm.backends.capabilities import EngineCapabilities, backend_capabilities
from inference_scaling.arllm.backends import loader
from inference_scaling.arllm.backends.openai_backend import OpenAIBackend
from inference_scaling.arllm.backends.scheduling import SchedulerConfig
from inference_scaling.arllm.backends.statistics import StatisticRecorder
from inference_scaling.arllm.backends.vllm_backend import VLLMBackend, _engine_worker
from inference_scaling.arllm.backends.vllm_statistics import (
    SCALAR_STATISTICS_KEY,
    install_scalar_statistics,
    scalar_generated_logprobs,
)
from inference_scaling.arllm.config import SamplingConfig, TokenPenalty
from inference_scaling.arllm.pipeline import ScalingOverrides
from inference_scaling.arllm.rewards.intrinsic import ConsilienceReward, SelfCertaintyReward
from inference_scaling.arllm.types import GenerationRequest, TokenStatistic
from inference_scaling.serve.protocol import ProtocolError, anthropic_conversation, openai_conversation
from inference_scaling.serve.reasoner import Reasoner
from inference_scaling.shared.model.output import ThinkingFormat, ThinkingParser
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.sampling.conditional_is import calibrate_reward_temperature
from inference_scaling.shared.types import pointwise
from tests.tabular import TabularAutoregressiveBackend
from tests.test_serve import _ChatBackend
from tests.test_streaming_vllm import DeltaEngine, backend_for
from tests.test_vllm_backend import FlatLogprobs, _SamplingParams


def test_gpu_worker_selected_for_scalar_self_certainty_on_cuda(monkeypatch) -> None:
    platform = SimpleNamespace(device_type="cuda")
    monkeypatch.setitem(sys.modules, "vllm.platforms", SimpleNamespace(current_platform=platform))
    arguments = {"logprobs_mode": "processed_logprobs", "additional_config": {SCALAR_STATISTICS_KEY: True}}
    with _engine_worker(arguments, False) as selected:
        assert selected["worker_cls"] == "inference_scaling.arllm.backends.vllm_gpu_worker.GPULogprobWorker"
    with pytest.raises(ValueError, match="cannot be combined"), _engine_worker(arguments, True):
        pass

    class TopSampler:
        logprobs_mode = "raw_logprobs"

    class FakeSampler:
        logprobs_mode = "raw_logprobs"

        def __init__(self):
            self.topk_topp_sampler = TopSampler()
            self.gather_logprobs = lambda lp, n, ids: None

    sampler = FakeSampler()
    runner = SimpleNamespace(sampler=sampler)

    class FakeGPUWorker:
        model_config = SimpleNamespace(logprobs_mode="processed_logprobs", get_vocab_size=lambda: 7)
        vllm_config = SimpleNamespace(additional_config={SCALAR_STATISTICS_KEY: True})

        def init_device(self):
            self.model_runner = runner

    monkeypatch.setitem(sys.modules, "vllm.v1.sample.sampler", SimpleNamespace(Sampler=FakeSampler))
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", SimpleNamespace(Worker=FakeGPUWorker))
    source = Path(__file__).resolve().parents[1] / "src/inference_scaling/arllm/backends/vllm_gpu_worker.py"
    spec = importlib.util.spec_from_file_location("gpu_worker_contract_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.GPULogprobWorker().init_device()
    assert sampler.logprobs_mode == sampler.topk_topp_sampler.logprobs_mode == "processed_logprobs"
    assert getattr(sampler, "_scalar_self_certainty_installed", False) is True


def test_v2_gpu_sampler_pre_mask_scalar_statistics() -> None:
    Payload = namedtuple("Payload", "logprob_token_ids logprobs selected_token_ranks")

    class FakeV2Sampler:
        def __init__(self, num_lp: int):
            self._num_lp = num_lp

        def get_logprobs_dims(self, idx_mapping_np):
            return (self._num_lp, 0) if self._num_lp > 0 else None

        def apply_sampling_params(self, logits, *args, skip_top_k_top_p: bool = False):
            return logits / 0.8

        def __call__(self, logits, input_batch):
            pre = self.apply_sampling_params(logits, skip_top_k_top_p=True)
            masked = pre.clone()
            masked[:, 2:] = float("-inf")
            sampled = torch.tensor([0, 1], dtype=torch.int64)
            logp = torch.log_softmax(masked, dim=-1)
            ids = torch.zeros((2, 1 + self._num_lp), dtype=torch.int64)
            vals = torch.zeros((2, 1 + self._num_lp), dtype=torch.float32)
            ids[:, 0] = sampled
            vals[:, 0] = logp.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
            return SimpleNamespace(logprobs_tensors=Payload(ids, vals, torch.zeros((2,), dtype=torch.int64)))

    raw = torch.tensor([[2.0, 0.5, -1.0, -2.5], [-0.5, 1.8, 0.2, -1.2]], dtype=torch.float32)
    scaled = raw / 0.8
    expected_kl = (torch.logsumexp(scaled, dim=-1) - scaled.mean(dim=-1) - math.log(4)).tolist()
    expected_top2 = (-torch.topk(torch.log_softmax(scaled, dim=-1), 2, dim=-1).values.mean(dim=-1)).tolist()
    batch = SimpleNamespace(idx_mapping_np=[0, 1])

    sampler_sc = FakeV2Sampler(num_lp=1)
    install_scalar_statistics(sampler_sc, 4)
    out_sc = sampler_sc(raw, batch).logprobs_tensors
    rows_sc = [(ids.tolist(), vals.tolist()) for ids, vals in zip(out_sc.logprob_token_ids, out_sc.logprobs)]
    _, sc_vals = scalar_generated_logprobs(FlatLogprobs(rows_sc), (0, 1), None)
    assert sc_vals == pytest.approx(tuple(expected_kl), abs=1e-5)

    sampler_cons = FakeV2Sampler(num_lp=3)
    install_scalar_statistics(sampler_cons, 4)
    out_cons = sampler_cons(raw, batch).logprobs_tensors
    rows_cons = [(ids.tolist(), vals.tolist()) for ids, vals in zip(out_cons.logprob_token_ids, out_cons.logprobs)]
    _, cons_vals = scalar_generated_logprobs(FlatLogprobs(rows_cons), (0, 1), 2)
    assert cons_vals == pytest.approx(tuple(expected_top2), abs=1e-5)


def test_pre_mask_scalar_statistics_stay_finite_and_exact_under_top_k_top_p() -> None:
    Payload = namedtuple("Payload", "logprob_token_ids logprobs selected_token_ranks")

    def fake_gather(logprobs, count, token_ids):
        if count == 0:
            return Payload(token_ids[:, None], logprobs.gather(-1, token_ids[:, None]), torch.ones(len(token_ids), dtype=torch.int32))
        values, ids = logprobs.topk(count, dim=-1)
        return Payload(
            torch.cat((token_ids[:, None], ids), dim=1),
            torch.cat((logprobs.gather(-1, token_ids[:, None]), values), dim=1),
            torch.ones(len(token_ids), dtype=torch.int32),
        )

    class FakeTopKTopP:
        def forward(self, logits, generators, k, p):
            # Simulate hard masking: keep only top-2 tokens, mask rest to -inf.
            masked = logits.clone()
            threshold = masked.topk(2, dim=-1).values[:, -1:]
            masked = masked.masked_fill(masked < threshold, float("-inf"))
            return masked.argmax(dim=-1), masked.log_softmax(dim=-1)

    class FakeSampler:
        def __init__(self):
            self.topk_topp_sampler = FakeTopKTopP()
            self.gather_logprobs = fake_gather

        def forward(self, logits, metadata):
            sampled, processed_lp = self.topk_topp_sampler.forward(logits, {}, metadata.top_k, metadata.top_p)
            return self.gather_logprobs(processed_lp, metadata.max_num_logprobs, sampled)

    sampler = FakeSampler()
    install_scalar_statistics(sampler, 7)
    logits = torch.tensor([[4.0, 2.0, 1.0, 0.5, -0.5, -1.0, -2.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    pre_lp = logits.log_softmax(dim=-1)
    expected_sc = (-pre_lp.mean(dim=-1) - math.log(7)).tolist()
    expected_top3 = (-pre_lp.topk(3, dim=-1).values.mean(dim=-1)).tolist()

    # Even though top_k=2 masks 5 tokens to -inf in processed_lp, pre-mask SC and top-3 are finite & exact!
    meta_sc = SimpleNamespace(max_num_logprobs=1, top_k=torch.tensor([2, 2]), top_p=torch.tensor([0.95, 0.95]))
    out_sc = sampler.forward(logits.clone(), meta_sc)
    rows_sc = [(ids.tolist(), vals.tolist()) for ids, vals in zip(out_sc.logprob_token_ids, out_sc.logprobs)]
    decoded_lp, decoded_sc = scalar_generated_logprobs(FlatLogprobs(rows_sc), (0, 0), None)
    assert decoded_sc == pytest.approx(expected_sc, abs=1e-5)
    assert all(math.isfinite(v) for v in decoded_lp)

    meta_top3 = SimpleNamespace(max_num_logprobs=4, top_k=torch.tensor([2, 2]), top_p=torch.tensor([0.95, 0.95]))
    out_top3 = sampler.forward(logits.clone(), meta_top3)
    rows_top3 = [(ids.tolist(), vals.tolist()) for ids, vals in zip(out_top3.logprob_token_ids, out_top3.logprobs)]
    decoded_lp2, decoded_top3 = scalar_generated_logprobs(FlatLogprobs(rows_top3), (0, 0), 3)
    assert decoded_top3 == pytest.approx(expected_top3, abs=1e-5)
    assert decoded_lp2 == pytest.approx(decoded_lp, abs=1e-6)

    # Backend.records accepts truncated sampling policies when scalar_self_certainty=True and temperature matches.
    backend = VLLMBackend(None, SimpleNamespace(bos_token_id=None), model_id="fake", parameter_count=7,
                          sampling_params_factory=_SamplingParams, scalar_self_certainty=True)
    stat = TokenStatistic(SamplingConfig(temperature=0.8))
    assert backend.records(stat, SamplingConfig(temperature=0.8, top_p=0.95, top_k=20))
    assert not backend.records(stat, SamplingConfig(temperature=1.0, top_p=0.95, top_k=20))


def test_self_certainty_and_consilience_refinements_and_incomplete_penalty() -> None:
    class FixedConf:
        model_id = "m"

        def __init__(self, values):
            self.values = values

        def token_statistics(self, requests, statistic):
            return [tuple(self.values[tok] for tok in cont) for req in requests for cont in req.continuations]

    # Tokens: 10(skip), 1(low), 2(low), 8(high), 9(high); 90=<think>, 91=</think>
    backend = FixedConf({10: 5.0, 1: 1.0, 2: 2.0, 3: 8.0, 4: 9.0, 90: 0.0, 91: 0.0})
    full_stat = TokenStatistic(SamplingConfig())
    top_stat = TokenStatistic(SamplingConfig(), 2)
    parser = ThinkingParser((ThinkingFormat((91,), (90,)),))

    # tail_mean focuses on lowest 50% after skipping 20% (skip index 0 -> body=[1, 2, 8, 9], lowest 2 -> mean(1,2)=1.5)
    sc_tail = SelfCertaintyReward(backend, full_stat, scope="full", aggregation="tail_mean",
                                  tail_fraction=0.5, skip_fraction=0.2)
    assert sc_tail((), (10, 1, 2, 3, 4)) == pytest.approx(1.5)
    # Length-invariance check on repeating trajectory (with skip=0).
    sc_tail0 = SelfCertaintyReward(backend, full_stat, scope="full", aggregation="tail_mean",
                                   tail_fraction=0.5, skip_fraction=0.0)
    assert sc_tail0((), (1, 2, 3, 4)) == pytest.approx(sc_tail0((), (1, 2, 3, 4, 1, 2, 3, 4)))

    # final_window blends body mean (mean(1,2,8,9)=5.0) and final 50% (mean(8,9)=8.5) with weight 0.5 -> 6.75.
    sc_win = SelfCertaintyReward(backend, full_stat, scope="full", aggregation="final_window",
                                 window_fraction=0.5, final_weight=0.5, skip_fraction=0.2)
    assert sc_win((), (10, 1, 2, 3, 4)) == pytest.approx(6.75)
    assert sc_win.describe()["aggregation"] == "final_window"

    # incomplete_penalty penalizes unclosed thinking trajectories in batch(), NOT complete ones.
    sc_pen = SelfCertaintyReward(backend, full_stat, thinking_format=parser, scope="thinking", incomplete_penalty=2.0)
    complete_seq = (90, 3, 4, 91)  # thinking=[3,4] -> mean 8.5, no penalty
    incomplete_seq = (90, 3, 4)    # fallback to full=[90,3,4] -> mean (0+8+9)/3 - 2.0
    scores = sc_pen.batch((), (complete_seq, incomplete_seq))
    assert scores[0] == pytest.approx(8.5)
    assert scores[1] == pytest.approx((0.0 + 8.0 + 9.0) / 3.0 - 2.0)
    assert sc_pen.describe()["incomplete_penalty"] == 2.0

    # Consilience also supports incomplete_penalty cleanly.
    c_pen = ConsilienceReward(backend, top_stat, thinking_format=parser, scope="thinking",
                              window_tokens=1, skip_fraction=0.0, initial_penalty=1.0, incomplete_penalty=1.5)
    assert c_pen((), complete_seq) == pytest.approx(9.0 - 8.0)
    assert c_pen((), incomplete_seq) == pytest.approx((9.0 - 0.0) - 1.5)

    # temperature_normalized scales Self-Certainty by score_temperature so pairwise gaps stay invariant to T.
    stat_t07 = TokenStatistic(SamplingConfig(temperature=0.7))
    sc_tnorm = SelfCertaintyReward(backend, stat_t07, scope="full", temperature_normalized=True)
    assert sc_tnorm((), (3, 4)) == pytest.approx(0.7 * 8.5)
    assert sc_tnorm.describe()["temperature_normalized"] is True


def test_adaptive_v2_stage2_forecast_and_final_pool_straggler_pruning() -> None:
    from inference_scaling.arllm.algorithms.conditional_is import forecast_candidate_count

    cfg = ConditionalISConfig(
        candidate_count=4, rollout_count=2, block_size=1024, total_length=12288,
        reward_temperature=0.25, block_first=False, exploration_schedule="adaptive",
        selection="block_metropolized",
    )
    # Low dispersion across unused Stage-1 groups -> Stage 2 skipped (0 fresh suffixes).
    flat_obs = ((1, 2, 4096, 10.0), (1, 2, 4096, 10.01), (2, 2, 4096, 10.02), (3, 1, 4096, 10.01))
    assert forecast_candidate_count(flat_obs, 1024, cfg, SeedStream(1)) == 0
    # High dispersion across unused Stage-1 groups -> Stage 2 capped strictly at M-1 = 3.
    wide_obs = ((1, 2, 8192, 10.0), (2, 2, 8192, 14.0), (3, 1, 8192, 6.0))
    fanout = forecast_candidate_count(wide_obs, 1024, cfg, SeedStream(1))
    assert 1 <= fanout <= cfg.candidate_count - 1

    # Verify Stage 2 (final_pool) straggler pruning actually aborts a runaway Stage-2 suffix!
    # Pick a seed whose Stage-2 length_window draw gives a wide upper bound (~1.9x) so the runaway crosses threshold_len.
    seed_s2 = next(
        s for s in range(1, 100)
        if float(SeedStream(s).generator("conditional_is", 1, "length_window").random()) > 0.85
    )
    stage2_straggler_seed = SeedStream(seed_s2).derive("conditional_is", 1, "candidate", 1)

    step0_seeds = {SeedStream(seed_s2).derive("conditional_is", 0, "candidate", i): i for i in range(4)}

    class Stage2StragglerEngine(DeltaEngine):
        async def generate(self, *, prompt, sampling_params, request_id, **kwargs):
            import asyncio
            params = sampling_params
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            is_s2_runaway = params.seed == stage2_straggler_seed
            # Step 0 prefixes finish at len=100 (suffix=96); Stage-2 fast peers finish at 96 (> min_len), runaway loops to 256.
            target_len = min(
                params.max_tokens,
                256 if is_s2_runaway else (96 if len(prompt["prompt_token_ids"]) > 1 else 100),
            )
            tok_lp = -3.0 if is_s2_runaway else -0.1
            step0_idx = step0_seeds.get(params.seed, 0)
            body_tok = 4 if (len(prompt["prompt_token_ids"]) == 1 and step0_idx % 2 == 0) else 3
            try:
                for offset in range(target_len):
                    await asyncio.sleep(0.0005 if is_s2_runaway else 0.0001)
                    tok = 2 if (not is_s2_runaway and offset + 1 == target_len) else body_tok
                    finished = offset + 1 == target_len or tok in params.stop_token_ids
                    pos = FlatLogprobs([([tok, 3], [tok_lp, tok_lp])])
                    yield SimpleNamespace(
                        request_id=request_id, num_cached_tokens=0, finished=finished,
                        outputs=[SimpleNamespace(token_ids=[tok], logprobs=pos,
                                                 finish_reason=("stop" if tok in params.stop_token_ids else "length")
                                                 if finished else None)],
                    )
                    if finished:
                        break
            finally:
                self.active -= 1

    engine = Stage2StragglerEngine(stream_interval=1)
    backend = backend_for(engine)
    backend.configure_scheduler(SchedulerConfig(max_in_flight=8))
    stat = TokenStatistic(SamplingConfig(), top_k=1)
    recorder = StatisticRecorder(backend, stat)

    def prefix_score(_prompt, prefix):
        vals = recorder.recorded_statistics((1,), prefix, stat)
        if vals is None:
            return None
        # Give Step 0 distinct group rewards so dispersion >= 0.15 triggers Stage 2.
        distinct = 1.5 if 4 in prefix[:4] else 0.0
        return -sum(vals) / len(vals) + distinct

    def full_reward(_prompt, seqs):
        return [prefix_score((1,), s) or 0.0 for s in seqs]

    try:
        cfg_s2 = ConditionalISConfig(
            candidate_count=4, rollout_count=1, block_size=4, total_length=256,
            reward_temperature=0.5, block_first=False, exploration_schedule="adaptive",
            selection="block_metropolized", pilot_gate=False, straggler_pruning=True,
        )
        res = run_conditional_is(
            recorder, (1,), cfg_s2, full_reward, SeedStream(seed_s2),
            sampling=SamplingConfig(eos_token_id=2), prefix_reward=prefix_score,
        )
        assert len(res.steps) == 2
        assert engine.aborted  # Stage-2 runaway suffix was aborted cleanly!
        assert len(res.token_ids) == 100
    finally:
        backend.close()


def test_calibrate_reward_temperature_bounds_ess_and_is_symmetric() -> None:
    # Already healthy spread: unchanged.
    assert calibrate_reward_temperature([[1.0], [1.2], [1.4], [1.6]], 0.5) == 0.5
    # Extreme spread causing ESS collapse (< 1.35): raises effective temperature.
    raised = calibrate_reward_temperature([[0.0], [0.0], [0.0], [10.0]], 0.25)
    assert 0.25 < raised <= 1.0
    # Flat cluster causing near-uniform ESS (> 0.85 * M): lowers effective temperature.
    lowered = calibrate_reward_temperature([[1.00], [1.01], [1.02], [1.03]], 2.0)
    assert 1.0 <= lowered < 2.0
    # Permutation invariance and equal-reward safety.
    assert calibrate_reward_temperature([[10.0], [0.0], [0.0], [0.0]], 0.25) == pytest.approx(raised)
    assert math.isfinite(calibrate_reward_temperature([[2.0], [2.0], [2.0], [2.0]], 0.5))


def test_adaptive_is_pilot_gate_early_exits_and_escalates_cleanly() -> None:
    # Case 1: Easy problem — EOS token (2) appears immediately at length 1 < B=4.
    easy_backend = TabularAutoregressiveBackend({(): (0.0, 0.0, 1.0)}, fallback=(0.0, 0.0, 1.0))
    cfg = ConditionalISConfig(
        candidate_count=4, rollout_count=2, block_size=4, total_length=12,
        reward_temperature=0.5, block_first=False, exploration_schedule="adaptive",
        selection="block_metropolized", pilot_gate=True, selective_lookahead=True,
    )
    res_easy = run_conditional_is(
        easy_backend, (), cfg, pointwise(lambda _p, g: float(len(g))), SeedStream(7),
        sampling=SamplingConfig(eos_token_id=2),
    )
    assert len(res_easy.steps) == 1
    assert res_easy.steps[0].rollout_evaluations_performed == 1
    assert len(res_easy.steps[0].candidates) == 1
    assert res_easy.token_ids == (2,)

    # Case 2: Hard problem — runs full length 10 > B=3, escalates pilot + runs unbiased selective K!
    hard_backend = TabularAutoregressiveBackend({}, fallback=(0.5, 0.5))
    stat = TokenStatistic(SamplingConfig(), top_k=1)
    recorder = StatisticRecorder(hard_backend, stat)
    reward = ConsilienceReward(recorder, stat, scope="full", window_tokens=1, skip_fraction=0.0, initial_penalty=0.0)
    cfg_hard = ConditionalISConfig(
        candidate_count=4, rollout_count=2, block_size=3, total_length=10,
        reward_temperature=0.5, block_first=False, exploration_schedule="adaptive",
        selection="block_metropolized", pilot_gate=True, selective_lookahead=True,
    )
    res_hard = run_conditional_is(
        recorder, (), cfg_hard, reward.batch, SeedStream(13),
        sampling=SamplingConfig(temperature=0.8, top_p=0.95, top_k=2),
        prefix_reward=reward.cached_prefix,
    )
    # Step 0 should have M=4 candidates; selective K=2 assigns 2 rollouts to ceil(4/2)=2 prefixes and 1 to the other 2 -> 6 total!
    step0 = res_hard.steps[0]
    assert len(step0.candidates) == 4
    rollout_counts = sorted(len(c.rollouts) for c in step0.candidates)
    assert rollout_counts == [1, 1, 2, 2]
    assert step0.rollout_evaluations_performed == 6


def test_adaptive_is_streaming_straggler_pruning_aborts_runaway_loop() -> None:
    straggler_seed = SeedStream(9).derive("conditional_is", 0, "candidate", 1)

    # Build a streaming engine where candidate 1 runs very long (len=96) with low confidence while others finish cleanly at len=5.
    class AsymmetricDeltaEngine(DeltaEngine):
        async def generate(self, *, prompt, sampling_params, request_id, **kwargs):
            import asyncio
            params = sampling_params
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            is_straggler = params.seed == straggler_seed
            target_len = min(params.max_tokens, 96 if is_straggler else 5)
            tok_lp = -3.0 if is_straggler else -0.1  # lower reward (-3.0 vs -0.1) on straggler!
            try:
                for offset in range(target_len):
                    await asyncio.sleep(0.002 if is_straggler else 0.0005)
                    tok = 2 if (not is_straggler and offset + 1 == target_len) else 3
                    finished = offset + 1 == target_len or tok in params.stop_token_ids
                    pos = FlatLogprobs([([tok, 3], [tok_lp, tok_lp])])
                    yield SimpleNamespace(
                        request_id=request_id, num_cached_tokens=0, finished=finished,
                        outputs=[SimpleNamespace(token_ids=[tok], logprobs=pos,
                                                 finish_reason=("stop" if tok in params.stop_token_ids else "length")
                                                 if finished else None)],
                    )
                    if finished:
                        break
            finally:
                self.active -= 1

    engine = AsymmetricDeltaEngine(stream_interval=1)
    backend = backend_for(engine)
    backend.configure_scheduler(SchedulerConfig(max_in_flight=8))
    stat = TokenStatistic(SamplingConfig(), top_k=1)
    recorder = StatisticRecorder(backend, stat)

    def prefix_score(_prompt, prefix):
        vals = recorder.recorded_statistics((1,), prefix, stat)
        return None if vals is None else -sum(vals) / len(vals)

    def full_reward(_prompt, seqs):
        return [prefix_score((1,), s) or 0.0 for s in seqs]

    try:
        cfg = ConditionalISConfig(
            candidate_count=4, rollout_count=1, block_size=4, total_length=96,
            reward_temperature=0.5, block_first=False, exploration_schedule="adaptive",
            selection="block_metropolized", pilot_gate=False, straggler_pruning=True,
        )
        res = run_conditional_is(
            recorder, (1,), cfg, full_reward, SeedStream(9),
            sampling=SamplingConfig(eos_token_id=2), prefix_reward=prefix_score,
        )
        assert engine.aborted  # The runaway straggler (candidate 1) was pruned!
        assert len(res.token_ids) == 5
    finally:
        backend.close()


def test_openai_backend_and_pluggable_registry() -> None:
    calls = []

    def fake_transport(path, payload):
        calls.append((path, payload))
        return {
            "choices": [{
                "token_ids": [4, 2],
                "finish_reason": "stop",
                "logprobs": {
                    "tokens": ["4", "2"],
                    "token_logprobs": [-0.2, -0.3],
                    "top_logprobs": [{"4": -0.2, "5": -1.2}, {"2": -0.3, "5": -1.3}],
                },
            }]
        }

    tok = SimpleNamespace(encode=lambda s, add_special_tokens=True: [1, 2], decode=lambda ids, skip_special_tokens=True: "ok")
    backend = OpenAIBackend(
        base_url="http://127.0.0.1:9999", model="qwen", tokenizer=tok,
        parameter_count=100, token_penalty=TokenPenalty((5,), 1.0), transport=fake_transport,
    )
    try:
        caps = backend_capabilities(backend)
        assert caps == EngineCapabilities(
            backend="openai", device="remote", streaming=False,
            native_continuous_batching=True, native_token_stops=True,
            scalar_self_certainty=False, pre_mask_statistics=False,
            exact_scoring=False, constrained_beam_search=False, sampling_constraints=False,
        )
        stat = TokenStatistic(SamplingConfig(), top_k=2)
        sample = backend.sample_batch([GenerationRequest((1,), 4, SamplingConfig(eos_token_id=2), 7, "req1", statistic=stat)])[0]
        assert sample.token_ids == (4, 2) and sample.finish_reason == "eos"
        assert sample.token_statistics == pytest.approx((0.7, 0.8))
        assert calls[0][1]["logit_bias"] == {"5": -1.0}
        assert backend.snapshot().generated_tokens == 2
        assert "openai" in loader.registered_backends()
    finally:
        backend.close()


def test_serve_per_request_scaling_overrides_and_health_capabilities() -> None:
    conv = openai_conversation({
        "messages": [{"role": "user", "content": "hi"}],
        "scaling": {"candidate_count": 3, "rollout_count": 2, "block_size": 1},
    })
    assert conv.scaling == {"candidate_count": 3, "rollout_count": 2, "block_size": 1}
    assert anthropic_conversation({
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 32,
        "scaling": {"candidate_count": 2},
    }).scaling == {"candidate_count": 2}
    with pytest.raises(ProtocolError, match="unknown scaling"):
        openai_conversation({"messages": [], "scaling": {"foo": 1}})
    with pytest.raises(ProtocolError, match="positive integer"):
        openai_conversation({"messages": [], "scaling": {"candidate_count": 0}})

    settings = copy.deepcopy(load_settings())
    settings["ar"]["algorithms"]["is"].update(selection_ready=False, exploration_schedule="adaptive", selection="block_metropolized")
    settings["ar"]["output"]["finish_sampling"] = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
    validate_settings(settings)

    backend = _ChatBackend()
    reasoner = Reasoner(settings, backend)
    caps = reasoner.capabilities()
    assert caps["exploration_schedule"] == "adaptive"
    assert caps["defaults"] == {"candidate_count": 4, "rollout_count": 2, "block_size": 1024}
    prepared = reasoner.prepare(conv)
    assert prepared.scaling_overrides == ScalingOverrides(3, 2, 1)
    reply = reasoner.generate(prepared)
    assert (reply.scaling["candidate_count"], reply.scaling["rollout_count"], reply.scaling["block_size"]) == (3, 2, 1)
    assert "engine_capabilities" in reply.scaling

    # Validate invalid adaptive + selection_ready conflict in settings.
    broken = copy.deepcopy(settings)
    broken["ar"]["algorithms"]["is"]["selection"] = "sir"
    broken["ar"]["algorithms"]["is"]["selection_ready"] = True
    with pytest.raises(SettingsError, match="adaptive"):
        validate_settings(broken)


def test_settings_env_var_override(tmp_path, monkeypatch) -> None:
    import json
    from inference_scaling.app.settings import SETTINGS_ENV_VAR

    custom = copy.deepcopy(load_settings())
    custom["serve"]["server"]["port"] = 8123
    preset = tmp_path / "preset.json"
    preset.write_text(json.dumps(custom), encoding="utf-8")
    monkeypatch.setenv(SETTINGS_ENV_VAR, str(preset))
    loaded = load_settings()
    assert loaded["serve"]["server"]["port"] == 8123


def test_all_hardware_presets_pass_strict_schema_validation() -> None:
    presets_dir = Path(__file__).resolve().parents[1] / "settings" / "presets"
    expected = {
        "ascend_910b3_tp2.json",
        "cuda_gpu_bf16.json",
        "cuda_gpu_fp8.json",
        "cuda_gpu_24g_mxfp4.json",
        "openai_remote.json",
    }
    found = {p.name for p in presets_dir.glob("*.json")}
    assert expected.issubset(found)

    loaded = {}
    for name in sorted(expected):
        cfg = load_settings(presets_dir / name)
        validate_settings(cfg)
        assert cfg["ar"]["model"]["token_penalty"] is None
        assert cfg["ar"]["sampling"] == {"temperature": 0.7, "top_p": 0.95, "top_k": 20}
        is_cfg = cfg["ar"]["algorithms"]["is"]
        assert is_cfg["exploration_schedule"] == "adaptive"
        assert is_cfg["selection_ready"] is False
        assert is_cfg["selection"] == "block_metropolized"
        loaded[name] = cfg

    assert loaded["ascend_910b3_tp2.json"]["ar"]["engine"]["device"] == "npu"
    assert loaded["ascend_910b3_tp2.json"]["ar"]["engine"]["vllm"]["tensor_parallel_size"] == 2
    assert loaded["ascend_910b3_tp2.json"]["ar"]["engine"]["vllm"]["gpu_memory_utilization"] == 0.88
    assert loaded["ascend_910b3_tp2.json"]["ar"]["engine"]["vllm"]["stream_interval"] == 8
    assert loaded["cuda_gpu_bf16.json"]["ar"]["engine"]["device"] == "cuda"
    assert loaded["cuda_gpu_bf16.json"]["ar"]["engine"]["vllm"]["scalar_self_certainty"] is True
    assert loaded["cuda_gpu_fp8.json"]["ar"]["engine"]["vllm"]["quantization"] == "fp8"
    assert loaded["cuda_gpu_24g_mxfp4.json"]["ar"]["engine"]["vllm"]["quantization"] == "mxfp4"
    assert loaded["cuda_gpu_24g_mxfp4.json"]["ar"]["algorithms"]["is"]["block_size"] == 832
    assert loaded["openai_remote.json"]["ar"]["engine"]["backend"] == "openai"
    assert loaded["openai_remote.json"]["run"]["reward"] == "consilience"


def test_progressive_single_chain_geometric_stages_and_decaying_lookahead() -> None:
    backend = TabularAutoregressiveBackend({}, fallback=(0.5, 0.5))
    # Give wide reward variance so dispersion stays well above 0.12 across all stages.
    reward = pointwise(lambda _p, seq: float(sum(tok * (i + 1) for i, tok in enumerate(seq))))
    cfg = ConditionalISConfig(
        candidate_count=4, rollout_count=4, block_size=2, total_length=32,
        reward_temperature=1.0, block_first=False, exploration_schedule="progressive",
        selection="block_metropolized", pilot_gate=False, selective_lookahead=True,
        max_stages=4, block_growth=2.0, beam_particles=1,
    )
    committed: list[tuple[int, ...]] = []
    res = run_conditional_is(backend, (), cfg, reward, SeedStream(42), on_commit=committed.append)
    assert len(res.token_ids) == 32
    # Geometric checkpoints: B0=2 (K=4), B1=4 (K=2), B2=8 (K=1), B3=18 (seal, K=1) -> 4 stages!
    assert len(res.steps) == 4
    plans = [step.sampling_plan for step in res.steps]
    assert [p[0] for p in plans] == [2, 4, 8, 18]
    assert [p[2] for p in plans] == [4, 2, 1, 1]
    assert [step.generated_length_before for step in res.steps] == [0, 2, 6, 14]
    assert committed[-1] == res.token_ids
    assert all(res.token_ids[:len(pref)] == pref for pref in committed)


def test_progressive_particle_smc_multistage_streaming_and_mass_conservation() -> None:
    from inference_scaling.shared.sampling.conditional_is import (
        ConditionalCandidate, RolloutEvaluation, resample_particles,
    )

    # 1) Systematic particle resampling preserves diversity when remaining mass > 1/P.
    cands = (
        ConditionalCandidate((0,), (RolloutEvaluation((0,), 0.0, math.log(0.6)),), math.log(0.6)),
        ConditionalCandidate((1,), (RolloutEvaluation((1,), 0.0, math.log(0.4)),), math.log(0.4)),
    )
    distinct_rate = sum(
        len({g for g, _ in resample_particles(cands, 2, SeedStream(s), "t")}) == 2
        for s in range(100)
    ) / 100.0
    assert distinct_rate > 0.7  # Exact systematic probability is 0.8 (vs 0.48 for i.i.d.)!

    # 2) Easy problem with pilot_gate=True exits at 1x cost even when beam_particles=2.
    easy_backend = TabularAutoregressiveBackend({(): (0.0, 0.0, 1.0)}, fallback=(0.0, 0.0, 1.0))
    cfg_easy = ConditionalISConfig(
        candidate_count=6, rollout_count=2, block_size=4, total_length=16,
        reward_temperature=0.5, block_first=False, exploration_schedule="progressive",
        selection="block_metropolized", pilot_gate=True, max_stages=4, block_growth=2.0, beam_particles=2,
    )
    res_easy = run_conditional_is(
        easy_backend, (), cfg_easy, pointwise(lambda _p, g: float(len(g))), SeedStream(3),
        sampling=SamplingConfig(eos_token_id=2),
    )
    assert len(res_easy.steps) == 1 and res_easy.steps[0].rollout_evaluations_performed == 1
    assert res_easy.token_ids == (2,)

    # 3) Hard multi-stage Block-SMC on batch backend runs 4 geometric stages with P=2 particles.
    hard_backend = TabularAutoregressiveBackend({}, fallback=(0.5, 0.5))
    stat = TokenStatistic(SamplingConfig(), top_k=1)
    recorder = StatisticRecorder(hard_backend, stat)
    reward = pointwise(lambda _p, seq: float(sum(tok * (i + 1) for i, tok in enumerate(seq))))
    cfg_smc = ConditionalISConfig(
        candidate_count=6, rollout_count=2, block_size=2, total_length=24,
        reward_temperature=1.0, block_first=False, exploration_schedule="progressive",
        selection="block_metropolized", pilot_gate=False, selective_lookahead=True,
        max_stages=4, block_growth=2.0, beam_particles=2,
    )
    committed: list[tuple[int, ...]] = []
    res_smc = run_conditional_is(
        recorder, (), cfg_smc, reward, SeedStream(11),
        sampling=SamplingConfig(temperature=0.7, top_p=0.95, top_k=2),
        on_commit=committed.append,
    )
    assert len(res_smc.token_ids) == 24
    assert len(res_smc.steps) == 4
    assert committed[-1] == res_smc.token_ids

    # 4) Verify streaming DeltaEngine works cleanly with P=2 progressive Block-SMC!
    engine = DeltaEngine(stream_interval=2)
    stream_backend = backend_for(engine)
    stream_backend.configure_scheduler(SchedulerConfig(max_in_flight=16))
    stream_stat = TokenStatistic(SamplingConfig(), top_k=1)
    stream_rec = StatisticRecorder(stream_backend, stream_stat)
    try:
        res_stream = run_conditional_is(
            stream_rec, (1,),
            ConditionalISConfig(
                candidate_count=6, rollout_count=2, block_size=2, total_length=16,
                reward_temperature=0.5, block_first=False, exploration_schedule="progressive",
                selection="block_metropolized", pilot_gate=False, max_stages=3, block_growth=2.0, beam_particles=2,
            ),
            pointwise(lambda _p, seq: float(sum(seq))),
            SeedStream(19),
            sampling=SamplingConfig(eos_token_id=2),
            prefix_reward=lambda _p, pref: float(sum(pref)) / max(1, len(pref)),
        )
        assert 1 <= len(res_stream.steps) <= 3
        assert len(res_stream.token_ids) > 0
    finally:
        stream_backend.close()

    # 5) Settings & Reasoner validation for progressive overrides (max_stages, beam_particles).
    settings = copy.deepcopy(load_settings())
    settings["ar"]["algorithms"]["is"].update(
        exploration_schedule="progressive", candidate_count=6, max_stages=4, block_growth=2.0, beam_particles=2,
    )
    validate_settings(settings)
    conv = openai_conversation({
        "messages": [{"role": "user", "content": "hi"}],
        "scaling": {"candidate_count": 4, "rollout_count": 2, "block_size": 2, "max_stages": 3, "beam_particles": 2},
    })
    reasoner = Reasoner(settings, _ChatBackend())
    prepared = reasoner.prepare(conv)
    assert prepared.scaling_overrides == ScalingOverrides(4, 2, 2, max_stages=3, beam_particles=2)
    reply = reasoner.generate(prepared)
    assert reply.finish == "stop"

    bad = copy.deepcopy(settings)
    bad["ar"]["algorithms"]["is"]["beam_particles"] = 8  # exceeds candidate_count=6
    with pytest.raises(SettingsError, match="beam_particles"):
        validate_settings(bad)
    bad2 = copy.deepcopy(settings)
    bad2["ar"]["algorithms"]["is"]["exploration_schedule"] = "adaptive"
    bad2["ar"]["algorithms"]["is"]["beam_particles"] = 2
    with pytest.raises(SettingsError, match="progressive"):
        validate_settings(bad2)
