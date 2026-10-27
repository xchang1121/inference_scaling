from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
import types
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy
import pytest

from inference_scaling.arllm.algorithms.mh import run_mh_chain
from inference_scaling.arllm.backends.vllm_backend import (AsyncVLLMBackend, VLLMBackend, _checkpoint_parameter_count,
                                                        _engine_worker, _generated_logprobs, _load_vllm_sampling_api)
from inference_scaling.arllm.algorithms.config import MHConfig
from inference_scaling.arllm.config import SamplingConfig, TokenPenalty
from inference_scaling.arllm.types import Draft, GenerationRequest, LogWeightStop, ScoreRequest, TokenStatistic
from inference_scaling.shared.rng import SeedStream


def test_ascend_worker_mode_and_platform_selection(monkeypatch) -> None:
    platform = SimpleNamespace(device_type="cuda")
    monkeypatch.setitem(sys.modules, "vllm.platforms", SimpleNamespace(current_platform=platform))
    arguments = {"logprobs_mode": "processed_logprobs", "worker_extension_cls": "custom.Extension"}
    with _engine_worker(arguments, False) as selected:
        assert selected is arguments
    platform.device_type = "npu"
    with _engine_worker(arguments, False) as selected:
        assert selected == {**arguments, "worker_cls":
            "inference_scaling.arllm.backends.vllm_ascend_worker.AscendLogprobWorker"}
    with pytest.raises(ValueError, match="unsupported on Ascend"), _engine_worker(arguments, True):
        pass
    assert "worker_cls" not in arguments

    class TopSampler:
        logprobs_mode = "raw_logprobs"

    class Sampler:
        logprobs_mode = "raw_logprobs"

        def __init__(self):
            self.topk_topp_sampler = TopSampler()

    sampler = Sampler()
    runner = SimpleNamespace(sampler=sampler, ascend_config=SimpleNamespace(enable_reduce_sample=False))

    class Worker:
        model_config = SimpleNamespace(logprobs_mode="processed_logprobs")

        def init_device(self):
            self.model_runner = runner

    monkeypatch.setitem(sys.modules, "vllm_ascend.sample.sampler",
                        SimpleNamespace(AscendSampler=Sampler, AscendTopKTopPSampler=TopSampler))
    monkeypatch.setitem(sys.modules, "vllm_ascend.worker.worker", SimpleNamespace(NPUWorker=Worker))
    source = Path(__file__).resolve().parents[1] / "src/inference_scaling/arllm/backends/vllm_ascend_worker.py"
    spec = importlib.util.spec_from_file_location("ascend_contract_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for _ in range(2):  # Also safe when upstream already implements the configured mode.
        module.AscendLogprobWorker().init_device()
        assert sampler.logprobs_mode == sampler.topk_topp_sampler.logprobs_mode == "processed_logprobs"
        assert runner.sampler is sampler
    runner.ascend_config.enable_reduce_sample = True
    with pytest.raises(ValueError, match="enable_reduce_sample=false"):
        module.AscendLogprobWorker().init_device()
    runner.ascend_config.enable_reduce_sample = False
    runner.sampler = None
    with pytest.raises(RuntimeError, match="V1 AscendSampler"):
        module.AscendLogprobWorker().init_device()


@dataclass
class _Logprob:
    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


class FlatLogprobs:
    """vLLM's primitive arrays, including sampled/top-K duplicates; dictionary reconstruction is forbidden."""
    def __init__(self, rows):
        self.start_indices, self.end_indices, self.token_ids, self.logprobs = [], [], [], []
        self.ranks, self.decoded_tokens = [], []
        for ids, values in rows:
            self.start_indices.append(len(self.token_ids))
            for token, value in zip(ids, values, strict=True):
                self.token_ids.append(token)
                self.logprobs.append(value)
                self.ranks.append(None)
                self.decoded_tokens.append(None)
            self.end_indices.append(len(self.token_ids))

    def __len__(self):
        return len(self.start_indices)

    def __iter__(self):
        raise AssertionError("flat probabilities must not be expanded into per-token dictionaries")


def test_flat_probabilities_match_dictionary_rewards_including_sampled_topk_duplicates():
    from inference_scaling.shared.rewards.consilience import confidence_windows

    # Sampled token outside top-K, inside top-K, and rows stored through append(dict).
    rows = [([9, 3, 4, 5], [-7.0, -.1, -2., -3.]),
            ([4, 3, 4, 5], [-2., -.1, -2., -3.]),
            ([4, 3, 5], [-2., -.1, -3.])] * 15
    tokens = (9, 4, 4) * 15
    dictionaries = [dict(zip(ids, map(_Logprob, values), strict=True)) for ids, values in rows]
    flat = FlatLogprobs(rows)
    for top_k in (None, 3):
        expected = _generated_logprobs(dictionaries, tokens, top_k)
        actual = _generated_logprobs(flat, tokens, top_k)
        assert actual == expected
        if top_k is not None:
            assert confidence_windows(actual[1]) == confidence_windows(expected[1])
    with pytest.raises(RuntimeError, match="required token"):
        _generated_logprobs(FlatLogprobs([([3], [-1.])]), (4,), None)


@dataclass
class _Completion:
    token_ids: list[int]
    logprobs: list[dict[int, _Logprob]]
    finish_reason: str = "length"


@dataclass
class _Output:
    outputs: list[_Completion]
    prompt_logprobs: list[dict[int, _Logprob] | None] | None = None
    num_cached_tokens: int = 0
    request_id: str | None = None


class _SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def test_initial_strata_preserve_aliases_concurrent_pools_and_native_payload():
    import torch
    from inference_scaling.arllm.backends.vllm_stratified import InitialStrataWorker
    from inference_scaling.shared.sampling.stratified import InitialStrata

    registry = InitialStrata()
    with pytest.raises(ValueError, match="exactly one"):
        registry.add([(10, 0, 2), (11, 0, 2)])
    entries = [(10, 0, 2), (11, 1, 2)]
    registry.add(entries)
    registry.bind([(10, 100)])
    with pytest.raises(ValueError, match="colliding"):
        registry.bind([(11, 100)])
    reference = InitialStrata()
    reference.add(entries)
    for probabilities in ([.95, .05], [.9, .1], [.2, .8]):
        assert registry.sample(100, numpy.array(probabilities)) == reference.sample(10, numpy.array(probabilities))
    registry.add([(20, 0, 2), (21, 1, 2)])
    registry.remove([10, 11])
    assert not registry.active(100) and registry.active(20)
    assert registry.remove([10, 11])["registered"] == 0

    payload = torch.arange(8).reshape(2, 4).float()
    def native(logits, generators, k, p):
        return torch.zeros(len(logits), dtype=torch.int64), payload
    sampler = SimpleNamespace(forward=native)
    worker = InitialStrataWorker()
    worker.model_runner = SimpleNamespace(sampler=SimpleNamespace(topk_topp_sampler=sampler))
    worker.begin_initial_strata(entries)
    unrelated = {0: torch.Generator().manual_seed(90), 1: torch.Generator().manual_seed(91)}
    logits = torch.zeros(2, 4)
    tokens, actual_payload = sampler.forward(logits, unrelated, None, None)
    assert tokens.tolist() == [0, 0] and actual_payload is payload
    generators = {row: torch.Generator().manual_seed(seed) for row, seed in enumerate((10, 11))}
    tokens, actual_payload = sampler.forward(logits, generators, None, None)
    assert tokens[0] < 2 <= tokens[1] and actual_payload is payload
    assert sampler.forward is native
    summary = worker.end_initial_strata([10, 11])
    assert summary["touched"] == 2 and summary["constrained_tokens"] == 2
    assert summary["trajectories"] == {"10": {"tokens": 1, "resolved": True}, "11": {"tokens": 1, "resolved": True}}

    calls = []
    def rpc(method, *, args):
        calls.append((method, args))
        return [getattr(worker, method)(*args)]
    backend = VLLMBackend(SimpleNamespace(collective_rpc=rpc), _Tokenizer(), model_id="test", parameter_count=1,
                          sampling_params_factory=_SamplingParams, initial_strata=True)
    with backend.stratified_initialization(entries) as first:
        with pytest.raises(ValueError, match="already in use"), backend.stratified_initialization(entries):
            pass
        with pytest.raises(RuntimeError, match="reward failed"):
            with backend.stratified_initialization([(20, 0, 2), (21, 1, 2)]):
                raise RuntimeError("reward failed")
        assert worker._initial_strata.active(10) and not worker._initial_strata.active(20)
        request = GenerationRequest((1,), 2, SamplingConfig(), 10, "continued", uniform_offset=3)
        params = backend._sampling_params(request)
        backend._bind_initial_strata([request], [params])
        assert worker._initial_strata.active(params.seed)
    assert first["registered"] == 2 and not worker._initial_strata.aliases
    assert not backend._initial_strata_seeds and sampler.forward is native
    assert [method for method, _ in calls].count("begin_initial_strata") == 2
    assert first["resolved_after"] == {}
    # A whole-trajectory refresh can register a strict subset of a partition.
    # Untouched TP ranks contribute no resolution; disagreeing sampled ranks fail.
    mismatch = False
    def ranked_rpc(method, *, args):
        row = getattr(worker, method)(*args)
        if method == "end_initial_strata":
            if mismatch:
                return [row, {**row, "trajectories": {"30": {"tokens": 999, "resolved": True}}}]
            return [row, {**row, "trajectories": {}}]
        return [row]
    backend._engine.collective_rpc = ranked_rpc
    with backend.stratified_initialization([(30, 1, 4)]) as refreshed:
        sampler.forward(logits[:1], {0: torch.Generator().manual_seed(30)}, None, None)
    assert refreshed["resolved_after"] == {30: 1}
    mismatch = True
    with pytest.raises(RuntimeError, match="disagree"), backend.stratified_initialization([(30, 1, 4)]):
        sampler.forward(logits[:1], {0: torch.Generator().manual_seed(30)}, None, None)
    assert not backend._initial_strata_seeds and not worker._initial_strata.aliases and sampler.forward is native


class _BeamParams(_SamplingParams):
    pass


def test_vllm_sampling_api_uses_025_and_026_public_import_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm = types.ModuleType("vllm")
    sampling_params = types.ModuleType("vllm.sampling_params")
    setattr(vllm, "SamplingParams", _SamplingParams)
    setattr(vllm, "TokensPrompt", dict)
    setattr(sampling_params, "BeamSearchParams", _BeamParams)
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling_params)

    assert _load_vllm_sampling_api() == (_SamplingParams, dict, _BeamParams)


class _Tokenizer:
    bos_token_id = 9
    eos_token_id = 2
    pad_token_id = 2

    def encode(self, text, add_special_tokens=True):
        values = [ord(value) % 10 for value in text]
        return ([self.bos_token_id] if add_special_tokens else []) + values

    def decode(self, tokens, skip_special_tokens=True):
        return ",".join(str(token) for token in tokens)


class _Engine:
    def __init__(self):
        self.calls = []
        self.closed = False

    @staticmethod
    def _ids(prompt):
        return list(prompt["prompt_token_ids"])

    def generate(self, prompts, *, sampling_params, use_tqdm, **kwargs):
        self.calls.append((prompts, sampling_params, use_tqdm, kwargs))
        params = sampling_params if isinstance(sampling_params, list) else [sampling_params] * len(prompts)
        outputs = []
        for prompt, policy in zip(prompts, params, strict=True):
            prompt_ids = self._ids(prompt)
            if hasattr(policy, "prompt_logprobs"):
                prompt_scores = [None] + [{token: _Logprob(-float(index) / 10)} for index, token in enumerate(prompt_ids[1:], 1)]
                outputs.append(_Output([_Completion([7], [{7: _Logprob(-0.7)}])], prompt_logprobs=prompt_scores,
                                       num_cached_tokens=min(1, len(prompt_ids))))
                continue
            tokens = [int(policy.seed % 5) + 3] * min(2, policy.max_tokens)
            if policy.stop_token_ids and policy.seed == 12:
                tokens[-1] = policy.stop_token_ids[0]
            # The sampled token and, when asked for, as many less likely ones.
            positions = [{token: _Logprob(-0.25), **{20 + other: _Logprob(-1.0) for other in range(getattr(policy, "logprobs", 0) or 0)}}
                         for token in tokens]
            outputs.append(_Output([_Completion(tokens, positions)],
                                   num_cached_tokens=min(2, len(prompt_ids))))
        return outputs

    def shutdown(self):
        self.closed = True


class _BeamEngine(_Engine):
    def beam_search(self, **kwargs):
        self.calls.append(kwargs)
        prompt = self._ids(kwargs["prompts"][0])
        sequence = type("Beam", (), {"tokens": prompt + [4, 2]})()
        return [type("BeamOutput", (), {"sequences": [sequence]})()]


class _FusedEngine(_Engine):
    """Every generated token's base log-probability is -0.4 and its Self-Certainty 3."""

    def __init__(self):
        super().__init__()
        self.fused = {}
        self.rpc_calls = []

    def generate(self, prompts, *, sampling_params, use_tqdm, **kwargs):
        outputs = super().generate(prompts, sampling_params=sampling_params, use_tqdm=use_tqdm, **kwargs)
        call = len(self.calls)
        for index, output in enumerate(outputs):
            output.request_id = f"engine:{call}:{index}"
            tokens = output.outputs[0].token_ids
            self.fused[output.request_id] = ((-0.4,) * len(tokens), (3.0,) * len(tokens))
        return outputs

    def collective_rpc(self, method, *, args):
        self.rpc_calls.append((method, args))
        return [{request_id: self.fused.pop(request_id) for request_id in args[0] if request_id in self.fused}]


class _Fallback:
    model_id = "fake"

    def __init__(self):
        self.slots = 0

    def _count(self, requests):
        self.slots += sum(len(request.prefix) + len(item) for request in requests for item in request.continuations)

    def snapshot(self):
        return SimpleNamespace(score_forward_token_slots=self.slots, estimated_dense_forward_flops=200 * self.slots)

    def sample_batch(self, requests):
        raise AssertionError("fallback generation must not be used")

    def score_batch(self, requests):
        self._count(requests)
        return [tuple(-0.5 for _ in continuation) for request in requests for continuation in request.continuations]

    def token_statistics(self, requests, statistic):
        self._count(requests)
        return [{"tokens": continuation} for request in requests for continuation in request.continuations]


def _backend(*, fallback=None):
    engine = _Engine()
    return VLLMBackend(engine, _Tokenizer(), model_id="fake", parameter_count=100, sampling_params_factory=_SamplingParams,
                       scoring_backend=fallback), engine


def test_vllm_sampling_preserves_per_request_seed_policy_and_order() -> None:
    backend, engine = _backend()
    sampling = SamplingConfig(temperature=0.7, top_p=0.9, top_k=4, eos_token_id=2)
    samples = backend.sample_batch([GenerationRequest((1, 2), 2, sampling, seed, f"r{seed}") for seed in (11, 12)])
    assert [sample.request_id for sample in samples] == ["r11", "r12"]
    assert samples[0].token_logprobs == (-0.25, -0.25)
    assert samples[1].token_ids[-1] == 2
    assert samples[1].finish_reason == "eos"
    params = engine.calls[0][1]
    assert [item.seed for item in params] == [11, 12]
    assert all(item.temperature == 0.7 for item in params)
    assert all(item.top_p == 0.9 and item.top_k == 4 for item in params)
    assert all(item.stop_token_ids == [2] and item.ignore_eos for item in params)
    snapshot = backend.snapshot()
    assert (snapshot.sampled_sequences, snapshot.generated_tokens, snapshot.shared_prefill_tokens_saved,
            snapshot.prefill_tokens) == (2, 4, 4, 0)


def test_vllm_stops_at_single_token_markers_and_labels_the_reference_policy() -> None:
    backend, engine = _backend()
    request = GenerationRequest((1,), 2, SamplingConfig(temperature=0.5, eos_token_id=2), 11, "r",
                                stop_sequences=((5,), (6, 7)), reference_temperature=0.5)
    sample = backend.sample_batch([request])[0]
    # The two-token marker is left to the caller.
    assert engine.calls[0][1][0].stop_token_ids == [2, 5]
    assert sample.reference_token_logprobs == sample.token_logprobs
    assert sample.reference_policy_id == request.reference_policy.policy_id


def test_vllm_fused_reference_eliminates_mh_score_forward() -> None:
    engine = _FusedEngine()
    backend = VLLMBackend(engine, _Tokenizer(), model_id="fake", parameter_count=100,
                          sampling_params_factory=_SamplingParams, fused_logprobs=True)
    result = run_mh_chain(backend, (1,), MHConfig(
        early_rejection=False, suffix_replay=False, alpha=2.0, reward_temperature=None, total_length=2, block_size=2,
        steps_per_block=1, suffix_schedule="uniform", iterations=None), SeedStream(7), base=SamplingConfig(),
        proposal=SamplingConfig(temperature=0.5))
    assert len(result.token_ids) == 2
    assert result.base_token_logprobs == (-0.4, -0.4)
    snapshot = backend.snapshot()
    assert snapshot.score_calls == 0
    assert snapshot.fused_logprobs
    assert snapshot.fused_reference_sequences == 2
    assert snapshot.fused_reference_tokens >= 3
    assert len(engine.rpc_calls) == 2 and not engine.fused


def test_vllm_native_score_extracts_continuation_prompt_logprobs() -> None:
    backend, _ = _backend()

    assert backend.score_batch([ScoreRequest((8, 6), ((4, 5), (), (3,)), SamplingConfig())]) == [(-0.2, -0.3), (), (-0.2,)]
    snapshot = backend.snapshot()
    assert (snapshot.native_score_sequences, snapshot.scored_tokens, snapshot.score_forward_token_slots,
            snapshot.shared_prefill_tokens_saved) == (2, 3, 5, 2)


def test_vllm_nonunit_score_requires_or_uses_exact_fallback() -> None:
    backend, _ = _backend()
    request = ScoreRequest((1,), ((2, 3),), SamplingConfig(temperature=0.7))
    with pytest.raises(ValueError, match="exact scoring_backend"):
        backend.score_batch([request])

    backend, _ = _backend(fallback=_Fallback())
    assert backend.score_batch([request]) == [(-0.5, -0.5)]
    snapshot = backend.snapshot()
    assert snapshot.delegated_score_sequences == 1
    assert snapshot.delegated_score_forward_token_slots == 3
    assert snapshot.delegated_estimated_dense_forward_flops == 600


def test_vllm_delegates_top_k_confidences_to_the_exact_backend() -> None:
    backend, _ = _backend(fallback=_Fallback())
    assert backend.token_statistics([ScoreRequest((1,), ((2, 3),))], TokenStatistic(SamplingConfig(), 5)) == [
        {"tokens": (2, 3)}]
    snapshot = backend.snapshot()
    assert (snapshot.native_score_sequences, snapshot.delegated_score_sequences) == (0, 1)
    assert snapshot.delegated_estimated_dense_forward_flops == 600


def test_vllm_generation_reports_the_statistics_of_its_own_policy() -> None:
    backend, engine = _backend()
    policy = SamplingConfig(eos_token_id=2)
    other_policy, full = TokenStatistic(SamplingConfig(temperature=0.5), 2), TokenStatistic(SamplingConfig())
    top, whole, other = backend.sample_batch([
        GenerationRequest((1,), 2, policy, 11, name, statistic=statistic) for name, statistic in (
            ("top", TokenStatistic(SamplingConfig(), 2)), ("whole", full), ("other", other_policy))])
    assert [params.logprobs for params in engine.calls[0][1]] == [2, 0, 0]
    # The two most likely tokens have log-probabilities -0.25 and -1; the engine never returns the whole vocabulary.
    assert top.token_statistics == (0.625, 0.625) and whole.token_statistics is None
    assert other.token_statistics is None and not backend.records(other_policy, policy) and not backend.records(full, policy)


def test_vllm_public_helpers_and_close() -> None:
    backend, engine = _backend()
    assert backend.encode("ab", add_special_tokens=False) == (7, 8)
    assert backend.decode((1, 2, 3)) == "1,2,3"
    assert backend.execution_config()["compile_sizes"] is None
    config = SimpleNamespace(model_config=SimpleNamespace(quantization="awq"), compilation_config=SimpleNamespace(
        cudagraph_capture_sizes=[1, 2, 4, 8], compile_sizes=[1], compile_ranges_endpoints=[2048]))
    engine.vllm_config = config
    observed = backend.execution_config()
    assert observed["quantization"] == "awq"
    assert observed["compile_sizes"] == [1] and observed["compile_ranges_endpoints"] == [2048]
    assert observed["cudagraph_capture_sizes"] == [1, 2, 4, 8]
    del engine.vllm_config
    engine.llm_engine = SimpleNamespace(vllm_config=config)
    assert backend.execution_config() == observed
    backend.close()
    backend.close()
    assert engine.closed


def test_parameter_count_skips_the_vision_tower_and_multi_token_prediction(tmp_path) -> None:
    from safetensors.numpy import save_file

    shapes = {"model.language_model.layers.0.weight": (2, 3), "model.visual.blocks.0.weight": (4, 4), "mtp.fc.weight": (5,)}
    save_file({name: numpy.zeros(shape, numpy.float32) for name, shape in shapes.items()}, str(tmp_path / "model.safetensors"))
    assert _checkpoint_parameter_count(str(tmp_path)) == 6

    # An exported duplicate is shared only when the model config says so.
    shapes.update({"model.embed_tokens.weight": (7, 2), "lm_head.weight": (7, 2)})
    save_file({name: numpy.zeros(shape, numpy.float32) for name, shape in shapes.items()}, str(tmp_path / "model.safetensors"))
    assert _checkpoint_parameter_count(str(tmp_path)) == 34
    (tmp_path / "config.json").write_text('{"tie_word_embeddings": true}', encoding="utf-8")
    assert _checkpoint_parameter_count(str(tmp_path)) == 20
    (tmp_path / "config.json").write_text(
        '{"tie_word_embeddings": true, "text_config": {"tie_word_embeddings": false}}', encoding="utf-8")
    assert _checkpoint_parameter_count(str(tmp_path)) == 34


def test_vllm_counts_the_engine_preemptions_since_the_backend_started() -> None:
    counters = [SimpleNamespace(name="vllm:num_preemptions", value=3), SimpleNamespace(name="vllm:num_preemptions", value=1),
                SimpleNamespace(name="vllm:prompt_tokens", value=50)]
    backend = VLLMBackend(_Engine(), _Tokenizer(), model_id="fake", parameter_count=100,
                          sampling_params_factory=_SamplingParams, metrics=lambda: counters)
    assert backend.snapshot().preemptions == 0
    counters[0].value, counters[1].value, counters[2].value = 4, 2, 90
    assert backend.snapshot().preemptions == 2
    backend.close()
    # Closing the engine unregisters its metrics.
    counters.clear()
    assert backend.snapshot().preemptions == 2


def test_vllm_direct_greedy_and_sync_beam_generation() -> None:
    engine = _BeamEngine()
    backend = VLLMBackend(engine, _Tokenizer(), model_id="fake", parameter_count=100,
                          sampling_params_factory=_SamplingParams, beam_search_params_factory=_BeamParams)
    assert backend.direct_generate((1,), max_new_tokens=2) == (3, 3)
    assert backend.direct_generate((1,), max_new_tokens=5, num_beams=4) == (4, 2)
    beam_call = engine.calls[-1]
    assert beam_call["params"].beam_width == 4
    assert beam_call["use_tqdm"] is False


def test_vllm_token_penalty_biases_every_request_and_is_scored_exactly_elsewhere() -> None:
    engine, penalty, fallback = _BeamEngine(), TokenPenalty((4, 5), 1.5), _Fallback()
    fallback.model_id = f"fake|{penalty.penalty_id}"
    options = {"model_id": "fake", "parameter_count": 100, "sampling_params_factory": _SamplingParams,
               "token_penalty": penalty}
    backend = VLLMBackend(engine, _Tokenizer(), scoring_backend=fallback, **options)
    assert backend.model_id == fallback.model_id
    backend.sample_batch([GenerationRequest((1,), 2, SamplingConfig(), 11, "r")])
    backend.direct_generate((1,), max_new_tokens=2)
    assert engine.calls[0][1][0].logit_bias == engine.calls[1][1].logit_bias == {4: -1.5, 5: -1.5}
    # Prompt log-probabilities would miss the bias.
    assert backend.score_batch([ScoreRequest((1,), ((2, 3),), SamplingConfig())]) == [(-0.5, -0.5)]
    assert backend.snapshot().native_score_sequences == 0
    with pytest.raises(ValueError, match="offline beam search"):
        backend.direct_generate((1,), max_new_tokens=2, num_beams=2)
    with pytest.raises(ValueError, match="token_penalty"):
        VLLMBackend(_Engine(), _Tokenizer(), **options).score_batch([ScoreRequest((1,), ((2,),), SamplingConfig())])


def test_vllm_continuing_request_draws_independent_randomness() -> None:
    backend, engine = _backend()
    backend.sample_batch([GenerationRequest((1,), 2, SamplingConfig(), 11, "a"),
                          GenerationRequest((1,), 2, SamplingConfig(), 11, "b", uniform_offset=3)])
    assert [params.seed for params in engine.calls[0][1]] == [11, SeedStream(11).derive("uniform-offset", 3)]


def test_vllm_rejects_drafts_and_log_weight_stops() -> None:
    backend, _ = _backend()
    draft = Draft((3,), (-0.1,), (-0.1,), ((0.0, 0.5),))
    with pytest.raises(ValueError, match="replay"):
        backend.sample_batch([GenerationRequest((1,), 2, SamplingConfig(), 1, "r", draft=draft)])
    with pytest.raises(ValueError, match="reference log-probabilities"):
        backend.sample_batch([GenerationRequest((1,), 2, SamplingConfig(), 1, "r",
                                                log_weight_stop=LogWeightStop(2.0, 0.0, -1.0))])


class _AsyncEngine(_Engine):
    """Also a fused worker: every token's base log-probability is -0.4 and its Self-Certainty 3."""

    def __init__(self):
        super().__init__()
        self.active = 0
        self.maximum_active = 0
        self.fused = {}

    async def _stream(self, *, prompt, sampling_params, request_id, **kwargs):
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        await asyncio.sleep(0.02)
        token = int(sampling_params.seed % 5) + 3
        self.active -= 1
        self.fused[request_id] = ((-0.4,), (3.0,))
        yield _Output([_Completion([token], [{token: _Logprob(-0.25)}])], request_id=request_id)

    def generate(self, **kwargs):
        return self._stream(**kwargs)

    async def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        return [{request_id: self.fused.pop(request_id) for request_id in args[0] if request_id in self.fused}]

    async def shutdown(self):
        self.closed = True


def test_the_fused_worker_records_self_certainty_on_either_engine() -> None:
    policy, full = SamplingConfig(eos_token_id=2), TokenStatistic(SamplingConfig())
    options = {"model_id": "fake", "parameter_count": 100, "sampling_params_factory": _SamplingParams, "fused_logprobs": True}
    for engine, backend_class in ((_FusedEngine(), VLLMBackend), (_AsyncEngine(), AsyncVLLMBackend)):
        backend = backend_class(engine, _Tokenizer(), **options)
        # The worker reads the model at temperature 1 only.
        assert backend.records(full, policy) and not backend.records(TokenStatistic(SamplingConfig(temperature=0.5)), policy)
        sample = backend.sample_batch([GenerationRequest((1,), 2, policy, 11, "r", statistic=full)])[0]
        assert sample.token_statistics == (3.0,) * len(sample.token_ids) and not engine.fused
        backend.close()


def test_async_vllm_overlaps_requests_from_independent_callers() -> None:
    engine = _AsyncEngine()
    backend = AsyncVLLMBackend(engine, _Tokenizer(), model_id="fake", parameter_count=100,
                               sampling_params_factory=_SamplingParams)
    start = threading.Barrier(2)

    def generate(seed):
        start.wait()
        return backend.sample_batch([GenerationRequest((1,), 1, SamplingConfig(), seed, f"r{seed}")])[0]

    with ThreadPoolExecutor(max_workers=2) as executor:
        samples = list(executor.map(generate, (1, 2)))

    assert [sample.request_id for sample in samples] == ["r1", "r2"]
    assert engine.maximum_active == 2
    assert backend.snapshot().maximum_in_flight_requests == 2
    assert backend.direct_generate((1,), max_new_tokens=2, num_beams=2) == (3, 3)
    backend.close()
    assert engine.closed
