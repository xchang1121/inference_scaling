from dataclasses import replace
from types import SimpleNamespace

import pytest

from experiments.arllm.reasoning_methods import (
    budget_plan, check_budget, compare_sir, compare_mh, majority_index, REWARDS, sampling_policy, reward_temperature,
)
from experiments.arllm.request_reuse import ColdCostRequestReplay
from experiments.arllm.reasoning_benchmark import build_parser
from inference_scaling.arllm.types import GenerationRequest
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend
from inference_scaling.arllm.backends.transformers_backend import TransformersBackendSnapshot


class CountedBackend(TabularAutoregressiveBackend):
    parameter_count = 7
    tokenizer = SimpleNamespace(eos_token_id=2, get_vocab=lambda: {"a": 0, "b": 1, "eos": 2})

    def __init__(self):
        super().__init__({}, fallback=(0.6, 0.2, 0.2))
        self.counts = TransformersBackendSnapshot(*([0] * 10))

    def _charge(self, key, count):
        self.counts = replace(self.counts, **{key: getattr(self.counts, key) + count},
                             estimated_dense_forward_flops=self.counts.estimated_dense_forward_flops + 14 * count)

    def sample_batch(self, requests):
        samples = super().sample_batch(requests)
        self._charge("generation_forward_token_slots", sum(len(r.prefix) + len(s.token_ids) - 1 for r, s in zip(requests, samples, strict=True)))
        return samples

    def score_batch(self, requests):
        self._charge("score_forward_token_slots", sum(len(r.prefix) + len(tokens) - 1 for r in requests for tokens in r.continuations))
        return super().score_batch(requests)

    def score_statistics_batch(self, requests, **kwargs):
        self.score_batch(requests)
        return [SimpleNamespace(token_topk_confidences=tuple(1 + 0.1 * token for token in tokens)) for r in requests for tokens in r.continuations]

    def decode(self, tokens, **kwargs):
        return str(tokens[0]) if tokens else ""

    def encode(self, text, **kwargs):
        return ()

    def snapshot(self):
        return self.counts


class Judge:
    def grade(self, prediction, reference):
        return {"correct": prediction == reference, "parseable": bool(prediction)}

    def equivalent(self, left, right):
        return bool(left and left == right)

    def answer_key(self, text):
        return text or None


@pytest.mark.parametrize("flag,value", [("--proposal-model", "org/other"), ("--mh-iterations", "1"),
                                       ("--thinking-mode", "enabled")])
def test_comparison_cli_rejects_unimplemented_shared_overrides(flag, value):
    with pytest.raises(SystemExit):
        build_parser().parse_args([flag, value])


@pytest.mark.parametrize("flag", ["--limit", "--draws", "--budgets", "--candidate-counts"])
def test_comparison_cli_rejects_nonpositive_work_before_model_loading(flag):
    with pytest.raises(SystemExit):
        build_parser().parse_args([flag, "0"])


def test_comparison_cli_keeps_generic_model_and_output_options():
    args = build_parser().parse_args(["--model", "org/model", "--model-revision", "fixed", "--allow-download",
                                     "--max-new-tokens", "32768", "--thinking-format", "json"])
    assert args.model == "org/model" and args.model_revision == "fixed" and args.allow_download
    assert args.max_new_tokens == 32768 and args.thinking_format == "json"


def test_sampling_configuration_is_applied_and_mh_support_is_checked():
    config = {"sampling": {"temperature": 0.7, "top_p": 0.9, "top_k": 20}}
    assert sampling_policy(config, eos_token_id=2) == SamplingConfig(0.7, 0.9, 20, 2)
    with pytest.raises(ValueError, match="reference policy"):
        sampling_policy(config, require_full_support=True)
    with pytest.raises(ValueError, match="unsupported"):
        sampling_policy({"sampling": {"temprature": 0.6}})


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("inf"), float("nan")])
def test_reward_temperature_rejects_invalid_values(temperature):
    with pytest.raises(ValueError, match="finite and positive"):
        reward_temperature("consilience", {"comparison": {"consilience_temperature": temperature}})


def test_sir_log_probability_reuses_the_configured_reward_scale():
    backend = CountedBackend()
    samples = [{"token_ids": (0, 2), "token_logprobs": (-0.4, -1.0)},
               {"token_ids": (1, 2), "token_logprobs": (-1.4, -1.0)}]
    result = compare_sir(backend=backend, judge=Judge(), reference="0", prompt=(0,),
        config={"reward": {"logprob_scale": 2.0}}, plan=budget_plan(128, 2, 1, 16),
        samples=samples, pilots=[], source="sequence_log_probability", seed=8,
        render_output=output, score_cache={})
    assert result["rewards"] == [-2.8, -4.8]
    assert backend.snapshot().score_forward_token_slots == 0


def output(backend, prompt, tokens, config):
    return {"content_text": str(tokens[0]), "thinking_status": "complete"}


def test_budget_reserves_all_generation_scoring_and_pilot_tokens():
    plan = budget_plan(32768, 2, 120, 32768)
    assert plan["max_new_tokens"] == 8072
    assert plan["reserved_forward_tokens"] == 32768
    with pytest.raises(ValueError):
        budget_plan(4, 2, 100, 1)
    with pytest.raises(RuntimeError, match="exceeds"):
        check_budget({"generation_forward_token_slots": 10, "score_forward_token_slots": 2}, 11)


def test_majority_ignores_missing_answers_and_uses_stable_ties():
    assert majority_index(["1", "2", "2"], Judge()) == 1
    assert majority_index(["1", "2"], Judge()) == 0
    assert majority_index(["", "", "2"], Judge()) == 2


def test_identical_request_reuse_preserves_samples_and_cold_cost():
    raw, independent = CountedBackend(), CountedBackend()
    replay = ColdCostRequestReplay(raw)
    request = GenerationRequest((0,), 32, SamplingConfig(eos_token_id=2), 81, "first")
    first = replay.sample_batch([request])[0]
    expected_first = independent.sample_batch([request])[0]
    assert first == expected_first
    same = replace(request, request_id="second")
    assert replay.sample_batch([same])[0] == independent.sample_batch([same])[0]
    assert replay.snapshot() == independent.snapshot()
    assert raw.snapshot().generation_forward_token_slots * 2 == replay.snapshot().generation_forward_token_slots
    assert replay.cache_hits == 1


def test_completed_eos_request_can_be_reused_at_a_longer_limit():
    raw, independent = CountedBackend(), CountedBackend()
    replay = ColdCostRequestReplay(raw)
    request = GenerationRequest((0,), 32, SamplingConfig(eos_token_id=2), 81, "first")
    sample = replay.sample_batch([request])[0]
    independent.sample_batch([request])
    assert sample.finish_reason == "eos"
    longer = replace(request, max_new_tokens=64)
    assert replay.sample_batch([longer])[0] == independent.sample_batch([longer])[0]
    assert replay.snapshot() == independent.snapshot()
    assert replay.cache_hits == 1


@pytest.mark.parametrize("source", REWARDS)
def test_mh_request_reuse_preserves_whole_chain_and_accounting(source):
    backend = CountedBackend()
    replay = ColdCostRequestReplay(backend)
    sample = {"token_ids": (0, 2), "token_logprobs": (-0.4, -1.0)}
    options = dict(judge=Judge(), prompt=(0,), reference="0", config={"sampling": {"temperature": 0.6}},
                   plan=budget_plan(128, 2, 1, 16), pilots=[sample, sample], source=source, seed=17, render_output=output)
    first = compare_mh(backend=replay, **options)
    second = compare_mh(backend=replay, **options)
    assert first["trace"] == second["trace"]
    assert first["content"] == second["content"]
    assert first["cost"] == second["cost"]
    assert replay.cache_hits >= 1


@pytest.mark.parametrize("source", REWARDS)
def test_sir_reward_and_budget_ignore_gold_until_evaluation(source):
    backend = CountedBackend()
    samples = [{"token_ids": (0, 2), "token_logprobs": (-0.4, -1.0)},
               {"token_ids": (1, 2), "token_logprobs": (-1.4, -1.0)}]
    options = dict(backend=backend, judge=Judge(), prompt=(0,), config={"sampling": {"temperature": 0.6}},
                   plan=budget_plan(128, 2, 1, 16), samples=samples, pilots=samples,
                   source=source, seed=8, render_output=output, score_cache={})
    left = compare_sir(reference="0", **options)
    right = compare_sir(reference="1", **options)
    assert left["probabilities"] == right["probabilities"]
    assert left["selected_index"] == right["selected_index"]
    assert left["correct"] != right["correct"]
    assert left["used_forward_tokens"] <= 128


@pytest.mark.parametrize("source", REWARDS)
def test_mh_all_model_rewards_run_with_same_token_budget(source):
    backend = CountedBackend()
    sample = {"token_ids": (0, 2), "token_logprobs": (-0.4, -1.0)}
    result = compare_mh(backend=backend, judge=Judge(), prompt=(0,), reference="0",
        config={"sampling": {"temperature": 0.6}}, plan=budget_plan(128, 2, 1, 16),
        pilots=[sample, sample], source=source, seed=17, render_output=output)
    assert result["updates"] == 1
    assert result["used_forward_tokens"] <= 128
    assert result["cost"]["estimated_dense_forward_flops"] == result["used_forward_tokens"] * 14
