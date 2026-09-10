from dataclasses import replace
from types import SimpleNamespace

import pytest

from experiments.arllm.reasoning_methods import budget_plan, check_budget, compare_sir, compare_mh, REWARDS
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
