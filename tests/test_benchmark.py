from dataclasses import replace

import pytest
import torch

from blockspec.benchmark import (BenchmarkConfig, aggregate, benchmark_streams,
                                compare_tokens, continuation_prompts)
from blockspec.checkpoint import adapter_state, base_fingerprint
from blockspec.decoding import Generation
from blockspec.model import Decoder, ModelConfig
from blockspec.online import OnlineConfig


def test_aggregate_counts_all_time_not_mean_tps():
    rows = [Generation([1] * 10, 1, 10, 10).summary(),
            Generation([1] * 30, 9, 30, 30, updates=2, update_seconds=3).summary()]
    result = aggregate(rows, setup_seconds=2)
    assert result["tps"] == 4
    assert result["tps_including_learner_setup"] == 40 / 12
    assert result["updates"] == 2 and result["update_seconds"] == 3


def test_continuation_selection_is_explicit_and_copied():
    sequences = [torch.arange(2), torch.arange(6), torch.arange(7)]
    prompts = continuation_prompts(sequences, count=2, length=4)
    assert [p.tolist() for p in prompts] == [[[0, 1, 2, 3]], [[0, 1, 2, 3]]]
    prompts[0][0, 0] = 7
    assert sequences[1][0] == 0
    with pytest.raises(ValueError):
        continuation_prompts(sequences, count=3, length=4)


def test_comparison_does_not_hide_length_or_token_mismatches():
    assert compare_tokens([1, 2, 3], [1, 4, 3])["common_prefix"] == 1
    assert not compare_tokens([1, 2], [1, 2, 3])["identical"]
    assert compare_tokens([], [])["identical"]


@pytest.mark.parametrize("sampler", ["linear", "tree"])
@pytest.mark.parametrize("last_layers", [None, 1])
def test_balanced_streams_restore_weights_and_keep_online_across_requests(sampler, last_layers):
    torch.manual_seed(17)
    model = Decoder(ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=2, adapter_rank=2)).train_adapters_only()
    before, frozen = adapter_state(model), base_fingerprint(model)
    requires_grad = [p.requires_grad for p in model.parameters()]
    progress = []
    result = benchmark_streams(model, [torch.tensor([[0, 1]]), torch.tensor([[0, 2]])],
                               BenchmarkConfig(tokens=8, block_size=3, warmup_tokens=4,
                                               sampler=sampler, prefix_budget=5),
                               OnlineConfig(stride=1, replay_blocks=1, train_last_layers=last_layers),
                               progress=progress.append)
    assert result["greedy_identical"]
    assert result["arms"]["ar"]["tokens"] == 32
    assert result["arms"]["online"]["updates"] > 0
    assert all(result["online_adapter_changed_per_stream"])
    online_requests = [p for p in progress if p["arm"] == "online"]
    assert online_requests[1]["adapter_version"] > online_requests[0]["adapter_version"]
    assert [(p["repeat"], p["arm"]) for p in progress[::2]] == [
        (0, "ar"), (0, "static"), (0, "online"), (1, "online"), (1, "static"), (1, "ar")]
    assert all(torch.equal(v, adapter_state(model)[n]) for n, v in before.items())
    assert base_fingerprint(model) == frozen
    assert requires_grad == [p.requires_grad for p in model.parameters()]
    count = sum(p.numel() for p in model.adapter_parameters())
    assert result["online_trainable_parameters"] == count / (2 if last_layers else 1)


def test_benchmark_restores_adapter_on_exception():
    model = Decoder(ModelConfig(vocab_size=8, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=1, adapter_rank=2))
    before = adapter_state(model)
    requires_grad = [p.requires_grad for p in model.parameters()]
    def stop(_):
        raise RuntimeError("test interruption")
    with pytest.raises(RuntimeError, match="test interruption"):
        benchmark_streams(model, [torch.tensor([[0, 1]])],
                          BenchmarkConfig(tokens=4, warmup_tokens=4), progress=stop)
    assert all(torch.equal(v, adapter_state(model)[n]) for n, v in before.items())
    assert requires_grad == [p.requires_grad for p in model.parameters()]


@pytest.mark.parametrize("key,value", [("repeats", 1), ("repeats", 3), ("tokens", 0),
                                      ("sampler", "invalid"), ("block_size", 1)])
def test_invalid_benchmark_configuration(key, value):
    with pytest.raises(ValueError):
        replace(BenchmarkConfig(), **{key: value})
