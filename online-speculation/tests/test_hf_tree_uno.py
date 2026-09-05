from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from online_speculation.hf_tree_uno import (
    HfTreeUnoRunner, compact_tree_cache, tree_attention_mask,
)
from online_speculation.hf_uno import HfUnoRuntime
from online_speculation.torch_sampling import SamplingConfig
from online_speculation.tree_uno import TreeConfig, build_tree


class _TreeCache:
    def __init__(self):
        self.layers = [SimpleNamespace(keys=torch.empty(1, 1, 0, 1), values=torch.empty(1, 1, 0, 1))]

    def get_seq_length(self):
        return self.layers[0].keys.size(-2)

    def crop(self, length):
        if length < 0:
            length += self.get_seq_length()
        for layer in self.layers:
            layer.keys = layer.keys[..., :length, :]
            layer.values = layer.values[..., :length, :]

    @property
    def tokens(self):
        return self.layers[0].keys.reshape(-1).long().tolist()


class _TreeHistoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(vocab_size=11, hidden_size=4)
        self.last_cache = None

    def forward(self, *, input_ids, past_key_values=None, use_cache=True, attention_mask=None, position_ids=None):
        del use_cache
        cache = _TreeCache() if past_key_values is None else past_key_values
        old = cache.tokens
        new = input_ids[0].tolist()
        all_tokens = old + new
        logits = torch.full((1, len(new), 11), -3.0)
        for i in range(len(new)):
            if attention_mask is None:
                history = all_tokens[:len(old)+i+1]
            else:
                history = [t for t, visible in zip(all_tokens, attention_mask[0, 0, i].tolist(), strict=True) if visible]
                # Position must reflect the logical path, not packed index.
                assert position_ids[0, i] == len(history) - 1
            next_token = (sum(history) + len(history) * 3) % 11
            logits[0, i, next_token] = 3
        packed = torch.tensor(all_tokens, dtype=torch.float32).reshape(1, 1, -1, 1)
        cache.layers[0].keys = packed
        cache.layers[0].values = packed.clone()
        self.last_cache = cache
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _runtime():
    return HfUnoRuntime(
        model=_TreeHistoryModel(), tokenizer=SimpleNamespace(decode=lambda ids, **_: " ".join(map(str, ids))),
        router=SimpleNamespace(set_token_mask=lambda _: None), device=torch.device("cpu"),
        sampling=SamplingConfig(temperature=0), mask_token_id=11,
        stop_token_ids=[], ignore_stop=True,
    )


def test_mask_blocks_siblings_and_descendants() -> None:
    tree = build_tree(9, [[1, 2], [3, 4]], [[0.4, 0.3], [0.4, 0.3]], nodes=7)
    mask = tree_attention_mask(tree, 3, device=torch.device("cpu"))[0, 0]
    assert mask[:, :3].all()
    for i in range(len(tree.nodes)):
        visible = mask[i, 3:].nonzero().reshape(-1).tolist()
        assert visible == sorted(tree.ancestor_indices(i))


def test_dominant_logit_stable_softmax_preserves_subprobabilities():
    logits = torch.tensor([[1000.0, 999.0, 0.0]])
    old = torch.exp(logits - torch.logsumexp(logits, -1, keepdim=True))
    assert old.sum() > 1 + 1e-6  # Reproduce the cancellation hazard.
    ids = logits.topk(2, dim=-1).indices
    prior = logits.softmax(-1).gather(-1, ids).tolist()
    tree = build_tree(4, ids.tolist(), prior, nodes=3)
    assert len(tree.nodes) == 3


def test_compaction_keeps_only_logical_noncontiguous_path() -> None:
    cache = _TreeCache()
    cache.layers[0].keys = torch.arange(9.0).reshape(1, 1, 9, 1)
    cache.layers[0].values = (torch.arange(9.0) + 20).reshape(1, 1, 9, 1)
    compact_tree_cache(cache, 3, (0, 2, 5), 6)
    assert cache.tokens == [0, 1, 2, 3, 5, 8]
    assert cache.layers[0].values.reshape(-1).tolist() == [20, 21, 22, 23, 25, 28]


@pytest.mark.parametrize("nodes", [8, 16, 32])
@pytest.mark.parametrize("online", [False, True])
def test_packed_tree_matches_content_dependent_ar_and_preserves_weights(nodes, online):
    ids = torch.tensor([[3, 5, 7]])
    runtime = _runtime()
    before = runtime.model.anchor.detach().clone()
    result = HfTreeUnoRunner(runtime).generate(
        ids, max_new_tokens=41, seed=53,
        config=TreeConfig(nodes=nodes, online_rank=online),
    )
    reference = _runtime().generate_ar(ids, max_new_tokens=41, seed=0)
    assert result.metrics.output_token_ids == reference.output_token_ids
    assert runtime.model.last_cache.tokens == ids[0].tolist() + list(reference.output_token_ids[:-1])
    assert torch.equal(before, runtime.model.anchor)
    assert result.diagnostics["model_parameters_frozen"]


@pytest.mark.parametrize("limit", [1, 2, 3, 4, 5, 8, 13])
def test_tree_budget_truncation_crops_actual_path(limit):
    ids = torch.tensor([[3, 5, 7]])
    runtime = _runtime()
    result = HfTreeUnoRunner(runtime).generate(
        ids, max_new_tokens=limit, seed=23, config=TreeConfig(nodes=16),
    )
    assert len(result.metrics.output_token_ids) == limit
    assert runtime.model.last_cache.tokens == ids[0].tolist() + list(result.metrics.output_token_ids[:-1])


def test_tree_eos_truncation_keeps_no_speculative_tokens_after_stop():
    ids = torch.tensor([[3, 5, 7]])
    reference = _runtime().generate_ar(ids, max_new_tokens=10, seed=0)
    runtime = _runtime()
    runtime.ignore_stop = False
    runtime.stop_token_ids = {reference.output_token_ids[4]}
    result = HfTreeUnoRunner(runtime).generate(ids, max_new_tokens=40, seed=11, config=TreeConfig())
    assert result.metrics.output_token_ids[-1] in runtime.stop_token_ids
    assert not any(t in runtime.stop_token_ids for t in result.metrics.output_token_ids[:-1])
    assert runtime.model.last_cache.tokens == ids[0].tolist() + list(result.metrics.output_token_ids[:-1])


@pytest.mark.parametrize("online_rank", [False, True])
def test_adaptive_budget_preserves_ar_and_explores_only_declared_actions(online_rank):
    ids = torch.tensor([[3, 5, 7]])
    runtime = _runtime()
    config = TreeConfig(nodes=16, node_budgets=(8, 16, 32), online_rank=online_rank)
    result = HfTreeUnoRunner(runtime).generate(ids, max_new_tokens=83, seed=59, config=config)
    reference = _runtime().generate_ar(ids, max_new_tokens=83, seed=0)
    assert result.metrics.output_token_ids == reference.output_token_ids
    assert runtime.model.last_cache.tokens == ids[0].tolist() + list(result.metrics.output_token_ids[:-1])
    assert set(result.diagnostics["tree_shapes"]) == {"8", "16", "32"}
    assert result.diagnostics["cycle_sync_for_cost"]
    assert all(n >= config.explore_each for n in result.diagnostics["budget_controller"]["counts"].values())
