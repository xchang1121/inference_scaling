import itertools
import math

import pytest
import torch

from blockspec.decoding import generate_ar
from blockspec.model import Decoder, ModelConfig, cache_length
from blockspec.online import OnlineConfig, OnlineLearner
from blockspec.tree import build_tree, compact_tree_cache, generate_tree, traverse_greedy, traverse_target


def tiny():
    torch.manual_seed(8)
    return Decoder(ModelConfig(vocab_size=5, hidden_size=16, intermediate_size=24,
                                num_hidden_layers=2, num_attention_heads=2,
                                num_key_value_heads=1, head_dim=8, adapter_rank=2)).double()


@pytest.mark.parametrize("budget", [1, 2, 4, 8, 15])
def test_heap_selects_globally_best_prefixes(budget):
    q = torch.tensor([[.6, .3, .1], [.1, .4, .5], [.2, .6, .2]], dtype=torch.float64)
    tree = build_tree(0, q, top_k=2, prefix_budget=budget)
    candidates = []
    choices = q.topk(2, dim=-1).indices.tolist()
    for depth in range(1, 4):
        for path in itertools.product(*choices[:depth]):
            score = sum(math.log(float(q[i, t])) for i, t in enumerate(path))
            candidates.append((-score, path))
    expected = [path for score, path in sorted(candidates)[:budget - 1]]
    got = [tuple(tree.tokens[i] for i in tree.path(node)[1:]) for node in range(1, len(tree.tokens))]
    assert got == expected
    assert all(tree.parents[i] < i for i in range(1, len(tree.tokens)))


def test_tree_attention_and_cache_equal_every_individual_path():
    model = tiny()
    prefix = torch.tensor([[0, 1, 3]])
    _, cache = model(prefix, return_cache=True)
    tree = build_tree(2, torch.tensor([[.5, .3, .1, .05, .05], [.2, .1, .5, .1, .1]]),
                      top_k=2, prefix_budget=7)
    ids, positions, allowed = tree.layout(3, device="cpu")
    logits, verified = model(ids, positions=positions, allowed=allowed, cache=cache, return_cache=True)
    for node in range(len(tree.tokens)):
        path = tree.path(node)
        tokens = torch.tensor([[tree.tokens[i] for i in path]])
        direct, direct_cache = model(tokens, cache=cache, return_cache=True)
        torch.testing.assert_close(logits[:, node], direct[:, -1], atol=2e-15, rtol=2e-14)
        compact = compact_tree_cache(verified, 3, path)
        assert cache_length(compact) == 3 + len(path)
        for layer_a, layer_b in zip(compact, direct_cache):
            for a, b in zip(layer_a, layer_b):
                torch.testing.assert_close(a, b, atol=2e-15, rtol=2e-14)


def test_target_traversal_samples_missing_child_and_stops():
    tree = build_tree(0, torch.tensor([[0., 1., 0.], [0., 1., 0.]]), top_k=1, prefix_budget=3)
    # Root -> token 1 is present, token 2 after it is absent and must still emit.
    target = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])
    result = traverse_target(tree, target, budget=10)
    assert result.tokens == [0, 1, 2]
    assert result.path == [0, 1] and result.teachers == [0, 1] and result.matched == 1


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 17])
@pytest.mark.parametrize("prefix_budget", [1, 4, 12])
def test_tree_greedy_matches_ar_for_tail_and_branch_shapes(budget, prefix_budget):
    model = tiny()
    prompt = torch.tensor([[0, 1, 3]])
    ar = generate_ar(model, prompt, budget)
    tree = generate_tree(model, prompt, budget, block_size=4, top_k=3, prefix_budget=prefix_budget)
    assert tree.tokens == ar.tokens


def test_tree_eos_and_true_online_continuation():
    model = tiny()
    prompt = torch.tensor([[0, 1, 3]])
    ar = generate_ar(model, prompt, 20)
    eos = ar.tokens[3]
    assert generate_tree(model, prompt, 20, eos_id=eos).tokens == generate_ar(model, prompt, 20, eos_id=eos).tokens
    learner = OnlineLearner(model, OnlineConfig(stride=1, replay_blocks=2, loss="forward_kl"))
    generated = generate_tree(model, prompt, 20, block_size=4, prefix_budget=12, learner=learner)
    assert generated.tokens == ar.tokens
    assert generated.updates > 0 and not learner.replay


def test_tree_target_sampling_probabilities_do_not_depend_on_tree_scores(monkeypatch):
    # Exhaustively integrate the first target draw over the three vocabulary items.
    tree = build_tree(0, torch.tensor([[.9, .1, 0.]]), top_k=1, prefix_budget=2)
    p = torch.tensor([[.2, .3, .5], [.6, .1, .3]], dtype=torch.float64)
    law = torch.zeros(3, dtype=torch.float64)
    for token in range(3):
        def forced_draw(distribution, generator, t=token):
            torch.testing.assert_close(distribution, p[0], atol=0, rtol=0)
            return torch.tensor(t)
        monkeypatch.setattr("blockspec.tree.draw", forced_draw)
        result = traverse_target(tree, p, budget=2)
        law[result.tokens[1]] += p[0, token]
    torch.testing.assert_close(law, p[0], atol=0, rtol=0)


def test_tree_greedy_specialization_equals_one_hot_traversal():
    tree = build_tree(0, torch.tensor([[.6, .4], [.3, .7]]), top_k=2, prefix_budget=6)
    for targets in itertools.product(range(2), repeat=len(tree.tokens)):
        ids = torch.tensor(targets)
        p = torch.nn.functional.one_hot(ids, 2).float()
        assert traverse_greedy(tree, ids, budget=4) == traverse_target(tree, p, budget=4)
