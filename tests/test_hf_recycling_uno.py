from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from online_speculation.hf_recycling_uno import HfRecyclingUnoRunner
from online_speculation.hf_uno import HfUnoRuntime
from online_speculation.recycling import RecyclingConfig
from online_speculation.torch_sampling import SamplingConfig


class _ContentsCache:
    def __init__(self) -> None:
        self.tokens: list[int] = []

    def get_seq_length(self) -> int:
        return len(self.tokens)

    def crop(self, length: int) -> None:
        if length < 0:
            length = max(0, len(self.tokens) + length)
        del self.tokens[length:]


class _HistoryModel(nn.Module):
    """Actual output depends on cached token contents, not only cache length."""
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = SimpleNamespace(vocab_size=11, hidden_size=4)
        self.last_cache = None

    def forward(self, *, input_ids, past_key_values=None, use_cache=True):
        del use_cache
        cache = _ContentsCache() if past_key_values is None else past_key_values
        logits = torch.full((1, input_ids.size(1), 11), -3.0)
        for i, token in enumerate(input_ids[0].tolist()):
            cache.tokens.append(token)
            next_token = (sum(cache.tokens) + len(cache.tokens) * 3) % 11
            logits[0, i, next_token] = 3.0
        self.last_cache = cache
        return SimpleNamespace(logits=logits, past_key_values=cache)


def _runtime(temperature: float = 0) -> HfUnoRuntime:
    return HfUnoRuntime(
        model=_HistoryModel(),
        tokenizer=SimpleNamespace(decode=lambda ids, **_: " ".join(map(str, ids))),
        router=SimpleNamespace(set_token_mask=lambda _: None),
        device=torch.device("cpu"),
        sampling=SamplingConfig(temperature=temperature, top_k=5, top_p=0.98),
        mask_token_id=11, stop_token_ids=[], ignore_stop=True,
    )


@pytest.mark.parametrize("policy", ["always", "bounded", "tps"])
@pytest.mark.parametrize("block", [2, 4, 8])
def test_recycling_matches_history_dependent_ar_and_kv_contents(policy, block) -> None:
    ids = torch.tensor([[3, 5, 7]])
    runtime = _runtime()
    before = runtime.model.anchor.detach().clone()
    result = HfRecyclingUnoRunner(runtime).generate(
        ids, max_new_tokens=37, seed=31,
        config=RecyclingConfig(block_size=block, policy=policy),
    )
    reference = _runtime().generate_ar(ids, max_new_tokens=37, seed=0)
    assert result.metrics.output_token_ids == reference.output_token_ids
    assert runtime.model.last_cache.tokens == ids[0].tolist() + list(reference.output_token_ids[:-1])
    assert result.diagnostics.get("recycle_cycles", 0) > 0
    assert torch.equal(runtime.model.anchor, before)
    assert not runtime.model.anchor.requires_grad
    assert result.metrics.decoder_forwards == (
        2 * result.diagnostics.get("refill_cycles", 0)
        + result.diagnostics.get("recycle_cycles", 0)
    )


@pytest.mark.parametrize("temperature", [0.0, 1.0])
def test_disabled_path_preserves_original_uno_rng_and_tokens(temperature) -> None:
    ids = torch.tensor([[3, 5, 7]])
    result = HfRecyclingUnoRunner(_runtime(temperature)).generate(
        ids, max_new_tokens=31, seed=103,
        config=RecyclingConfig(block_size=4, policy="disabled"),
    )
    reference = _runtime(temperature).generate_uno(
        ids, max_new_tokens=31, seed=103, block_size=4,
    )
    assert result.metrics.output_token_ids == reference.output_token_ids
    assert result.metrics.decoder_forwards == reference.decoder_forwards


@pytest.mark.parametrize("limit", [1, 2, 3, 8, 19])
def test_partial_cycle_token_limit_preserves_exact_kv(limit) -> None:
    ids = torch.tensor([[3, 5, 7]])
    runtime = _runtime()
    result = HfRecyclingUnoRunner(runtime).generate(
        ids, max_new_tokens=limit, seed=7,
        config=RecyclingConfig(block_size=8, policy="always"),
    )
    assert len(result.metrics.output_token_ids) == limit
    assert runtime.model.last_cache.tokens == ids[0].tolist() + list(result.metrics.output_token_ids[:-1])


def test_stop_token_truncates_recycling_cycle_and_retains_valid_kv() -> None:
    ids = torch.tensor([[3, 5, 7]])
    reference = _runtime().generate_ar(ids, max_new_tokens=10, seed=0)
    runtime = _runtime()
    runtime.ignore_stop = False
    runtime.stop_token_ids = {reference.output_token_ids[4]}
    result = HfRecyclingUnoRunner(runtime).generate(
        ids, max_new_tokens=40, seed=11,
        config=RecyclingConfig(block_size=8, policy="always"),
    )
    assert result.metrics.output_token_ids[-1] in runtime.stop_token_ids
    assert not any(t in runtime.stop_token_ids for t in result.metrics.output_token_ids[:-1])
    assert runtime.model.last_cache.tokens == ids[0].tolist() + list(result.metrics.output_token_ids[:-1])


@pytest.mark.parametrize("scale", [0.0, 0.5, 1.0])
def test_warmstart_changes_noise_but_preserves_history_target_and_kv(scale) -> None:
    ids = torch.tensor([[3, 5, 7]])
    runtime = _runtime()
    result = HfRecyclingUnoRunner(runtime).generate(
        ids, max_new_tokens=43, seed=47,
        config=RecyclingConfig(
            block_size=8, policy="warmstart", noise_lora_scale=scale,
        ),
    )
    reference = _runtime().generate_ar(ids, max_new_tokens=43, seed=0)
    assert result.metrics.output_token_ids == reference.output_token_ids
    assert result.diagnostics.get("warmstart_input_tokens", 0) > 0
    assert result.diagnostics.get("recycle_cycles", 0) == 0
    assert runtime.model.last_cache.tokens == ids[0].tolist() + list(reference.output_token_ids[:-1])
