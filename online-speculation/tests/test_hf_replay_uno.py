from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import torch
from torch import nn

from online_speculation.hf_replay_uno import (
    HfReplayUnoRunner,
    ReplayRuntimeConfig,
)
from online_speculation.hf_uno import HfUnoRuntime
from online_speculation.replay_cache import (
    CostAwareReplayRouter,
    ReplayCacheConfig,
    ReplayRouteConfig,
    VerifierReplayCache,
)
from online_speculation.torch_sampling import SamplingConfig


class _Cache:
    def __init__(self) -> None:
        self.length = 0

    def get_seq_length(self) -> int:
        return self.length

    def crop(self, length: int) -> None:
        if length < 0:
            self.length = max(0, self.length + length)
        else:
            self.length = min(self.length, length)


class _TinyMarkovModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(vocab_size=6, hidden_size=4)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: _Cache | None = None,
        use_cache: bool,
    ):
        del use_cache
        cache = _Cache() if past_key_values is None else past_key_values
        cache.length += input_ids.size(1)
        next_ids = (input_ids + 1) % self.config.vocab_size
        logits = torch.full(
            (*input_ids.shape, self.config.vocab_size),
            -2.0,
            device=input_ids.device,
        )
        logits.scatter_(2, next_ids.unsqueeze(-1), 3.0)
        return SimpleNamespace(
            logits=logits + self.anchor * 0.0,
            past_key_values=cache,
        )

    @contextmanager
    def disable_adapter(self):
        try:
            yield
        finally:
            self.anchor.requires_grad_(True)


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return " ".join(str(token) for token in token_ids)


class _LoraRouter:
    def __init__(self) -> None:
        self.mask = None

    def set_token_mask(self, mask: torch.Tensor) -> None:
        self.mask = mask.clone()


def _runtime(*, temperature: float) -> HfUnoRuntime:
    model = _TinyMarkovModel()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return HfUnoRuntime(
        model=model,
        tokenizer=_Tokenizer(),
        router=_LoraRouter(),
        device=torch.device("cpu"),
        sampling=SamplingConfig(
            temperature=temperature,
            top_k=3,
            top_p=0.95,
        ),
        mask_token_id=6,
        stop_token_ids=[],
        ignore_stop=True,
    )


def _runner(
    *,
    temperature: float,
    replay_cache: VerifierReplayCache | None = None,
) -> HfReplayUnoRunner:
    namespace = "tiny-markov@v1|greedy" if temperature <= 0 else "tiny-markov@v1|sample"
    cache = replay_cache or VerifierReplayCache(
        namespace=namespace,
        config=ReplayCacheConfig(
            min_suffix_length=2,
            max_suffix_length=8,
            max_continuation_length=3,
            min_confidence=0.5,
        ),
    )
    router = CostAwareReplayRouter(
        namespace=namespace,
        config=ReplayRouteConfig(
            min_match_length=2,
            min_proposal_tokens=1,
            min_cache_confidence=0.5,
            exploration_trials_per_match_length=1,
            probe_interval=8,
            ema_decay=0.0,
            throughput_margin=0.0,
        ),
    )
    return HfReplayUnoRunner(
        _runtime(temperature=temperature),
        replay_cache=cache,
        router=router,
    )


def test_empty_cache_path_is_bitwise_static_uno_for_greedy() -> None:
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    hybrid = _runner(temperature=0.0).generate(
        input_ids,
        max_new_tokens=23,
        seed=101,
        config=ReplayRuntimeConfig(block_size=4),
    )
    static = _runtime(temperature=0.0).generate_uno(
        input_ids,
        max_new_tokens=23,
        block_size=4,
        seed=101,
    )
    assert hybrid.metrics.output_token_ids == static.output_token_ids
    assert hybrid.metrics.decoder_forwards == static.decoder_forwards
    assert hybrid.diagnostics.replay_cycles == 0
    assert hybrid.diagnostics.static_cycles == hybrid.metrics.cycles
    assert hybrid.diagnostics.cache_records_added > 0


def test_empty_cache_path_preserves_static_stochastic_random_state() -> None:
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    hybrid = _runner(temperature=1.0).generate(
        input_ids,
        max_new_tokens=23,
        seed=103,
        config=ReplayRuntimeConfig(block_size=4),
    )
    static = _runtime(temperature=1.0).generate_uno(
        input_ids,
        max_new_tokens=23,
        block_size=4,
        seed=103,
    )
    assert hybrid.metrics.output_token_ids == static.output_token_ids
    assert hybrid.metrics.decoder_forwards == static.decoder_forwards
    assert hybrid.diagnostics.exactness_mode == "filtered-psi-spec-delta-correction"


def test_second_greedy_request_uses_one_forward_replay_and_matches_ar() -> None:
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    runner = _runner(temperature=0.0)
    first = runner.generate(
        input_ids,
        max_new_tokens=23,
        seed=107,
        config=ReplayRuntimeConfig(block_size=4),
    )
    second = runner.generate(
        input_ids,
        max_new_tokens=23,
        seed=109,
        config=ReplayRuntimeConfig(block_size=4),
    )
    ar = _runtime(temperature=0.0).generate_ar(
        input_ids,
        max_new_tokens=23,
        seed=999,
    )
    assert second.metrics.output_token_ids == ar.output_token_ids
    assert second.diagnostics.replay_cycles > 0
    assert second.diagnostics.static_cycles == 0
    assert second.diagnostics.cache_miss_cycles == 0
    assert second.metrics.decoder_forwards < first.metrics.decoder_forwards
    assert second.diagnostics.replay_accepted_tokens > 0
    assert second.diagnostics.replay_tokens_per_forward > 1.0


def test_wrong_replay_is_corrected_and_then_falls_back_losslessly() -> None:
    namespace = "tiny-markov@v1|greedy"
    replay_cache = VerifierReplayCache(
        namespace=namespace,
        config=ReplayCacheConfig(
            min_suffix_length=2,
            max_suffix_length=8,
            max_continuation_length=3,
            min_confidence=0.5,
        ),
    )
    replay_cache.observe_sequence(
        prompt_tokens=(0, 1, 2),
        verified_completion_tokens=(3, 3, 3, 3, 3, 3),
    )
    runner = _runner(temperature=0.0, replay_cache=replay_cache)
    runner.router.observe_static(committed_tokens=3, forwards=2)
    input_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    actual = runner.generate(
        input_ids,
        max_new_tokens=17,
        seed=113,
        config=ReplayRuntimeConfig(block_size=4, observe_after_request=False),
    )
    ar = _runtime(temperature=0.0).generate_ar(
        input_ids,
        max_new_tokens=17,
        seed=999,
    )
    assert actual.metrics.output_token_ids == ar.output_token_ids
    assert actual.diagnostics.replay_cycles >= 1
    assert (
        actual.diagnostics.replay_accepted_tokens
        < actual.diagnostics.replay_attempted_tokens
    )
    assert actual.diagnostics.static_cycles >= 1
    assert actual.diagnostics.cache_records_added == 0


def test_invalid_cached_token_never_reaches_embedding_table() -> None:
    namespace = "tiny-markov@v1|greedy"
    replay_cache = VerifierReplayCache(
        namespace=namespace,
        config=ReplayCacheConfig(
            min_suffix_length=2,
            max_suffix_length=8,
            max_continuation_length=3,
            min_confidence=0.5,
        ),
    )
    replay_cache.observe_sequence(
        prompt_tokens=(0, 1, 2),
        verified_completion_tokens=(3, 99, 99),
    )
    runner = _runner(temperature=0.0, replay_cache=replay_cache)
    runner.router.observe_static(committed_tokens=3, forwards=2)
    result = runner.generate(
        torch.tensor([[0, 1, 2]], dtype=torch.long),
        max_new_tokens=8,
        seed=127,
        config=ReplayRuntimeConfig(block_size=4, observe_after_request=False),
    )
    assert result.metrics.output_tokens == 8
    assert result.diagnostics.invalid_candidate_cycles >= 1
    assert result.diagnostics.route_reason_counts["invalid-candidate-token"] >= 1


def test_runner_rejects_cross_namespace_cache_router_pair() -> None:
    cache = VerifierReplayCache(
        namespace="model-a",
        config=ReplayCacheConfig(),
    )
    router = CostAwareReplayRouter(
        namespace="model-b",
        config=ReplayRouteConfig(),
    )
    try:
        HfReplayUnoRunner(
            _runtime(temperature=0.0),
            replay_cache=cache,
            router=router,
        )
    except ValueError as error:
        assert "namespaces" in str(error)
    else:
        raise AssertionError("cross-namespace replay state was accepted")
