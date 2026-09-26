from types import SimpleNamespace

import pytest
import torch

from inference_scaling.arllm.algorithms.conditional_is import run_conditional_is
from inference_scaling.arllm.algorithms.config import ConditionalISConfig, PowerMHConfig
from inference_scaling.arllm.algorithms.mh import run_power_mh_chain
from inference_scaling.arllm.backends.transformers_backend import TransformersBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest, ScoreRequest
from inference_scaling.shared.model.generation import generation_budget
from inference_scaling.shared.rng import SeedStream
from inference_scaling.shared.types import pointwise


def _tiny_model(family):
    from transformers import AutoModelForCausalLM, GPT2Config, Qwen2Config

    config = (GPT2Config(vocab_size=31, n_layer=1, n_head=2, n_embd=16, n_positions=128) if family == "gpt2" else
              Qwen2Config(vocab_size=31, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                          hidden_size=16, intermediate_size=32, max_position_embeddings=128, attn_implementation="eager"))
    with torch.random.fork_rng():
        torch.manual_seed(21)
        return AutoModelForCausalLM.from_config(config).eval()


TOKENIZER = SimpleNamespace(pad_token_id=0, eos_token_id=2, bos_token_id=1)


@pytest.mark.parametrize("family", ["gpt2", "qwen2"])
def test_chunked_scoring_matches_complete_causal_context(family):
    model, tokenizer = _tiny_model(family), TOKENIZER
    complete = TransformersBackend(model, tokenizer, device="cpu", max_score_batch_size=8, score_chunk_size=128)
    chunked = TransformersBackend(model, tokenizer, device="cpu", max_score_batch_size=8, score_chunk_size=4,
                                  prefix_cache_bytes=2**24)
    requests = [ScoreRequest((1, 4, 3, 5, 8, 9), ((8, 4, 3, 9, 1, 6, 2), (5, 6)), SamplingConfig(temperature=0.7)),
                ScoreRequest((), ((8,),), SamplingConfig())]
    expected = complete.score_statistics_batch(requests, confidence_top_k=5)
    actual = chunked.score_statistics_batch(requests, confidence_top_k=5)
    for left, right in zip(expected, actual, strict=True):
        assert right.token_topk_confidences == pytest.approx(left.token_topk_confidences, abs=2e-6)
    for left, right in zip(complete.score_batch(requests), chunked.score_batch(requests), strict=True):
        assert right == pytest.approx(left, abs=2e-6)
    # A batch holds at most 8 * 4 padded positions: the two shorter inputs share one padded to 7 and
    # the 12-token input runs alone; the last target has no forward.
    assert chunked.snapshot().score_forward_token_slots == 2 * (2 * 7 + 12)
    requests = [GenerationRequest(tuple(range(1, length + 1)), 4, SamplingConfig(), 8, str(length)) for length in (11, 5)]
    left, right = complete.sample_batch(requests), chunked.sample_batch(requests)
    for expected, actual in zip(left, right, strict=True):
        assert expected.token_ids == actual.token_ids
        assert expected.token_logprobs == pytest.approx(actual.token_logprobs, abs=2e-6)
    # A request resumes a stored KV state over their common prefix.
    first = chunked.sample_batch([GenerationRequest((1, 2, 3, 4, 5), 6, SamplingConfig(), 3, "first")])[0]
    request = GenerationRequest((1, 2, 3, 4, 5) + first.token_ids[:4], 3, SamplingConfig(), 4, "resumed")
    before = chunked.snapshot().prefill_tokens
    resumed = chunked.sample_batch([request])[0]
    fresh = TransformersBackend(model, tokenizer, device="cpu", max_score_batch_size=8, score_chunk_size=4)
    assert chunked.snapshot().prefill_tokens - before == 1
    expected = fresh.sample_batch([request])[0]
    assert resumed.token_ids == expected.token_ids
    assert resumed.token_logprobs == pytest.approx(expected.token_logprobs, abs=2e-6)
    del model, complete, chunked, fresh


def test_context_budget_caps_the_requested_length():
    backend = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(max_position_embeddings=1024)),
                              tokenizer=SimpleNamespace(model_max_length=10**30))
    budget = generation_budget(32768, 1000, [backend], context_window=None)
    assert budget["effective_max_new_tokens"] == 24
    assert budget["requested_max_new_tokens"] == 32768
    with pytest.raises(ValueError, match="fills"):
        generation_budget(32768, 1024, [backend], context_window=None)


def test_every_backend_context_is_respected():
    budget = generation_budget(100, 8, [SimpleNamespace(max_model_len=200), SimpleNamespace(max_model_len=24)],
                               context_window=None)
    assert budget["effective_max_new_tokens"] == 16


@pytest.mark.parametrize("family", ["gpt2", "qwen2"])
def test_prefix_store_in_place_kv_and_fixed_shape_decoding_leave_samples_unchanged(family):
    model, options = _tiny_model(family), {"device": "cpu", "max_score_batch_size": 8, "score_chunk_size": 4}
    backends = [TransformersBackend(model, TOKENIZER, **options),
                TransformersBackend(model, TOKENIZER, **options, prefix_cache_bytes=2**24, in_place_kv=True),
                TransformersBackend(model, TOKENIZER, **options, prefix_cache_bytes=2**24, cuda_graphs=True)]
    prompt = (1, 4, 3, 5, 8, 9)
    # IS-like calls: outputs of three lengths from the prompt, then completions after each output's first block.
    outputs = [backend.sample_batch([GenerationRequest(prompt, 5 + 2 * seed, SamplingConfig(), seed, str(seed))
                                     for seed in range(3)]) for backend in backends]
    blocks = [prompt + output.token_ids[:3] for output in outputs[0]]
    before = [backend.snapshot().prefill_tokens for backend in backends]
    completions = [backend.sample_batch([GenerationRequest(block, 5, SamplingConfig(), 10 + index, f"c{index}")
                                         for index, block in enumerate(blocks) for _ in range(2)]) for backend in backends]
    for runs in zip(*(output + completion for output, completion in zip(outputs, completions))):
        assert all(run.token_ids == runs[0].token_ids for run in runs)
        assert all(run.token_logprobs == pytest.approx(runs[0].token_logprobs, abs=2e-6) for run in runs)
    # Each continued block feeds only its last token; rows and positions are rounded up to powers of two.
    assert [backend.snapshot().prefill_tokens - start for backend, start in zip(backends[1:], before[1:])] == [len(set(blocks))] * 2
    assert backends[2]._static._mask.shape == (8, 1, 1, 16)


def test_growing_cache_matches_the_dynamic_cache_through_crops_and_row_selection():
    from transformers.cache_utils import DynamicCache

    from inference_scaling.arllm.backends.kv_cache import GrowingCache, cache_layers

    growing, dynamic = GrowingCache(), DynamicCache()
    generator, batch = torch.Generator().manual_seed(0), 3
    for count, edit in ((5, None), (1, None), (1, "crop"), (2, "select"), (1, None)):
        for cache in (growing, dynamic):
            if edit == "crop":
                cache.crop(4)
            if edit == "select":
                cache.batch_select_indices(torch.tensor([2, 0]))
        batch = 2 if edit == "select" else batch
        states = [torch.randn((batch, 2, count, 4), generator=generator) for _ in range(2)]
        for cache in (growing, dynamic):
            cache.update(states[0], states[1], 0)
        for left, right in zip(cache_layers(growing), cache_layers(dynamic), strict=True):
            assert torch.equal(left[0], right[0]) and torch.equal(left[1], right[1])


def test_every_speedup_at_once_leaves_is_and_power_mh_unchanged():
    model, options = _tiny_model("qwen2"), {"device": "cpu", "max_score_batch_size": 8, "score_chunk_size": 4}
    with torch.no_grad():
        # Sharper distributions, so that the chain replays suffix tokens and rejects proposals early.
        model.lm_head.weight.mul_(8)
    runs = []
    for on in (False, True):
        backend = TransformersBackend(model, TOKENIZER, **options, prefix_cache_bytes=2**24 * on, in_place_kv=on,
                                      cuda_graphs=on)
        runs.append((backend, run_conditional_is(
            backend, (1, 4, 3), ConditionalISConfig(block_first=on, total_length=8, block_size=2, candidate_count=3,
                                                    rollout_count=2, reward_temperature=0.5),
            pointwise(lambda _prompt, tokens: float(sum(tokens) % 5)), SeedStream(5), sampling=SamplingConfig(eos_token_id=2)),
            run_power_mh_chain(backend, (1, 4, 3), PowerMHConfig(
                early_rejection=on, suffix_replay=on, alpha=4.0, total_length=8, block_size=4, steps_per_block=3,
                suffix_schedule="uniform", iterations=None), SamplingConfig(temperature=0.25, eos_token_id=2), SeedStream(8))))
    (_, is_plain, mh_plain), (fast, is_fast, mh_fast) = runs
    assert mh_fast.early_rejected and fast.snapshot().replayed_tokens
    assert is_fast.token_ids == is_plain.token_ids
    assert [step.selected_index for step in is_fast.steps] == [step.selected_index for step in is_plain.steps]
    assert mh_fast.token_ids == mh_plain.token_ids
    assert [(step.cut, step.accepted) for step in mh_fast.trace] == [(step.cut, step.accepted) for step in mh_plain.trace]
