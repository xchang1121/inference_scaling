"""A real causal-model reference for append/fork cache correctness; no downloads."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from inference_scaling.arllm.backends.transformers_backend import TransformersBackend
from inference_scaling.arllm.config import SamplingConfig
from inference_scaling.arllm.types import GenerationRequest


def test_evidence_append_and_sibling_forks_match_full_recomputation():
    from transformers import AutoModelForCausalLM, GPT2Config

    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = AutoModelForCausalLM.from_config(GPT2Config(
            vocab_size=32, n_layer=1, n_head=2, n_embd=16, n_positions=128))
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=31, bos_token_id=1)
    options = dict(device="cpu", max_score_batch_size=4, score_chunk_size=16)
    cached = TransformersBackend(model, tokenizer, prefix_cache_bytes=2**20, **options)
    reference = TransformersBackend(model, tokenizer, **options)
    policy = SamplingConfig()
    prompt = (1, 4, 5, 6)
    head = cached.sample_batch([GenerationRequest(prompt, 4, policy, 19, "head")])[0]
    checkpoint = prompt + head.token_ids
    before = cached.snapshot()
    # Two distinct references on the same generated checkpoint. The second
    # branch must not inherit the first branch's appended state.
    jobs = [GenerationRequest(checkpoint + evidence, 5, policy, 37, f"branch-{i}")
            for i, evidence in enumerate(((20, 21), (22, 23)))]
    actual = cached.sample_batch(jobs)
    expected = reference.sample_batch(jobs)
    for a, b in zip(actual, expected, strict=True):
        assert a.prefix == b.prefix
        assert a.token_ids == b.token_ids
        assert a.token_logprobs == pytest.approx(b.token_logprobs, abs=2e-6)
    after = cached.snapshot()
    assert after.shared_prefill_tokens_saved > before.shared_prefill_tokens_saved
    assert after.prefill_tokens-before.prefill_tokens < sum(len(j.prefix) for j in jobs)
