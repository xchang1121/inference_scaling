import importlib.util
from pathlib import Path

import pytest
import torch

from blockspec.feedback import Feedback
from blockspec.online import Feedback as LegacyFeedback
from blockspec.parallel import DualViewConfig, DualViewDecoder, MaskedAttentionBranch, generate
from blockspec.parallel.audit import AuditSampler, METRICS, SuffixAudit, audit_summary, paired_audit_intervals
from blockspec.parallel.feedback import OnlineFeedback
from blockspec.sampling import SamplingConfig


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"))]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("temperature", [0., 1.])
@torch.no_grad()
def test_common_prefix_audit_preserves_generation_and_equal_checkpoint(device, temperature):
    torch.manual_seed(860)
    net = DualViewDecoder(DualViewConfig(vocab_size=17, hidden_size=16, intermediate_size=24,
                                        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                        head_dim=8)).to(device).eval().requires_grad_(False)
    alternative = {name: p for name, p in net.named_parameters() if name.startswith("layers.1.attention.draft.")}
    sampling = SamplingConfig(temperature, 5, .8)
    saved = {name: p.clone() for name, p in net.state_dict().items()}
    sampler = AuditSampler(sampling)
    inspector = SuffixAudit(net, 1, alternative, 4, sampling, recorded_logits=lambda: sampler.last_logits)
    prompt = torch.tensor([[3, 4, 5]], device=device)
    kwargs = {"sampling": sampling, "block_size": 4}
    expected = generate(MaskedAttentionBranch(net), prompt, 24,
                        generator=torch.Generator(device=device).manual_seed(861), **kwargs)
    actual = generate(MaskedAttentionBranch(net), prompt, 24, feedback=OnlineFeedback(learner=inspector),
                      generator=torch.Generator(device=device).manual_seed(861), sampler=sampler, **kwargs)
    assert actual.tokens == expected.tokens
    assert actual.accepted_per_round == expected.accepted_per_round
    assert actual.updates == 0 and inspector.feedback_blocks > 0
    assert inspector.max_replay_logit_error == 0.
    assert all(torch.equal(value, net.state_dict()[name]) for name, value in saved.items())
    summary = audit_summary(inspector.totals)
    assert summary["positions"] == sum(row["positions"] for row in summary["by_depth"])
    for pair in ((0, 1), (2, 3), (4, 5)):
        assert summary["mean"][METRICS[pair[0]]] == summary["mean"][METRICS[pair[1]]]
    assert 0 <= summary["mean"]["original_sampling_tv"] <= 1
    intervals = paired_audit_intervals([summary, summary], repeats=10)
    assert all(row["difference"] == 0. and row["paired_request_ci95"] == [0., 0.] for row in intervals.values())
    assert Feedback is LegacyFeedback


def test_prompt_windows_are_disjoint_and_bounded(tmp_path):
    path = Path(__file__).parents[1] / "scripts" / "dual_view.py"
    spec = importlib.util.spec_from_file_location("dual_view_prompt_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    data = tmp_path / "questions.jsonl"
    data.write_text('\n'.join('{"question": "question %d"}' % i for i in range(12)))
    assert module.prompt_texts(data, 4, offset=8) == ["question 8", "question 9", "question 10", "question 11"]
    assert not set(module.prompt_texts(data, 8)) & set(module.prompt_texts(data, 4, offset=8))
    with pytest.raises(ValueError):
        module.prompt_texts(data, 5, offset=8)
    with pytest.raises(ValueError):
        module.prompt_texts(data, 1, offset=-1)
    with pytest.raises(ValueError):
        module.prompt_texts(data, 0)


def test_empty_audit_is_explicit_and_bootstrap_keeps_the_callers_random_state():
    empty = audit_summary(torch.zeros(3, len(METRICS) + 1))
    assert empty["positions"] == 0 and all(value is None for value in empty["mean"].values())
    assert all(row["original_raw_tv"] is None for row in empty["by_depth"])
    with pytest.raises(ValueError):
        paired_audit_intervals([empty])
    values = torch.ones(3, len(METRICS) + 1)
    record = audit_summary(values)
    state = torch.get_rng_state()
    paired_audit_intervals([record], repeats=10)
    assert torch.equal(state, torch.get_rng_state())
