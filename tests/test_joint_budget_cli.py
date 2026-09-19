from __future__ import annotations

import json

import pytest

from experiments.arllm import joint_budget_is as cli
from inference_scaling.arllm.backends import TransformersBackend
from test_transformers_backend import ConstantLogitModel, TinyTokenizer


class TextTokenizer(TinyTokenizer):
    chat_template = None
    model_max_length = 32

    def encode(self, _text, **_kwargs):
        return [0]

    def decode(self, tokens, **_kwargs):
        return " ".join(map(str, tokens))


def arguments(*extra):
    return cli.build_parser().parse_args(
        [
            "--model",
            "fixture/model",
            "--prompt",
            "a test prompt",
            "--max-new-tokens",
            "4",
            "--budget-forward-tokens",
            "100",
            "--block-sizes",
            "1",
            "2",
            "--candidate-counts",
            "2",
            "4",
            "--rollout-counts",
            "1",
            "2",
            *extra,
        ]
    )


@pytest.mark.parametrize("backend_name", ["transformers", "vllm", "vllm-sync"])
@pytest.mark.parametrize("source", ["sequence_log_probability", "consilience"])
def test_cli_loader_routing_rewards_and_cpu_execution(
    monkeypatch, backend_name, source
):
    # The common loader is mocked; real Transformers sampling/scoring runs on CPU.
    # This does not claim a live vLLM engine test.
    backend = TransformersBackend(
        ConstantLogitModel([0.55, 0.3, 0.15]), TextTokenizer(), device="cpu"
    )
    closed = []

    def load(path, config, *, role):
        assert path == "fixture/model" and role == "base"
        assert config["runtime"]["backend"] == backend_name
        return backend

    monkeypatch.setattr(cli, "load_backend_from_config", load)
    monkeypatch.setattr(cli, "close_backend", lambda value: closed.append(value))
    result = cli.run(arguments("--backend", backend_name, "--reward", source))
    json.dumps(result, allow_nan=False)
    assert result["reserved_forward_tokens"] <= 100
    assert result["actual_backend_cost"]["generated_tokens"] > 0
    assert result["actual_backend_cost"]["score_forward_token_slots"] > 0
    assert result["actual_backend_cost"]["estimated_dense_forward_flops"] > 0
    assert result["generation_budget"]["effective_max_new_tokens"] == 4
    assert result["steps"] and result["output_segments"]
    assert closed == [backend]


def test_cli_closes_backend_when_budget_is_insufficient(monkeypatch):
    backend = TransformersBackend(
        ConstantLogitModel([0.55, 0.3, 0.15]), TextTokenizer(), device="cpu"
    )
    closed = []
    monkeypatch.setattr(cli, "load_backend_from_config", lambda *_a, **_kw: backend)
    monkeypatch.setattr(cli, "close_backend", lambda value: closed.append(value))
    with pytest.raises(ValueError, match="at least"):
        cli.run(arguments("--budget-forward-tokens", "1"))
    assert backend.snapshot().generated_tokens == 0
    assert closed == [backend]


@pytest.mark.parametrize(
    "extra",
    [
        ("--pilot-fraction", "1"),
        ("--sampling-scope", "thinking"),
        ("--reward-temperature", "0"),
        ("--candidate-counts", "1"),
    ],
)
def test_cli_rejects_unsupported_or_invalid_settings_before_loading(monkeypatch, extra):
    monkeypatch.setattr(
        cli,
        "load_backend_from_config",
        lambda *_a, **_kw: pytest.fail("loaded weights"),
    )
    with pytest.raises(ValueError):
        cli.run(arguments(*extra))
