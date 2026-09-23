from dataclasses import dataclass
import json
from types import SimpleNamespace

import pytest

from experiments.arllm import joint_budget_is as cli
from test_joint_budget_is import RecordingBackend


def arguments(*extra):
    return cli.build_parser().parse_args([
        "--model", "fixture/model", "--prompt", "test", "--max-new-tokens", "24",
        "--budget-forward-tokens", "5000", "--reward-temperature", "1",
        "--block-sizes", "2", "4", "8", "--candidate-counts", "2", "4", "8",
        "--rollout-counts", "1", "2", "4", *extra,
    ])


ADAPTIVE = (
    "--planning-mode", "chunk_adaptive", "--initial-block-size", "4",
    "--initial-candidate-count", "4", "--initial-rollout-count", "2",
)


@dataclass
class Snapshot:
    requests: int


class Backend(RecordingBackend):
    tokenizer = SimpleNamespace(eos_token_id=None, model_max_length=1024)

    def encode(self, _text, **_kwargs):
        return (0,)

    def decode(self, tokens, **_kwargs):
        return " ".join(map(str, tokens))

    def snapshot(self):
        return Snapshot(len(self.requests))


class Reward:
    def __call__(self, _prompt, _tokens):
        return 0.0

    def describe(self):
        return {"source": "constant_fixture"}


def test_cli_forwards_mode_presets_and_reports_adjustments(monkeypatch):
    backend = Backend()
    closed = []
    monkeypatch.setattr(cli, "load_backend_from_config", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(cli, "close_backend", closed.append)
    monkeypatch.setattr(cli, "model_reward_from_config", lambda *_args, **_kwargs: Reward())
    monkeypatch.setattr(cli, "render_prompt", lambda *_args: "test")
    monkeypatch.setattr(cli.SamplingScope, "from_config", lambda *_args: SimpleNamespace(
        describe_output=lambda *_args: {},
    ))
    result = cli.run(arguments(*ADAPTIVE, "--pilot-fraction", "0"))
    assert result["config"]["planning_mode"] == "chunk_adaptive"
    first = result["steps"][0]
    assert tuple(first["plan"][name] for name in ("block_size", "candidate_count", "rollout_count")) == (4, 4, 2)
    assert first["adjustment"]["status"] == "initial"
    assert len(result["steps"]) > 1
    assert result["generation_budget"]["effective_max_new_tokens"] == 24
    assert result["reserved_forward_tokens"] <= 5000
    assert closed == [backend]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("extra", [
    ("--planning-mode", "chunk_adaptive"),
    (*ADAPTIVE, "--initial-block-size", "3"),
    (*ADAPTIVE, "--adjustment-min-improvement", "0"),
    ("--initial-candidate-count", "4"),
])
def test_cli_rejects_invalid_adaptive_settings_before_loading(monkeypatch, extra):
    monkeypatch.setattr(cli, "load_backend_from_config", lambda *_args, **_kwargs: pytest.fail("loaded weights"))
    with pytest.raises(ValueError):
        cli.run(arguments(*extra))


def test_cli_default_stays_full_horizon():
    args = arguments()
    assert args.planning_mode == "full_horizon"
    assert args.initial_block_size is None
