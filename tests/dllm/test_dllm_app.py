from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_scaling.app.run import Choices, run

torch = pytest.importorskip("torch")


class TinyMaskedModel(torch.nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(bias, dtype=torch.float32))
        self.config = SimpleNamespace(_name_or_path="tiny", mask_token_id=3)

    def forward(self, token_ids):
        batch, length = token_ids.shape
        return SimpleNamespace(logits=self.bias.view(1, 1, -1).expand(batch, length, -1).clone())


class TinyTokenizer:
    mask_token_id = 3
    eos_token_id = 1

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        return [0]

    def decode(self, token_ids, skip_special_tokens=True):
        return "#### " + str(sum(token == 2 for token in token_ids))


def _tiny_backend():
    from inference_scaling.dllm.backends.llada import LLaDATransformersBackend

    backend = LLaDATransformersBackend(TinyMaskedModel((0.0, 0.5, 1.0, -2.0)), TinyTokenizer(), mask_token_id=3)
    proposal = LLaDATransformersBackend(TinyMaskedModel((1.0, 0.0, 0.5, -2.0)), TinyTokenizer(), mask_token_id=3)
    backend.with_prefix_layers = lambda layers: proposal
    return backend


@pytest.fixture
def dllm_settings(base_settings, tmp_path, monkeypatch):
    settings = base_settings
    settings["datasets"]["gsm8k"]["max_new_tokens"] = 5
    model = tmp_path / "llada"
    model.mkdir()
    (model / "w.safetensors").write_bytes(b"w")
    settings["dllm"]["model"].update(path=str(model), weight_files=["w.safetensors"], weight_bytes=[1],
                                     weight_sha256=[hashlib.sha256(b"w").hexdigest()], mask_token_id=3)
    for section in ("sampling", "exact_sampling"):
        settings["dllm"][section].update(block_length=2, steps_per_block=2)
    algorithms = settings["dllm"]["algorithms"]
    algorithms["beam"].update(decision_block_size=2, width=2, branching_factor=2)
    algorithms["best_of_n"]["samples"] = 3
    algorithms["mh"].update(decision_block_size=2, updates_per_stage=2)
    algorithms["reward_mh"].update(updates=3)
    algorithms["reward_mh"]["frozen_history"]["samples"] = 2
    algorithms["is"].update(candidate_count=2, rollout_count=2, decision_block_size=2)
    settings["rewards"]["vote"]["pool_size"] = 2
    monkeypatch.setattr("inference_scaling.app.dllm.load_llada_backend", lambda model, engine: _tiny_backend())
    return settings


DLLM_RUNS = [
    ("sample", None, {}), ("greedy", None, {}), ("beam", None, {}), ("mh", None, {}),
    ("best_of_n", "vote", {}), ("best_of_n", "verifier", {}),
    ("reward_mh", "verifier", {}), ("reward_mh", "vote", {"proposal": "frozen_history"}),
    ("is", "vote", {}), ("is", "verifier", {"rollout_model": "proposal"}),
]


@pytest.mark.parametrize(("algorithm", "reward", "options"), DLLM_RUNS)
def test_every_dllm_algorithm_writes_graded_records(dllm_settings, tmp_path, algorithm, reward, options):
    algorithms = dllm_settings["dllm"]["algorithms"]
    if "proposal" in options:
        algorithms["reward_mh"]["proposal"] = options["proposal"]
    if "rollout_model" in options:
        algorithms["is"]["rollout_model"] = options["rollout_model"]
    summary = run(Choices(algorithm, "dllm", reward, "gsm8k"), dllm_settings, tmp_path / "results")
    records = [json.loads(line) for line in
               (Path(summary["directory"]) / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    for record in records:
        assert record["output"]["tokens"] == 4 and record["parseable"]
        assert record["cost"]["forward_token_slots"] > 0
    if options.get("rollout_model") == "proposal":
        assert records[0]["cost"]["phases"]["search"]["proposal"]["model_token_slots"] > 0
        assert records[0]["trace"]["corrected_rollouts"] > 0


@pytest.mark.parametrize("reward", ["logprob", "consilience"])
def test_diffusion_rejects_autoregressive_probability_rewards(dllm_settings, tmp_path, reward):
    with pytest.raises(ValueError, match="autoregressive token probabilities"):
        run(Choices("is", "dllm", reward, "gsm8k"), dllm_settings, tmp_path / "results")
