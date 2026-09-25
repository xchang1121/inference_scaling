from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_scaling.app.cli import parse
from inference_scaling.app.run import Choices, run
from inference_scaling.app.settings import SCHEMA, SettingsError, check, load_settings
from inference_scaling.arllm.backends.tabular import TabularAutoregressiveBackend


def test_cli_defaults_and_reward_rules() -> None:
    choices, output = parse([])
    assert choices == Choices("is", "ar", "vote", "gsm8k") and output == Path("results")
    assert parse(["--algorithm", "sample"])[0].reward is None
    assert parse(["--algorithm", "best_of_n", "--reward", "verifier"])[0].label == "best_of_n-verifier"
    with pytest.raises(SystemExit):
        parse(["--algorithm", "greedy", "--reward", "vote"])


def test_settings_reject_missing_unknown_and_mistyped_keys() -> None:
    settings = load_settings()
    broken = copy.deepcopy(settings)
    broken["ar"]["sampling"]["extra"] = 1
    with pytest.raises(SettingsError, match="unknown keys \\['extra'\\]"):
        check(broken, SCHEMA, "settings")
    broken = copy.deepcopy(settings)
    del broken["rewards"]["vote"]["pool_size"]
    with pytest.raises(SettingsError, match="missing keys \\['pool_size'\\]"):
        check(broken, SCHEMA, "settings")
    broken = copy.deepcopy(settings)
    broken["run"]["draws"] = True
    with pytest.raises(SettingsError, match="must be an integer"):
        check(broken, SCHEMA, "settings")
    broken = copy.deepcopy(settings)
    broken["ar"]["algorithms"]["is"]["planning"] = "sometimes"
    with pytest.raises(SettingsError, match="must be one of"):
        check(broken, SCHEMA, "settings")


class ThinkingTokenizer:
    eos_token_id = 2

    def get_vocab(self):
        return {"<think>": 3, "</think>": 1, "7": 0, "<eos>": 2}


@dataclass(frozen=True)
class Counters:
    generation_forward_token_slots: int
    score_forward_token_slots: int
    estimated_dense_forward_flops: int


class ThinkingBackend(TabularAutoregressiveBackend):
    """Every prompt opens a thought: the model thinks "7", closes it, answers "7" and stops."""

    tokenizer = ThinkingTokenizer()
    parameter_count = 10

    def __init__(self) -> None:
        super().__init__({(3,): (1, 0, 0, 0), (3, 0): (0, 1, 0, 0), (3, 0, 1): (1, 0, 0, 0),
                          (3, 0, 1, 0): (0, 0, 1, 0)}, fallback=(0, 0, 1, 0))
        self.generated = 0

    def encode(self, text, *, add_special_tokens=False):
        return (self.tokenizer.get_vocab().get(text, 3),)

    def decode(self, tokens, *, skip_special_tokens=True):
        return "#### " + "".join("7" for token in tokens if token == 0)

    def sample_batch(self, requests):
        samples = super().sample_batch(requests)
        self.generated += sum(len(sample.token_ids) for sample in samples)
        return samples

    def snapshot(self):
        return Counters(self.generated, 0, 2 * self.parameter_count * self.generated)

    def direct_generate(self, prefix, *, max_new_tokens, num_beams):
        return (0, 1, 0, 2)[:max_new_tokens]

    def score_statistics_batch(self, requests, *, confidence_top_k=None):
        return [SimpleNamespace(token_topk_confidences=tuple(1.0 for _ in tokens))
                for request in requests for tokens in request.continuations]


@pytest.fixture
def ar_settings(base_settings, tmp_path, monkeypatch):
    settings = base_settings
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"weights")
    settings["ar"]["model"].update(path=str(model), revision=None, weight_sha256=None)
    settings["ar"]["output"]["generation_chunk_size"] = 1
    settings["rewards"]["vote"]["pool_size"] = 3
    joint = settings["ar"]["algorithms"]["is"]["joint"]
    joint.update(block_sizes=[2, 4], candidate_counts=[2, 4], rollout_counts=[1, 2])
    settings["ar"]["algorithms"]["is"]["chunk_adaptive"].update(initial_block_size=2, initial_candidate_count=2,
                                                                  initial_rollout_count=1)
    monkeypatch.setattr("inference_scaling.app.ar.load_backend", lambda *args, **kwargs: ThinkingBackend())
    return settings


AR_RUNS = [
    ("sample", None, {}), ("greedy", None, {}), ("beam", None, {}),
    ("best_of_n", "vote", {}), ("best_of_n", "verifier", {}), ("best_of_n", "logprob", {}),
    ("best_of_n", "consilience", {}),
    ("mh_power", None, {}), ("mh_power", None, {"sampling_scope": "thinking"}),
    ("mh", "vote", {}), ("mh", "verifier", {"proposal": "frozen_history"}),
    ("mh", "logprob", {"sampling_scope": "thinking"}), ("mh", "consilience", {}),
    ("is", "vote", {}), ("is", "verifier", {"planning": "fixed"}), ("is", "logprob", {"planning": "chunk_adaptive"}),
    ("is", "consilience", {"planning": "fixed", "sampling_scope": "thinking"}),
]


@pytest.mark.parametrize(("algorithm", "reward", "options"), AR_RUNS)
def test_every_ar_algorithm_writes_graded_records_and_resumes(ar_settings, tmp_path, algorithm, reward, options):
    ar = ar_settings["ar"]
    if "sampling_scope" in options:
        ar["output"]["sampling_scope"] = options["sampling_scope"]
    if "proposal" in options:
        ar["algorithms"]["mh"]["proposal"] = options["proposal"]
    if "planning" in options:
        ar["algorithms"]["is"]["planning"] = options["planning"]
    choices = Choices(algorithm, "ar", reward, "gsm8k")

    summary = run(choices, ar_settings, tmp_path / "results")
    directory = Path(summary["directory"])
    assert directory.parent == tmp_path / "results" / "gsm8k" / "ar" / choices.label
    records = [json.loads(line) for line in (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(record["problem_id"], record["draw"]) for record in records] == [("0", 0), ("1", 0)]
    for record in records:
        # The graded text is the answer after the thought, not the whole output.
        assert record["output"]["content"] == "#### 7" and record["correct"] and record["answer"] == "7"
        assert record["cost"]["forward_token_slots"] > 0
        assert (record["reward"] is None) == (reward is None or (algorithm, reward) == ("best_of_n", "vote"))
    assert summary["accuracy"] == 1.0 and summary["records"] == 2
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["choices"] == {"algorithm": algorithm, "model": "ar", "reward": reward, "dataset": "gsm8k"}
    assert manifest["effective_settings"]["algorithm"] == ar["algorithms"][algorithm]
    assert run(choices, ar_settings, tmp_path / "results")["records"] == 2
    assert len((directory / "records.jsonl").read_text(encoding="utf-8").splitlines()) == 2


def test_extra_draws_resume_the_same_run_and_report_pass_at_k(ar_settings, tmp_path):
    choices = Choices("sample", "ar", None, "gsm8k")
    first = run(choices, ar_settings, tmp_path / "results")
    ar_settings["run"]["draws"] = 2
    second = run(choices, ar_settings, tmp_path / "results")
    assert second["directory"] == first["directory"]
    assert second["records"] == 4 and second["pass_at_k"] == {"1": 1.0, "2": 1.0}


def test_concurrent_problems_share_a_batching_backend_without_per_problem_cost(ar_settings, tmp_path):
    ar_settings["ar"]["engine"]["continuous_batching"]["workers"] = 2
    summary = run(Choices("best_of_n", "ar", "vote", "gsm8k"), ar_settings, tmp_path / "results")
    records = (Path(summary["directory"]) / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["problem_id"] for line in records) == ["0", "1"]
    assert all(json.loads(line)["cost"] is None for line in records)


def test_budgeted_is_plans_the_thinking_segment(ar_settings, tmp_path):
    ar_settings["ar"]["output"]["sampling_scope"] = "thinking"
    summary = run(Choices("is", "ar", "logprob", "gsm8k"), ar_settings, tmp_path / "results")
    record = json.loads((Path(summary["directory"]) / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["output"]["sampling_scope"] == "thinking" and record["correct"]
    # The kept thought ends at its closing boundary, not at EOS or the length limit.
    assert record["trace"]["stopping_reason"] == "stop"


def test_text_rewards_fall_back_from_the_thinking_scope(ar_settings, tmp_path):
    ar_settings["ar"]["output"]["sampling_scope"] = "thinking"
    ar_settings["ar"]["algorithms"]["is"]["planning"] = "fixed"
    summary = run(Choices("is", "ar", "verifier", "gsm8k"), ar_settings, tmp_path / "results")
    record = json.loads((Path(summary["directory"]) / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["output"]["sampling_scope"] == "full"
    assert record["fallbacks"] == ["reward_uses_full_sequence"]
    assert summary["failures"]["fallbacks"] == {"reward_uses_full_sequence": 2}
