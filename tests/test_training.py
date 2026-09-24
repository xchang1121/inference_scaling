from __future__ import annotations

import copy

import pytest

from inference_scaling.app.settings import SettingsError, check
from inference_scaling.datasets.gsm8k import GSM8K
from training.grpo import VerifierReward, _combine, _latest_checkpoint, _text
from training.settings import SCHEMA, load_settings


def test_training_settings_validate_and_list_known_stages() -> None:
    settings = load_settings()
    assert settings["stages"] == ["download", "grpo", "vrpo_preferences", "vrpo"]
    broken = copy.deepcopy(settings)
    broken["stages"] = ["download", "evaluate"]
    with pytest.raises(SettingsError, match="must be one of"):
        check(broken, SCHEMA, "training")
    broken = copy.deepcopy(settings)
    del broken["vrpo"]["training"]["beta"]
    with pytest.raises(SettingsError, match="missing keys \\['beta'\\]"):
        check(broken, SCHEMA, "training")


def test_grpo_reward_grades_each_completion_against_its_row_reference(base_settings) -> None:
    dataset = GSM8K(base_settings["datasets"]["gsm8k"])
    reward = VerifierReward(load_settings()["grpo"]["verifier"], dataset)
    prompts = [[{"role": "user", "content": "q0"}]] * 3
    completions = [[{"role": "assistant", "content": text}] for text in ("#### 7", "#### 8", "no answer")]
    values = reward(prompts, completions, completion_ids=[[1, 2], [3], [4, 5, 6]], reference=["7", "7", "7"],
                    problem_id=["0", "0", "0"], trainer_state=None)
    assert values == [1.0, 0.0, 0.0]
    snapshot = reward.snapshot(num_generations=3)
    assert snapshot["generated_completions"] == 3 and snapshot["generated_prompt_groups"] == 1
    assert snapshot["generated_completion_tokens"] == 6 and snapshot["observed_maximum_reward"] == 1.0


def test_resumed_rollout_totals_accumulate() -> None:
    first = {"reward_calls": 2, "generated_completions": 8, "generated_prompt_groups": 2,
             "generated_completion_tokens": 40, "reward_sum": 3.0, "observed_minimum_reward": 0.0,
             "observed_maximum_reward": 1.0}
    second = {**first, "reward_sum": 5.0, "observed_minimum_reward": None, "observed_maximum_reward": 0.5}
    total = _combine(_combine({}, first), second)
    assert total["generated_completions"] == 16 and total["reward_sum"] == 8.0
    assert total["observed_mean_reward"] == 0.5 and total["observed_maximum_reward"] == 1.0


def test_latest_checkpoint_and_conversational_text(tmp_path) -> None:
    assert _latest_checkpoint(tmp_path) is None
    for step in (5, 25, 10):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "checkpoint-final").mkdir()
    assert _latest_checkpoint(tmp_path) == tmp_path / "checkpoint-25"
    assert _text([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]) == "a\nb"
