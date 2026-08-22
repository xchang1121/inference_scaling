from pathlib import Path

import pytest

from experiments.shared.attempt_registry import load_attempt_registry


REGISTRY = Path("results/arllm/qwen15b_optimization/attempt_registry.json")


def test_qwen15b_attempt_registry_is_valid_and_excludes_dllm_experiments() -> None:
    document = load_attempt_registry(REGISTRY)
    assert document["scope"]["dllm_experiments"] is False
    assert {attempt["status"] for attempt in document["attempts"]} >= {
        "planned",
        "accepted_existing",
        "conditional",
        "rejected",
    }


def test_attempt_registry_rejects_nonaccepted_active_arm(tmp_path: Path) -> None:
    target = tmp_path / "registry.json"
    target.write_text(
        """{
          "schema_version": 1,
          "scope": {
            "model_family": "arllm",
            "primary_model": "Qwen2.5-1.5B-Instruct",
            "dllm_experiments": false
          },
          "attempts": [{
            "id": "bad",
            "category": "infra",
            "status": "planned",
            "active_execution": true,
            "comparison": "baseline",
            "decision_basis": "none"
          }]
        }""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="active without an accepted decision"):
        load_attempt_registry(target)
