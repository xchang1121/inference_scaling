from __future__ import annotations

import copy
import hashlib
import json

import pytest

from inference_scaling.app.settings import load_settings


@pytest.fixture
def tiny_dllm_settings():
    """CPU-only diffusion fixture; no production model or device preset."""
    sampling = {"block_length": 2, "steps_per_block": 2, "temperature": 1.0,
                "top_k": 0, "top_p": 1.0, "remasking": "low_confidence"}
    return {
        "model": {"path": "tiny", "revision": None, "weight_sha256": None,
                  "mask_token_id": 3, "trust_remote_code": True, "adapter": None},
        "engine": {"device": "cpu", "dtype": "float32", "attn_implementation": "sdpa",
                   "max_batch_size": 3, "block_logits_only": False},
        "prompt": {"system": None}, "max_new_tokens": 5,
        "sampling": sampling, "exact_sampling": {**sampling, "remasking": "random"},
        "algorithms": {
            "sample": {}, "greedy": {}, "beam": {"decision_block_size": 2, "width": 2, "branching_factor": 2},
            "best_of_n": {"samples": 3},
            "mh": {"updates": 3, "proposal": "base", "frozen_history": {"samples": 2, "mixture": 0.25}},
            "mh_power": {"alpha": 4.0, "decision_block_size": 2, "updates_per_stage": 2},
            "is": {"candidate_count": 2, "rollout_count": 2, "decision_block_size": 2,
                   "candidate_canvas": "block", "kept_sequence": False},
        },
    }


@pytest.fixture
def base_settings(tmp_path):
    """The shipped settings with a local two-problem GSM8K file whose answers are 7."""

    settings = copy.deepcopy(load_settings())
    settings["run"]["hash_cache_dir"] = str(tmp_path / "hashes")
    data = tmp_path / "gsm8k.jsonl"
    data.write_text("".join(json.dumps({"question": f"q{index}", "answer": "#### 7"}) + "\n" for index in range(2)),
                    encoding="utf-8")
    gsm8k = settings["datasets"]["gsm8k"]
    gsm8k.update(path=str(data), download=False)
    settings["ar"]["max_new_tokens"] = 6
    gsm8k["source"].update(sha256=hashlib.sha256(data.read_bytes()).hexdigest(), rows=2)
    gsm8k["selection"]["count"] = None
    return settings
