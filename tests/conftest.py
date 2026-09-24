from __future__ import annotations

import copy
import hashlib
import json

import pytest

from inference_scaling.app.settings import load_settings


@pytest.fixture
def base_settings(tmp_path):
    """The shipped settings with a local two-problem GSM8K file whose answers are 7."""

    settings = copy.deepcopy(load_settings())
    settings["run"]["hash_cache_dir"] = str(tmp_path / "hashes")
    data = tmp_path / "gsm8k.jsonl"
    data.write_text("".join(json.dumps({"question": f"q{index}", "answer": "#### 7"}) + "\n" for index in range(2)),
                    encoding="utf-8")
    gsm8k = settings["datasets"]["gsm8k"]
    gsm8k.update(path=str(data), download=False, max_new_tokens=6)
    gsm8k["source"].update(sha256=hashlib.sha256(data.read_bytes()).hexdigest(), rows=2)
    gsm8k["selection"]["count"] = None
    return settings
