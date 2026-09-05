"""The fixed evaluation suite is input data, never an instruction source."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("benchmark_native_uno", ROOT / "scripts/benchmark_native_uno.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fixed_new_suite_is_separate_and_filterable():
    path = ROOT / "config/evaluation_prompts.json"
    suite = module.select_workloads(path)
    assert len(suite) == 12
    assert not {name for name, _ in suite} & {name for name, _ in module.WORKLOADS}
    assert not {prompt for _, prompt in suite} & {prompt for _, prompt in module.WORKLOADS}
    assert len(module.select_workloads(path, "en_cache,math_modular")) == 2
    with pytest.raises(ValueError):
        module.select_workloads(path, "en_cache,en_cache")


@pytest.mark.parametrize("data", [[], [["same", "one"], ["same", "two"]], [["id", ""]], ["text"]])
def test_malformed_prompt_suite_rejected(tmp_path, data):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        module.select_workloads(path)


@pytest.mark.parametrize("count", [3, 4, 5])
def test_adjacent_repetitions_counterbalance_each_method_position(count):
    methods = list(range(count))
    original = list(methods)
    for prompt in range(12):
        for rep in range(0, 6, 2):
            first = module.paired_method_order(methods, prompt, rep)
            second = module.paired_method_order(methods, prompt, rep + 1)
            assert second == list(reversed(first))
            for method in methods:
                assert first.index(method) + second.index(method) == count - 1
    assert methods == original
