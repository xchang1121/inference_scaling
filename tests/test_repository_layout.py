from pathlib import Path


REMOVED_COMPATIBILITY_MODULES = (
    "acceleration.py",
    "algorithms",
    "backends",
    "compute.py",
    "config.py",
    "evaluation",
    "metrics.py",
    "replay.py",
    "rng.py",
    "rollout_broker.py",
    "types.py",
    "vllm_suffix_proposer.py",
)


def test_model_families_and_shared_code_have_distinct_namespaces():
    package = Path("src/inference_scaling")
    assert (package / "arllm").is_dir()
    assert (package / "dllm").is_dir()
    assert (package / "shared").is_dir()
    for name in REMOVED_COMPATIBILITY_MODULES:
        path = package / name
        if path.suffix == ".py":
            assert not path.exists()
        else:
            assert not any(path.glob("*.py"))


def test_experiment_root_contains_only_the_paired_entrypoint():
    assert sorted(path.name for path in Path("experiments").glob("*.py")) == [
        "__init__.py",
        "run_reproduction.py",
    ]


def test_ar_experiment_paths_resolve_from_the_repository_root():
    from experiments.arllm.gsm8k_reproduction import REPOSITORY_ROOT
    from experiments.arllm.plot_gsm8k_passk import ROOT as PLOT_ROOT

    expected = Path.cwd().resolve()
    assert REPOSITORY_ROOT == expected
    assert PLOT_ROOT == expected


def test_ar_adapter_override_updates_every_identity_field(tmp_path):
    from experiments.arllm.runtime import set_rl_adapter_override

    config = {"models": {"rl": "old", "rl_source": "old", "rl_revision": "old"}}
    adapter = tmp_path / "adapter"
    set_rl_adapter_override(config, adapter)

    assert config["models"] == {
        "rl": str(adapter),
        "rl_source": "local GRPO adapter from the current reproduction suite",
        "rl_revision": "suite-output",
        "rl_kind": "peft_adapter",
    }
