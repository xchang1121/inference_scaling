import ast
from pathlib import Path
import subprocess
import sys

from experiments.shared.components import FULL_COMPONENTS


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
# Modules now grouped by concern inside each package.
MOVED_MODULES = {
    "shared": (
        "budget.py", "consilience.py", "generation.py", "importance.py", "joint_budget.py", "mh.py",
        "model_loading.py", "output.py", "prompting.py", "smc.py", "stepwise.py",
        "structured_output.py", "verifier.py",
    ),
    "arllm": (
        "acceleration.py", "replay.py", "reward_factory.py", "rewards.py", "rollout_broker.py",
        "vllm_suffix_proposer.py",
    ),
    "dllm": ("dynamic_is.py", "preferences.py", "replay.py", "vrpo.py"),
}
SUBPACKAGES = {
    "shared": ("budget", "evaluation", "model", "rewards", "sampling"),
    "arllm": ("acceleration", "algorithms", "backends", "rewards"),
    "dllm": ("algorithms", "backends", "training"),
    "archive": ("arllm",),
}
# Lower layers never import the layers built on them.
FORBIDDEN_IMPORTS = {
    "inference_scaling.shared": (
        "inference_scaling.arllm", "inference_scaling.dllm", "inference_scaling.experimental",
        "inference_scaling.archive",
    ),
    # Archived methods are reached only by name from experiment assembly.
    "inference_scaling.arllm": ("inference_scaling.archive",),
    "inference_scaling.dllm": ("inference_scaling.archive",),
    "inference_scaling.experimental": ("inference_scaling.archive",),
    "inference_scaling.arllm.acceleration": (
        "inference_scaling.arllm.algorithms", "inference_scaling.arllm.backends",
        "inference_scaling.arllm.rewards",
    ),
    "inference_scaling.arllm.backends": (
        "inference_scaling.arllm.algorithms", "inference_scaling.arllm.rewards",
    ),
    "inference_scaling.dllm.backends": (
        "inference_scaling.dllm.algorithms", "inference_scaling.dllm.training",
    ),
}


def _module_imports(root: str):
    for path in Path(root).rglob("*.py"):
        parts = path.with_suffix("").parts
        module = ".".join(parts[1:] if parts[0] == "src" else parts).removesuffix(".__init__")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                yield module, node.module, [alias.name for alias in node.names]


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
    for parent, names in MOVED_MODULES.items():
        for name in names:
            assert not (package / parent / name).exists()
    for parent, names in SUBPACKAGES.items():
        for name in names:
            assert (package / parent / name / "__init__.py").is_file()


def test_lower_layers_never_import_the_layers_built_on_them():
    violations = [
        (module, target)
        for module, target, _ in _module_imports("src/inference_scaling")
        for layer, forbidden in FORBIDDEN_IMPORTS.items()
        if (module == layer or module.startswith(layer + ".")) and target.startswith(forbidden)
    ]
    assert violations == []


def test_private_names_are_not_imported_across_modules():
    # Acceleration modules extend the MH kernel beside them in the same package.
    extensions = {
        ("inference_scaling.arllm.algorithms.mh_acceleration", "inference_scaling.arllm.algorithms.mh"),
        ("inference_scaling.dllm.algorithms.mh_acceleration", "inference_scaling.dllm.algorithms.mh"),
    }
    violations = [
        (module, target, name)
        for root in ("src", "experiments")
        for module, target, names in _module_imports(root)
        for name in names
        if name.startswith("_") and not name.startswith("__") and (module, target) not in extensions
    ]
    assert violations == []


def test_production_defaults_exclude_research_components():
    assert FULL_COMPONENTS == (
        "quality",
        "matched_target",
        "replay",
        "async",
        "passk",
        "distribution",
    )


def test_production_algorithm_import_does_not_load_experimental_modules():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import inference_scaling.arllm.algorithms; "
                "assert not any(name.startswith('inference_scaling.experimental') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_experiment_root_contains_only_the_paired_entrypoint():
    assert sorted(path.name for path in Path("experiments").glob("*.py")) == [
        "__init__.py",
        "run_reproduction.py",
    ]


def test_family_experiment_directories_hold_only_entry_points():
    # Shared assembly code lives in experiments/<family>/assembly.
    for family in ("arllm", "dllm"):
        assert (Path("experiments") / family / "assembly" / "__init__.py").is_file()
        library = [
            path.name
            for path in (Path("experiments") / family).glob("*.py")
            if path.name != "__init__.py"
            and 'if __name__ == "__main__":' not in path.read_text(encoding="utf-8")
        ]
        assert library == []


def test_ar_experiment_paths_resolve_from_the_repository_root():
    from experiments.arllm.assembly.runtime import REPOSITORY_ROOT as RUNTIME_ROOT
    from experiments.arllm.gsm8k_reproduction import REPOSITORY_ROOT
    from experiments.arllm.run_arllm_suite import REPOSITORY_ROOT as SUITE_ROOT
    from experiments.dllm.assembly.runtime import REPOSITORY_ROOT as DLLM_RUNTIME_ROOT

    expected = Path.cwd().resolve()
    assert REPOSITORY_ROOT == expected
    assert SUITE_ROOT == expected
    assert RUNTIME_ROOT == expected
    assert DLLM_RUNTIME_ROOT == expected


def test_ar_adapter_override_updates_every_identity_field(tmp_path):
    from experiments.arllm.assembly.runtime import set_rl_adapter_override

    config = {"models": {"rl": "old", "rl_source": "old", "rl_revision": "old"}}
    adapter = tmp_path / "adapter"
    set_rl_adapter_override(config, adapter)

    assert config["models"] == {
        "rl": str(adapter),
        "rl_source": "local GRPO adapter from the current reproduction suite",
        "rl_revision": "suite-output",
        "rl_kind": "peft_adapter",
    }
