import ast
import subprocess
import sys
from pathlib import Path

SUBPACKAGES = {
    "": ("app", "arllm", "datasets", "dllm", "shared"),
    "shared": ("budget", "model", "rewards", "sampling"),
    "arllm": ("algorithms", "backends", "rewards"),
    "dllm": ("algorithms", "backends", "training"),
}
# Lower layers never import the layers built on them; the two model families
# (and their app modules, which run in different environments) never import each other.
FORBIDDEN_IMPORTS = {
    "inference_scaling.datasets": ("inference_scaling.shared", "inference_scaling.arllm", "inference_scaling.dllm",
                                   "inference_scaling.app"),
    "inference_scaling.shared": ("inference_scaling.arllm", "inference_scaling.dllm", "inference_scaling.app"),
    "inference_scaling.arllm": ("inference_scaling.dllm", "inference_scaling.app"),
    "inference_scaling.dllm": ("inference_scaling.arllm", "inference_scaling.app"),
    "inference_scaling.arllm.backends": ("inference_scaling.arllm.algorithms", "inference_scaling.arllm.rewards"),
    "inference_scaling.dllm.backends": ("inference_scaling.dllm.algorithms", "inference_scaling.dllm.training"),
    "inference_scaling.app.dllm": ("inference_scaling.app.ar", "inference_scaling.arllm"),
    "inference_scaling.app.ar": ("inference_scaling.app.dllm", "inference_scaling.dllm"),
}


def _module_imports(root: str):
    for path in Path(root).rglob("*.py"):
        parts = path.with_suffix("").parts
        module = ".".join(parts[1:] if parts[0] == "src" else parts).removesuffix(".__init__")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                yield module, node.module, [alias.name for alias in node.names]


def test_packages_group_code_by_concern():
    package = Path("src/inference_scaling")
    for parent, names in SUBPACKAGES.items():
        for name in names:
            assert (package / parent / name / "__init__.py").is_file()
    assert not Path("experiments").exists() and not Path("configs").exists()
    assert sorted(path.name for path in Path("settings").iterdir()) == ["inference.json", "training.json"]


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
        for root in ("src", "training")
        for module, target, names in _module_imports(root)
        for name in names
        if name.startswith("_") and not name.startswith("__") and (module, target) not in extensions
    ]
    assert violations == []


def test_the_cli_and_the_settings_load_without_model_libraries():
    completed = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import inference_scaling.app.cli; import training.settings; "
            "assert not {'torch', 'transformers', 'vllm'} & set(sys.modules)"
        )],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
