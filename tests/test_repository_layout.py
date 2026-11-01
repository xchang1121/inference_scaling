import subprocess
import sys


def test_the_cli_and_the_settings_load_without_model_libraries():
    completed = subprocess.run(
        [sys.executable, "-c", (
            "import sys; import inference_scaling.app.cli; "
            "assert not {'torch', 'transformers', 'vllm'} & set(sys.modules)"
        )],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
