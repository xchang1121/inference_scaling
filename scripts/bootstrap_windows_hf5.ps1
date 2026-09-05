[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$TransformersVersion = "5.16.1"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoRoot = (Resolve-Path (Join-Path $projectRoot "..")).Path
if (-not $Python) {
    $Python = Join-Path $repoRoot ".venv\Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$overrideRoot = Join-Path $projectRoot ".hf-overrides"
New-Item -ItemType Directory -Force -Path $overrideRoot | Out-Null
& $Python -m pip install --upgrade --target $overrideRoot "transformers==$TransformersVersion"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install the isolated Transformers runtime."
}

$previousPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = "$overrideRoot;$projectRoot\src"
    & $Python -c "import transformers, torch; from transformers.configuration_utils import PreTrainedConfig; print({'transformers': transformers.__version__, 'torch': torch.__version__, 'compat_symbol': PreTrainedConfig.__name__})"
    if ($LASTEXITCODE -ne 0) {
        throw "The isolated Transformers runtime failed its import check."
    }
}
finally {
    $env:PYTHONPATH = $previousPythonPath
}

Write-Host "Isolated HF override ready at $overrideRoot"
Write-Host "Before a benchmark, prepend this directory to PYTHONPATH."
