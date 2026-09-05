[CmdletBinding()]
param([string]$Distribution = 'Ubuntu-22.04')

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$taskProject = Split-Path -Parent $PSScriptRoot
$taskInstallRecordPath = Join-Path $taskProject 'results\stage11_wsl_install.json'
if (-not (Test-Path -LiteralPath $taskInstallRecordPath)) {
    throw 'Missing signed WSL installation checkpoint; inspect installation first.'
}
$taskRecord = Get-Content -LiteralPath $taskInstallRecordPath -Raw | ConvertFrom-Json
if (-not $taskRecord.succeeded) { throw 'Previous WSL installation did not succeed.' }
$taskBoot = [DateTimeOffset](Get-CimInstance Win32_OperatingSystem).LastBootUpTime
$taskInstalled = [DateTimeOffset]::Parse($taskRecord.updated_at)
if ($taskRecord.restart_required -and $taskBoot -le $taskInstalled) {
    throw 'Windows must be restarted before Linux initialization. This script will not reboot the PC.'
}

$taskList = @(& "$env:SystemRoot\System32\wsl.exe" --list --quiet 2>&1 | ForEach-Object {
    ([string]$_).Replace([string][char]0, '').Trim()
})
if ($Distribution -notin $taskList) {
    & "$env:SystemRoot\System32\wsl.exe" --install --distribution $Distribution --no-launch --web-download
    if ($LASTEXITCODE -ne 0) { throw "WSL distro installation failed: $LASTEXITCODE" }
}
& "$env:SystemRoot\System32\wsl.exe" --distribution $Distribution --user root -- uname -m
if ($LASTEXITCODE -ne 0) { throw 'Linux did not launch; stop before installing runtime packages.' }
& (Join-Path $PSScriptRoot 'run_wsl_bootstrap.ps1') -Distribution $Distribution
if ($LASTEXITCODE -ne 0) { throw "Linux bootstrap failed: $LASTEXITCODE" }
