[CmdletBinding()]
param(
    [string]$Distribution = "Ubuntu-22.04",
    [string]$LinuxUser = "singm",
    [string]$UnoSource = "C:\Users\singm\Desktop\hw\akg_related\.tmp_uno_upstream",
    [string]$BaseModel = "C:\Users\singm\Desktop\hw\akg_related\.tmp_k2_horizon_09b",
    [string]$Adapter = "C:\Users\singm\Desktop\hw\akg_related\.tmp_k2_horizon_09b_uno"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function ConvertTo-WslMountPath {
    param([Parameter(Mandatory)][string]$WindowsPath)

    $resolved = (Resolve-Path -LiteralPath $WindowsPath).Path
    if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Only absolute drive-letter paths are supported: $resolved"
    }
    $drive = $Matches[1].ToLowerInvariant()
    $tail = $Matches[2].Replace("\", "/")
    return "/mnt/$drive/$tail"
}

function Invoke-WslChecked {
    param([Parameter(Mandatory)][string[]]$Arguments)

    & "$env:SystemRoot\System32\wsl.exe" @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "wsl.exe failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$systemScript = ConvertTo-WslMountPath (
    Join-Path $PSScriptRoot "bootstrap_wsl_system.sh"
)
$runtimeScript = ConvertTo-WslMountPath (
    Join-Path $PSScriptRoot "bootstrap_uno_runtime.sh"
)
$wslConfig = ConvertTo-WslMountPath (
    Join-Path $projectRoot "config\wsl.conf"
)
$projectMount = ConvertTo-WslMountPath $projectRoot
$unoMount = ConvertTo-WslMountPath $UnoSource
$baseMount = ConvertTo-WslMountPath $BaseModel
$adapterMount = ConvertTo-WslMountPath $Adapter
$resultMount = "$projectMount/results/stage9_wsl_runtime.json"

Invoke-WslChecked -Arguments @("--set-default-version", "2")
$taskWslRows = @(& "$env:SystemRoot\System32\wsl.exe" --list --verbose 2>&1 | ForEach-Object {
    ([string]$_).Replace([string][char]0, '')
})
if ($LASTEXITCODE -ne 0) { throw 'Could not read the installed WSL distribution versions.' }
$taskDistroPattern = '^\s*\*?\s*' + [regex]::Escape($Distribution) + '\s+.+?\s+([12])\s*$'
$taskInstalledVersion = $null
foreach ($taskWslRow in $taskWslRows) {
    if ($taskWslRow -match $taskDistroPattern) { $taskInstalledVersion = [int]$Matches[1] }
}
if ($null -eq $taskInstalledVersion) { throw "Distribution version not found: $Distribution" }
if ($taskInstalledVersion -ne 2) {
    Invoke-WslChecked -Arguments @("--set-version", $Distribution, "2")
}
Invoke-WslChecked -Arguments @(
    "--distribution", $Distribution,
    "--user", "root",
    "--", "bash", $systemScript, $LinuxUser, $wslConfig
)

# Reload /etc/wsl.conf before running the unprivileged phase.
Invoke-WslChecked -Arguments @("--terminate", $Distribution)
Invoke-WslChecked -Arguments @(
    "--distribution", $Distribution,
    "--user", $LinuxUser,
    "--", "bash", $runtimeScript,
    $unoMount, $projectMount, $baseMount, $adapterMount, $resultMount
)

Write-Host "WSL Uno runtime bootstrap and smoke test completed."
