[CmdletBinding()]
param(
    [string]$ResultPath
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $ResultPath) {
    $ResultPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'results\stage11_wsl_platform.json'
}
$taskRecord = [ordered]@{
    started_at = (Get-Date).ToUniversalTime().ToString('o')
    feature = 'VirtualMachinePlatform'
    automatically_reboot = $false
    stage = 'starting'
    before = $null
    after = $null
    restart_required = $false
    succeeded = $false
    failure = $null
}
function Save-PlatformCheckpoint {
    [IO.File]::WriteAllText($ResultPath, ($taskRecord | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
}
try {
    $taskIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $taskPrincipal = [Security.Principal.WindowsPrincipal]::new($taskIdentity)
    if (-not $taskPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Administrator privileges required to enable WSL2 platform.'
    }
    $taskRecord.before = [string](Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform).State
    $taskRecord.stage = 'enabling'
    Save-PlatformCheckpoint
    $taskFeature = Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All -NoRestart
    $taskRecord.after = [string](Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform).State
    $taskRecord.restart_required = [bool]$taskFeature.RestartNeeded -or $taskRecord.after -match 'Pending'
    $taskRecord.stage = 'complete'
    $taskRecord.succeeded = $true
} catch {
    $taskRecord.failure = $_.Exception.Message
    $taskRecord.stage = 'failed'
} finally {
    $taskRecord.finished_at = (Get-Date).ToUniversalTime().ToString('o')
    Save-PlatformCheckpoint
}
if (-not $taskRecord.succeeded) { exit 1 }
