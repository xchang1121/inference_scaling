[CmdletBinding()]
param(
    [string]$PackagePath,
    [string]$ResultPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $ResultPath) {
    $ResultPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'results\stage11_wsl_install.json'
}
$taskRecord = [ordered]@{
    schema_version = 1
    started_at = (Get-Date).ToUniversalTime().ToString('o')
    stage = 'starting'
    package_path = $PackagePath
    reboot_automatically = $false
    commands = @()
    succeeded = $false
    restart_required = $false
    failure = $null
}
function Save-Checkpoint {
    $taskRecord.updated_at = (Get-Date).ToUniversalTime().ToString('o')
    $taskJson = $taskRecord | ConvertTo-Json -Depth 8
    [IO.File]::WriteAllText($ResultPath, $taskJson, [Text.UTF8Encoding]::new($false))
}
try {
    $taskIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $taskPrincipal = [Security.Principal.WindowsPrincipal]::new($taskIdentity)
    if (-not $taskPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Administrator privileges are required for the WSL MSI and Windows feature.'
    }
    $taskExpected = 'A3505A50F4CC585551D11D9DE824BA4375448D7A68F2E71D3FB315FA986FC754'
    $taskHash = (Get-FileHash -LiteralPath $PackagePath -Algorithm SHA256).Hash
    if ($taskHash -ne $taskExpected) { throw 'Official WSL 2.7.13 x64 MSI hash mismatch.' }
    $taskSignature = Get-AuthenticodeSignature -LiteralPath $PackagePath
    if ($taskSignature.Status -ne 'Valid' -or $taskSignature.SignerCertificate.Subject -notmatch 'Microsoft Corporation') {
        throw 'WSL MSI must have a valid Microsoft Authenticode signature.'
    }
    $taskRecord.package_sha256 = $taskHash
    $taskRecord.signer = $taskSignature.SignerCertificate.Subject
    $taskRecord.stage = 'installing_msi'
    Save-Checkpoint
    $taskLogPath = [IO.Path]::ChangeExtension($ResultPath, '.msi.log')
    $taskMsiArguments = @('/i', ('"' + $PackagePath + '"'), '/qn', '/norestart', '/L*v', ('"' + $taskLogPath + '"'))
    $taskMsi = Start-Process -FilePath "$env:SystemRoot\System32\msiexec.exe" -ArgumentList $taskMsiArguments -WindowStyle Hidden -Wait -PassThru
    $taskRecord.commands += @{ command = 'msiexec /i official_wsl /qn /norestart'; exit_code = $taskMsi.ExitCode }
    if ($taskMsi.ExitCode -notin @(0,3010)) { throw "WSL MSI failed with exit code $($taskMsi.ExitCode)." }
    $taskRecord.restart_required = $taskMsi.ExitCode -eq 3010
    $taskRecord.stage = 'enabling_virtual_machine_platform'
    Save-Checkpoint
    $taskFeature = Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -All -NoRestart
    $taskFeatureState = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
    $taskRecord.virtual_machine_platform = @{ state = [string]$taskFeatureState.State; restart_needed = [bool]$taskFeature.RestartNeeded }
    $taskRecord.restart_required = $taskRecord.restart_required -or [bool]$taskFeature.RestartNeeded
    $taskRecord.stage = 'system_components_installed'
    $taskRecord.succeeded = $true
} catch {
    $taskRecord.failure = $_.Exception.Message
    $taskRecord.stage = 'failed'
} finally {
    Save-Checkpoint
}
if (-not $taskRecord.succeeded) { exit 1 }
