[CmdletBinding()]
param(
    [ValidateNotNullOrEmpty()]
    [string]$Distribution = "Ubuntu-22.04",

    [ValidateNotNullOrEmpty()]
    [string]$ResultPath = (
        Join-Path (Split-Path -Parent $PSScriptRoot) `
            "results\stage9_wsl_install.json"
    )
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator
    )
}

function Get-FeatureSnapshot {
    $names = @(
        "Microsoft-Windows-Subsystem-Linux",
        "VirtualMachinePlatform"
    )
    $snapshots = foreach ($name in $names) {
        try {
            $feature = Get-WindowsOptionalFeature -Online -FeatureName $name
            [ordered]@{
                feature_name = $feature.FeatureName
                state = [string]$feature.State
                restart_required = [string]$feature.RestartRequired
            }
        }
        catch {
            [ordered]@{
                feature_name = $name
                error = $_.Exception.Message
            }
        }
    }
    return @($snapshots)
}

function Invoke-WslCommand {
    param(
        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    $lines = @(& "$env:SystemRoot\System32\wsl.exe" @Arguments 2>&1 |
        ForEach-Object { [string]$_ })
    return [ordered]@{
        arguments = @($Arguments)
        exit_code = [int]$LASTEXITCODE
        output = @($lines)
    }
}

$resultDirectory = Split-Path -Parent $ResultPath
if (-not (Test-Path -LiteralPath $resultDirectory)) {
    New-Item -ItemType Directory -Path $resultDirectory -Force | Out-Null
}

$record = [ordered]@{
    schema_version = 1
    started_at = (Get-Date).ToUniversalTime().ToString("o")
    host = $env:COMPUTERNAME
    user = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    administrator = Test-Administrator
    distribution = $Distribution
    command_policy = [ordered]@{
        install_linux_gpu_driver = $false
        install_cuda_meta_package = $false
        reboot_automatically = $false
    }
    before = [ordered]@{}
    commands = @()
    after = [ordered]@{}
    restart_required = $false
    succeeded = $false
    failure = $null
}

try {
    if (-not $record.administrator) {
        throw "This installer must run from an elevated PowerShell process."
    }

    $pendingRebootPaths = @(
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending",
        "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\WindowsUpdate\Auto Update\RebootRequired"
    )
    $pendingBefore = @($pendingRebootPaths | Where-Object {
        Test-Path -LiteralPath $_
    })
    if ($pendingBefore.Count -gt 0) {
        throw "Windows already has a pending reboot: $($pendingBefore -join ', ')"
    }

    $processor = Get-CimInstance Win32_Processor | Select-Object -First 1
    if (-not $processor.VirtualizationFirmwareEnabled) {
        throw "CPU virtualization is disabled in firmware."
    }
    if (-not $processor.SecondLevelAddressTranslationExtensions) {
        throw "CPU SLAT support is unavailable."
    }

    $systemDriveName = $env:SystemDrive.TrimEnd(":")
    $systemDrive = Get-PSDrive -Name $systemDriveName
    if ($systemDrive.Free -lt 25GB) {
        throw "Less than 25 GiB is free on the Windows system drive."
    }

    $record.before = [ordered]@{
        windows = Get-CimInstance Win32_OperatingSystem |
            Select-Object Caption, Version, BuildNumber, OSArchitecture
        free_bytes_system_drive = [int64]$systemDrive.Free
        virtualization_firmware_enabled = [bool]$processor.VirtualizationFirmwareEnabled
        slat_available = [bool]$processor.SecondLevelAddressTranslationExtensions
        features = Get-FeatureSnapshot
        pending_reboot_registry_paths = $pendingBefore
    }

    $install = Invoke-WslCommand -Arguments @(
        "--install",
        "--distribution", $Distribution,
        "--no-launch",
        "--web-download"
    )
    $record.commands += $install

    if ($install.exit_code -eq 0) {
        $update = Invoke-WslCommand -Arguments @("--update", "--web-download")
        $record.commands += $update
    }

    $featuresAfter = Get-FeatureSnapshot
    $list = Invoke-WslCommand -Arguments @("--list", "--verbose")
    $record.commands += $list

    $pendingAfter = @($pendingRebootPaths | Where-Object {
        Test-Path -LiteralPath $_
    })
    $featurePending = @($featuresAfter | Where-Object {
        $_.Contains("state") -and $_.state -match "Pending"
    }).Count -gt 0
    $outputRequestsRestart = @($record.commands | ForEach-Object {
        $_.output
    }) -match "restart|reboot|重新启动|重启"

    $record.after = [ordered]@{
        features = $featuresAfter
        pending_reboot_registry_paths = $pendingAfter
    }
    $record.restart_required = [bool](
        $featurePending -or
        $pendingAfter.Count -gt 0 -or
        $outputRequestsRestart.Count -gt 0
    )
    $record.succeeded = [bool](
        $install.exit_code -eq 0 -or $record.restart_required
    )

    if (-not $record.succeeded) {
        throw "wsl.exe --install exited with code $($install.exit_code)."
    }
}
catch {
    $record.failure = $_.Exception.Message
}
finally {
    $record.finished_at = (Get-Date).ToUniversalTime().ToString("o")
    $json = $record | ConvertTo-Json -Depth 10
    [IO.File]::WriteAllText(
        $ResultPath,
        $json + [Environment]::NewLine,
        [Text.UTF8Encoding]::new($false)
    )
}

if (-not $record.succeeded) {
    Write-Error $record.failure
    exit 1
}

if ($record.restart_required) {
    Write-Host "WSL installation staged successfully; Windows must be restarted."
    exit 0
}

Write-Host "WSL installation completed without a detected restart requirement."
exit 0

