[CmdletBinding()]
param([string]$ResultPath)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if (-not $ResultPath) {
    $ResultPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'results\stage11_wsl_boot_audit.json'
}
$taskRecord = [ordered]@{
    captured_at = (Get-Date).ToUniversalTime().ToString('o')
    read_only_system_checks = $true
    failure = $null
}
try {
    $taskCpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $taskRecord.firmware_virtualization = [bool]$taskCpu.VirtualizationFirmwareEnabled
    $taskRecord.slat = [bool]$taskCpu.SecondLevelAddressTranslationExtensions
    $taskRecord.hypervisor_present = [bool](Get-CimInstance Win32_ComputerSystem).HypervisorPresent
    $taskRecord.last_boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')
    $taskRecord.virtual_machine_platform = [string](Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform).State
    $taskBcd = @(& "$env:SystemRoot\System32\bcdedit.exe" /enum 2>&1 | ForEach-Object { [string]$_ })
    $taskRecord.bcd_exit_code = $LASTEXITCODE
    $taskBcdText = $taskBcd -join [Environment]::NewLine
    $taskMatch = [regex]::Match($taskBcdText, '(?im)^hypervisorlaunchtype\s+(\S+)')
    $taskRecord.explicit_hypervisor_launch_type = if ($taskMatch.Success) { $taskMatch.Groups[1].Value } else { $null }
} catch {
    $taskRecord.failure = $_.Exception.Message
} finally {
    [IO.File]::WriteAllText($ResultPath, ($taskRecord | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false))
}
if ($taskRecord.failure) { exit 1 }
