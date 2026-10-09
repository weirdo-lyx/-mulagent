# 离线检查脚本控制流：替换 Read-Host 和 python，不访问模型服务，也不读取真实密钥。
# 分别模拟成功和失败，检查三次调用顺序以及退出后的进程密钥恢复。
$ErrorActionPreference = 'Stop'
$pilotScript = Join-Path (Split-Path $PSScriptRoot -Parent) 'scripts\run_api_pilot.ps1'
$keyNames = @('DEEPSEEK_API_KEY', 'GPT_RELAY_API_KEY', 'CLAUDE_RELAY_API_KEY')
$originalValues = @{}
foreach ($name in $keyNames) {
    $originalValues[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
    [Environment]::SetEnvironmentVariable($name, $null, 'Process')
}
$global:researchPilotCalls = [System.Collections.Generic.List[string]]::new()
$global:researchPilotFail = $false

function Read-Host {
    param([string]$Prompt, [switch]$AsSecureString)
    if (-not $AsSecureString) { throw 'Secret prompt must use AsSecureString.' }
    ConvertTo-SecureString 'OFFLINE-FIXTURE-NOT-A-KEY' -AsPlainText -Force
}

function python {
    foreach ($name in $keyNames) {
        if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($name, 'Process'))) {
            throw 'All private keys must be available before a pilot call.'
        }
    }
    $arguments = $args -join ' '
    if ($arguments -notmatch '--mode live' -or $arguments -notmatch '--limit 10' -or
        $arguments -match '--full-run') {
        throw 'Unexpected pilot command.'
    }
    $global:researchPilotCalls.Add($arguments)
    $global:LASTEXITCODE = [int]$global:researchPilotFail
}

try {
    & $pilotScript
    if ($global:researchPilotCalls.Count -ne 3) { throw 'Expected exactly V1, V2, and V3.' }
    if ($global:researchPilotCalls[0] -notmatch '--verifier v1' -or
        $global:researchPilotCalls[1] -notmatch '--verifier v2' -or
        $global:researchPilotCalls[2] -notmatch '--verifier v3') { throw 'Wrong Verifier order.' }
    foreach ($name in $keyNames) {
        if ([Environment]::GetEnvironmentVariable($name, 'Process')) { throw 'Key was not restored.' }
    }
    $global:researchPilotFail = $true
    $failureObserved = $false
    try { & $pilotScript } catch {
        $failureObserved = $_.Exception.Message -match 'pilot reported failures'
    }
    if (-not $failureObserved) { throw 'Failed pilots must be reported.' }
    foreach ($name in $keyNames) {
        if ([Environment]::GetEnvironmentVariable($name, 'Process')) {
            throw 'Key was not restored after a failed pilot.'
        }
    }
    'API pilot wrapper: success and failure restoration checks passed; external requests=0'
} finally {
    foreach ($name in $keyNames) {
        [Environment]::SetEnvironmentVariable($name, $originalValues[$name], 'Process')
    }
    Remove-Variable researchPilotCalls, researchPilotFail -Scope Global -ErrorAction SilentlyContinue
}
