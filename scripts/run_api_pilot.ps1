# 可选的终端密钥输入方式：输入时不回显，只在当前进程临时保存，结束后恢复。
# 三个 API 各试跑 10 条训练 Claim，使用 verifier-v3；默认本地配置仍优先于环境值。
$ErrorActionPreference = 'Stop'
$pilotRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$pilotVariables = @('DEEPSEEK_API_KEY', 'GPT_RELAY_API_KEY', 'CLAUDE_RELAY_API_KEY')
$previousValues = @{}
foreach ($variable in $pilotVariables) {
    # 保存调用前的进程环境，finally 同时覆盖成功、失败和用户中断的退出路径。
    $previousValues[$variable] = [Environment]::GetEnvironmentVariable($variable, 'Process')
}

function Set-PilotKeyFromPrompt([string]$variable) {
    # 已有进程环境值时复用，不重复询问。
    if (-not [string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($variable, 'Process'))) {
        return
    }
    $secureValue = Read-Host -Prompt $variable -AsSecureString
    if ($secureValue.Length -eq 0) { throw "Empty key for $variable; no API requests started." }
    # 仅在传给环境变量时短暂转换明文，随后清零并释放这段非托管内存。
    $pointer = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureValue)
    try {
        [Environment]::SetEnvironmentVariable(
            $variable, [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer), 'Process'
        )
    } finally {
        [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
        $secureValue.Dispose()
    }
}

try {
    Get-Command python -ErrorAction Stop | Out-Null
    # 三套密钥均准备好后再发请求，避免因后一个缺密钥而留下半轮试跑。
    foreach ($variable in $pilotVariables) { Set-PilotKeyFromPrompt $variable }
    Push-Location -LiteralPath $pilotRoot
    try {
        $anyFailure = $false
        foreach ($verifier in @('v1', 'v2', 'v3')) {
            & python -m src.run_verifiers --mode live --verifier $verifier --limit 10 --report "outputs/verifiers/live/verifier-v3/${verifier}_summary.json"
            if ($LASTEXITCODE -ne 0) { $anyFailure = $true }
        }
        if ($anyFailure) {
            throw 'A pilot reported failures; inspect outputs/verifiers/live/verifier-v3/*_errors.jsonl before continuing.'
        }
    } finally {
        Pop-Location
    }
} finally {
    foreach ($variable in $pilotVariables) {
        [Environment]::SetEnvironmentVariable($variable, $previousValues[$variable], 'Process')
    }
}
