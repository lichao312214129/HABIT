[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("AutoML", "Analysis")]
    [string]$Profile
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = [System.IO.Path]::GetFullPath((Split-Path -Parent $ScriptDir)).TrimEnd("\", "/")
$EnvPath = Join-Path $Root ".mamba\envs\habit"
$PythonExe = Join-Path $EnvPath "python.exe"
$ToolsBin = Join-Path $Root "tools\bin"
$ConstraintsLock = Join-Path $ScriptDir "constraints-runtime-win-py310.lock"
$VerifyScript = Join-Path $ScriptDir "verify_habit_env.py"
$profileKey = $Profile.ToLowerInvariant()
$profileConfig = @{
    automl = @{
        Lock = Join-Path $ScriptDir "requirements-automl-win-py310.lock"
        Label = "AutoML"
    }
    analysis = @{
        Lock = Join-Path $ScriptDir "requirements-analysis-win-py310.lock"
        Label = "进阶分析"
    }
}
$selected = $profileConfig[$profileKey]
$StampFile = Join-Path $EnvPath (".habit_profile_" + $profileKey + ".sha256")
$LogDir = Join-Path $Root "logs"
$LogFile = Join-Path $LogDir (
    "profile_" + $profileKey + "_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log"
)

function Write-Log {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Message)
    $Message | Out-File -LiteralPath $LogFile -Encoding utf8 -Append
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    Write-Log ""
    Write-Log ($FilePath + " " + ($Arguments -join " "))
    $previousPreference = $ErrorActionPreference
    try {
        # Pip writes normal progress to stderr, so process exit status remains
        # the authoritative success signal.
        $ErrorActionPreference = "Continue"
        & $FilePath @Arguments 2>&1 | ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            Write-Log $line
        }
        $exitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "命令执行失败，退出码 $exitCode。"
    }
}

function Add-HabitRuntimePath {
    $items = @(
        $EnvPath,
        (Join-Path $EnvPath "Scripts"),
        (Join-Path $EnvPath "Library\bin"),
        $ToolsBin
    )
    $current = [Environment]::GetEnvironmentVariable("Path", "Process")
    $runtime = (($items + @($current)) -join [System.IO.Path]::PathSeparator)
    [Environment]::SetEnvironmentVariable("Path", $runtime, "Process")
    [Environment]::SetEnvironmentVariable("PATH", $runtime, "Process")
}

if (-not (Test-Path -LiteralPath $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

Write-Host ""
Write-Host ("HABIT " + $selected.Label + " 可选功能") -ForegroundColor Green
Write-Host "该入口仅在你需要对应功能时运行。"
Write-Host "日志文件：$LogFile"

try {
    foreach ($required in @(
        $PythonExe,
        $selected.Lock,
        $ConstraintsLock,
        $VerifyScript,
        (Join-Path $EnvPath ".habit_env_spec.sha256")
    )) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "缺少基础安装文件：$required。请先完成「一键安装HABIT」。"
        }
    }
    $hashParts = @(
        foreach ($path in @($selected.Lock, $ConstraintsLock)) {
            (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    )
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($hashParts -join "|"))
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $profileSpec = ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }

    Add-HabitRuntimePath
    if ((Test-Path -LiteralPath $StampFile) -and ((Get-Content -LiteralPath $StampFile -Raw).Trim() -eq $profileSpec)) {
        Write-Host "可选功能规格未变化，跳过依赖安装。"
    }
    else {
        Write-Host ""
        Write-Host ("正在安装 " + $selected.Label + " 的锁定依赖...") -ForegroundColor Cyan
        Invoke-LoggedCommand -FilePath $PythonExe -Arguments @(
            "-m", "pip", "--isolated", "install", "--prefer-binary",
            "-r", $selected.Lock, "-c", $ConstraintsLock
        )
    }
    Invoke-LoggedCommand -FilePath $PythonExe -Arguments @(
        "-u", $VerifyScript, "--profile", $profileKey
    )
    $profileSpec | Out-File -LiteralPath $StampFile -Encoding ascii -Force
    Write-Host ""
    Write-Host ($selected.Label + " 功能已启用并通过导入验证。") -ForegroundColor Green
}
catch {
    Write-Log ("Optional profile installation failed: " + $_.Exception.Message)
    Write-Host ""
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "默认 HABIT 环境未被删除；详情见：$LogFile"
    exit 1
}
