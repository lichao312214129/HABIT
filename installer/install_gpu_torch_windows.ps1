[CmdletBinding()]
param(
    [version]$MinimumWindowsDriver = [version]"527.41",
    [int]$MinimumFreeSpaceGB = 8
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = [System.IO.Path]::GetFullPath((Split-Path -Parent $ScriptDir)).TrimEnd("\", "/")
$EnvPath = Join-Path $Root ".mamba\envs\habit"
$PythonExe = Join-Path $EnvPath "python.exe"
$ToolsBin = Join-Path $Root "tools\bin"
$GpuLock = Join-Path $ScriptDir "requirements-gpu-torch-win-py310.lock"
$ConstraintsLock = Join-Path $ScriptDir "constraints-runtime-win-py310.lock"
$VerifyScript = Join-Path $ScriptDir "verify_habit_env.py"
$GpuStamp = Join-Path $EnvPath ".habit_gpu_spec.sha256"
$LogDir = Join-Path $Root "logs"
$LogFile = Join-Path $LogDir ("gpu_install_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

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
Write-Host "HABIT NVIDIA GPU 一键增强" -ForegroundColor Green
Write-Host "日志文件：$LogFile"

try {
    foreach ($required in @($PythonExe, $GpuLock, $ConstraintsLock, $VerifyScript)) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "缺少基础安装文件：$required。请先完成「一键安装HABIT」。"
        }
    }
    if (-not [Environment]::Is64BitOperatingSystem) {
        throw "GPU 增强仅支持 64 位 Windows。"
    }
    $drive = New-Object System.IO.DriveInfo([System.IO.Path]::GetPathRoot($Root))
    $freeGB = [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
    if ($freeGB -lt $MinimumFreeSpaceGB) {
        throw "GPU wheel、临时文件和回滚至少需要 $MinimumFreeSpaceGB GB，当前仅剩 $freeGB GB。"
    }

    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        throw "未找到 nvidia-smi。未检测到可用 NVIDIA 驱动，CPU 环境保持不变。"
    }
    $gpuRows = @(
        & $nvidiaSmi.Source `
            "--query-gpu=driver_version,name,memory.total" `
            "--format=csv,noheader,nounits" 2>$null
    )
    if ($LASTEXITCODE -ne 0 -or $gpuRows.Count -lt 1) {
        throw "nvidia-smi 无法读取 GPU 信息，CPU 环境保持不变。"
    }
    $firstParts = @($gpuRows[0].Split(",") | ForEach-Object { $_.Trim() })
    if ($firstParts.Count -lt 3) {
        throw "无法解析 nvidia-smi 输出：$($gpuRows[0])"
    }
    try {
        $driverVersion = [version]$firstParts[0]
    }
    catch {
        throw "无法解析 NVIDIA 驱动版本：$($firstParts[0])"
    }
    if ($driverVersion -lt $MinimumWindowsDriver) {
        throw "NVIDIA 驱动 $driverVersion 过旧；CUDA 12.1 至少需要 Windows 驱动 $MinimumWindowsDriver。CPU 环境保持不变。"
    }
    Write-Host "检测到：$($firstParts[1])，显存 $($firstParts[2]) MB，驱动 $driverVersion"
    Write-Log ($gpuRows -join [Environment]::NewLine)

    Add-HabitRuntimePath
    $gpuSpecHash = (Get-FileHash -LiteralPath $GpuLock -Algorithm SHA256).Hash.ToLowerInvariant()

    Write-Host ""
    Write-Host "正在安装锁定的 CUDA 12.1 Torch 层..." -ForegroundColor Cyan
    try {
        Invoke-LoggedCommand -FilePath $PythonExe -Arguments @(
            "-m", "pip", "--isolated", "install", "--force-reinstall",
            "-r", $GpuLock, "-c", $ConstraintsLock
        )
        Invoke-LoggedCommand -FilePath $PythonExe -Arguments @("-u", $VerifyScript, "--gpu")
        $gpuSpecHash | Out-File -LiteralPath $GpuStamp -Encoding ascii -Force
    }
    catch {
        $gpuFailure = $_.Exception.Message
        Write-Warning "GPU 增强失败，正在移除可选 Torch 层：$gpuFailure"
        Invoke-LoggedCommand -FilePath $PythonExe -Arguments @(
            "-m", "pip", "--isolated", "uninstall", "-y", "torch"
        )
        Remove-Item -LiteralPath $GpuStamp -Force -ErrorAction SilentlyContinue
        Invoke-LoggedCommand -FilePath $PythonExe -Arguments @("-u", $VerifyScript)
        throw "GPU 增强失败，已恢复并验证无 Torch 的默认环境。原始错误：$gpuFailure"
    }

    Write-Host ""
    Write-Host "GPU 增强完成，HABIT 已通过 CUDA 实算验证。" -ForegroundColor Green
}
catch {
    Write-Log ("GPU installation failed: " + $_.Exception.Message)
    Write-Host ""
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "详细日志：$LogFile"
    exit 1
}
