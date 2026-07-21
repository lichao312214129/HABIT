[CmdletBinding()]
param(
    [int]$MinimumFreeSpaceGB = 20,
    [int]$MaximumRootPathLength = 64
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = [System.IO.Path]::GetFullPath((Split-Path -Parent $ScriptDir)).TrimEnd("\", "/")
$MambaRoot = Join-Path $Root ".mamba"
$EnvPath = Join-Path $MambaRoot "envs\habit"
$PythonExe = Join-Path $EnvPath "python.exe"
$MicromambaExe = Join-Path $Root "tools\micromamba\micromamba.exe"
$ToolsBin = Join-Path $Root "tools\bin"
$EnvironmentFile = Join-Path $ScriptDir "environment-cpu.yml"
$RuntimeLock = Join-Path $ScriptDir "requirements-runtime-win-py310.lock"
$ConstraintsLock = Join-Path $ScriptDir "constraints-runtime-win-py310.lock"
$CondarcFile = Join-Path $ScriptDir "condarc_cn.yml"
$VendorManifestFile = Join-Path $ScriptDir "vendor_assets.json"
$VerifyScript = Join-Path $ScriptDir "verify_habit_env.py"
$PyradiomicsWheel = Join-Path $Root "tools\vendor\pyradiomics-3.0.1-cp310-cp310-win_amd64.whl"
$HabitWheelDir = Join-Path $Root "tools\wheels"
$StampFile = Join-Path $EnvPath ".habit_env_spec.sha256"
$LogDir = Join-Path $Root "logs"
$LogFile = Join-Path $LogDir ("install_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")
$env:MAMBA_ROOT_PREFIX = $MambaRoot
$env:MAMBA_SSL_NO_REVOKE = "true"

# Keep the multi-gigabyte package cache on the installation drive and at a
# short prefix. This avoids filling C: when HABIT is intentionally installed
# elsewhere and reduces exposure to legacy MAX_PATH behavior.
$rootDrive = [System.IO.Path]::GetPathRoot($Root)
$systemDrive = [System.IO.Path]::GetPathRoot("$env:SystemDrive\")
if ($rootDrive.TrimEnd("\") -ieq $systemDrive.TrimEnd("\")) {
    $PackageCache = Join-Path $env:USERPROFILE ".habit-pkgs"
}
else {
    $PackageCache = [System.IO.Path]::Combine($rootDrive, ".habit-pkgs")
}
$env:CONDA_PKGS_DIRS = $PackageCache

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host ""
    Write-Host "== $Message ==" -ForegroundColor Cyan
}

function Write-Log {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Message)
    $Message | Out-File -LiteralPath $LogFile -Encoding utf8 -Append
}

function New-Directory {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Path $Path | Out-Null
    }
}

function Get-Asset {
    param(
        [Parameter(Mandatory = $true)][object]$Manifest,
        [Parameter(Mandatory = $true)][string]$Id
    )
    $asset = @($Manifest.static_assets | Where-Object { "$($_.id)" -eq $Id })
    if ($asset.Count -ne 1) {
        throw "vendor_assets.json 必须且只能包含一个资产：$Id"
    }
    return $asset[0]
}

function Assert-Hash {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][string]$Expected,
        [Parameter(Mandatory = $true)][string]$Name
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "安装包不完整，缺少 $Name：$Path"
    }
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "$Name 的 SHA-256 校验失败，文件可能损坏或被替换。"
    }
}

function Get-HabitWheel {
    $wheels = @(
        Get-ChildItem -LiteralPath $HabitWheelDir -Filter "HABIT-*.whl" -File -ErrorAction SilentlyContinue
    )
    if ($wheels.Count -ne 1) {
        throw "tools\wheels 必须包含且只能包含一个预编译 HABIT wheel。"
    }
    if ($wheels[0].Name -notmatch "(?i)-cp310-cp310-win_amd64\.whl$") {
        throw "HABIT wheel 与 Windows CPython 3.10 x64 ABI 不匹配：$($wheels[0].Name)"
    }
    return $wheels[0].FullName
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
        # Native package managers use stderr for normal progress. Capture both
        # streams and judge success only from the process exit code.
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
    Write-Log "Exit code: $exitCode"
    if ($exitCode -ne 0) {
        throw "命令执行失败，详情见日志：$LogFile"
    }
}

function Add-HabitRuntimePath {
    $items = @(
        $EnvPath,
        (Join-Path $EnvPath "Scripts"),
        (Join-Path $EnvPath "Library\bin"),
        (Join-Path $EnvPath "Library\usr\bin"),
        $ToolsBin
    )
    $current = [Environment]::GetEnvironmentVariable("Path", "Process")
    $runtime = (($items + @($current)) -join [System.IO.Path]::PathSeparator)
    [Environment]::SetEnvironmentVariable("Path", $runtime, "Process")
    [Environment]::SetEnvironmentVariable("PATH", $runtime, "Process")
}

function Assert-InstallPreconditions {
    Write-Step "安装前电脑与路径检查"
    if (-not $IsWindows -and $env:OS -ne "Windows_NT") {
        throw "该安装器仅支持 64 位 Windows 10/11。"
    }
    if (-not [Environment]::Is64BitOperatingSystem) {
        throw "HABIT 仅支持 64 位 Windows。"
    }
    if ($Root.StartsWith("\\") -or $Root.StartsWith("//")) {
        throw "不能从 UNC 或网络共享路径安装，请先复制到本地磁盘。"
    }
    if ($Root.Length -gt $MaximumRootPathLength) {
        throw "安装路径过长（$($Root.Length) 字符），请使用 D:\HABIT 这类短路径。"
    }
    if ($Root -match "\s") {
        throw "安装路径不能包含空格，请使用 D:\HABIT 这类短路径。"
    }
    foreach ($character in $Root.ToCharArray()) {
        if ([int][char]$character -gt 126) {
            throw "安装路径不能包含中文或其他非 ASCII 字符：$Root"
        }
    }
    $rootItem = Get-Item -LiteralPath $Root -Force
    if (($rootItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "安装根目录不能是符号链接或重解析点。"
    }
    $drive = New-Object System.IO.DriveInfo($rootDrive)
    if ($drive.DriveType -ne [System.IO.DriveType]::Fixed) {
        throw "HABIT 必须安装到本地固定磁盘。"
    }
    $freeGB = [math]::Round($drive.AvailableFreeSpace / 1GB, 1)
    if ($freeGB -lt $MinimumFreeSpaceGB) {
        throw "磁盘空间不足：剩余 $freeGB GB，至少需要 $MinimumFreeSpaceGB GB。"
    }
    $probe = Join-Path $Root (".habit_write_probe_" + [guid]::NewGuid().ToString("N"))
    try {
        Set-Content -LiteralPath $probe -Value "ok" -Encoding ascii
    }
    finally {
        Remove-Item -LiteralPath $probe -Force -ErrorAction SilentlyContinue
    }

    $computer = Get-CimInstance Win32_ComputerSystem -ErrorAction SilentlyContinue
    $processors = @(Get-CimInstance Win32_Processor -ErrorAction SilentlyContinue)
    $video = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue)
    $memoryGB = if ($computer) { [math]::Round([double]$computer.TotalPhysicalMemory / 1GB, 1) } else { $null }
    if ($memoryGB -and $memoryGB -lt 8) {
        throw "物理内存不足：$memoryGB GB；HABIT 最低要求 8 GB。"
    }
    if ($memoryGB -and $memoryGB -lt 16) {
        Write-Warning "当前内存为 $memoryGB GB，大型 AutoGluon/影像任务可能较慢。"
    }
    $hardware = [ordered]@{
        timestamp = (Get-Date).ToString("o")
        os = [Environment]::OSVersion.VersionString
        root = $Root
        free_disk_gb = $freeGB
        memory_gb = $memoryGB
        logical_processors = if ($computer) { $computer.NumberOfLogicalProcessors } else { $env:NUMBER_OF_PROCESSORS }
        processors = @($processors | ForEach-Object { $_.Name })
        video_controllers = @($video | ForEach-Object { $_.Name })
        nvidia_smi = [bool](Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue)
    }
    Write-Log ($hardware | ConvertTo-Json -Depth 4)
    Write-Host "磁盘可用：$freeGB GB；内存：$memoryGB GB"
}

function Test-Endpoint {
    param([Parameter(Mandatory = $true)][string]$Uri)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        # The package managers maintain their own TLS stack. A raw TCP probe
        # avoids false failures from legacy Windows PowerShell certificate
        # revocation behavior while still detecting DNS/firewall outages.
        $parsed = New-Object System.Uri($Uri)
        $operation = $client.BeginConnect($parsed.DnsSafeHost, $parsed.Port, $null, $null)
        if (-not $operation.AsyncWaitHandle.WaitOne(5000, $false)) {
            return $false
        }
        $client.EndConnect($operation)
        return $true
    }
    catch {
        return $false
    }
    finally {
        $client.Close()
    }
}

function Assert-Network {
    Write-Step "安装源连通性检查"
    $endpoints = @(
        "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/",
        "https://pypi.tuna.tsinghua.edu.cn/simple/",
        "https://download.pytorch.org/whl/cpu/"
    )
    foreach ($endpoint in $endpoints) {
        if (-not (Test-Endpoint -Uri $endpoint)) {
            Write-Warning "TCP 预检无法连接：$endpoint；将由包管理器进行最终连通性判断。"
            continue
        }
        Write-Host "可访问：$endpoint"
    }
}

function Remove-ManagedEnvironment {
    param([Parameter(Mandatory = $true)][string]$Reason)
    if (-not (Test-Path -LiteralPath $EnvPath)) {
        return
    }
    $resolvedRoot = [System.IO.Path]::GetFullPath($MambaRoot).TrimEnd("\") + "\"
    $resolvedTarget = [System.IO.Path]::GetFullPath($EnvPath)
    if (-not $resolvedTarget.StartsWith($resolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "安全检查失败，拒绝删除项目环境之外的目录：$resolvedTarget"
    }
    Write-Host $Reason -ForegroundColor Yellow
    Remove-Item -LiteralPath $resolvedTarget -Recurse -Force
}

function Get-SpecHash {
    param([Parameter(Mandatory = $true)][string]$HabitWheel)
    $files = @(
        $EnvironmentFile,
        $RuntimeLock,
        $ConstraintsLock,
        $CondarcFile,
        $VendorManifestFile,
        $PyradiomicsWheel,
        $HabitWheel
    )
    $parts = @($files | ForEach-Object {
        if (-not (Test-Path -LiteralPath $_ -PathType Leaf)) {
            throw "环境规格缺少文件：$_"
        }
        (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
    })
    $bytes = [System.Text.Encoding]::UTF8.GetBytes(($parts -join "|"))
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        return ([BitConverter]::ToString($sha.ComputeHash($bytes))).Replace("-", "").ToLowerInvariant()
    }
    finally {
        $sha.Dispose()
    }
}

New-Directory -Path $LogDir
New-Directory -Path $MambaRoot
try {
    New-Directory -Path $PackageCache
}
catch {
    $PackageCache = Join-Path $env:USERPROFILE ".habit-pkgs"
    $env:CONDA_PKGS_DIRS = $PackageCache
    New-Directory -Path $PackageCache
}

Write-Host ""
Write-Host "HABIT Windows 一键安装" -ForegroundColor Green
Write-Host "安装位置：$Root"
Write-Host "日志文件：$LogFile"

try {
    Assert-InstallPreconditions
    Write-Step "校验安装包"
    foreach ($required in @(
        $EnvironmentFile, $RuntimeLock, $ConstraintsLock, $CondarcFile,
        $VendorManifestFile, $VerifyScript
    )) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
            throw "安装包不完整，缺少：$required"
        }
    }
    $manifest = Get-Content -LiteralPath $VendorManifestFile -Raw -Encoding UTF8 | ConvertFrom-Json
    if ("$($manifest.schema)" -ne "habit.vendor-assets/v1") {
        throw "vendor_assets.json schema 无效。"
    }
    Assert-Hash -Path $MicromambaExe -Expected (Get-Asset $manifest "micromamba").sha256 -Name "micromamba"
    Assert-Hash -Path $PyradiomicsWheel -Expected (Get-Asset $manifest "pyradiomics-wheel").sha256 -Name "PyRadiomics wheel"
    Assert-Hash -Path (Join-Path $ToolsBin "dcm2niix.exe") -Expected (Get-Asset $manifest "dcm2niix").sha256 -Name "dcm2niix"
    Assert-Hash -Path (Join-Path $ToolsBin "elastix.exe") -Expected (Get-Asset $manifest "elastix").sha256 -Name "elastix"
    Assert-Hash -Path (Join-Path $ToolsBin "transformix.exe") -Expected (Get-Asset $manifest "transformix").sha256 -Name "transformix"
    $habitWheel = Get-HabitWheel

    Invoke-LoggedCommand -FilePath $MicromambaExe -Arguments @("--version")
    $currentSpec = Get-SpecHash -HabitWheel $habitWheel
    $needsUpdate = $true
    if ((Test-Path -LiteralPath $EnvPath) -and -not (Test-Path -LiteralPath (Join-Path $EnvPath "conda-meta"))) {
        Remove-ManagedEnvironment -Reason "检测到未完成的环境，正在安全重建。"
    }
    if (Test-Path -LiteralPath $PythonExe) {
        $installedVersion = & $PythonExe -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
        if ($LASTEXITCODE -ne 0 -or "$installedVersion".Trim() -ne "3.10.20") {
            Remove-ManagedEnvironment -Reason "Python ABI 与发行合同不一致，正在安全重建。"
        }
        elseif ((Test-Path -LiteralPath $StampFile) -and ((Get-Content -LiteralPath $StampFile -Raw).Trim() -eq $currentSpec)) {
            $needsUpdate = $false
            Write-Host "环境规格未变化，跳过依赖安装。"
        }
        else {
            # Updating in place cannot remove packages that left the contract.
            # Recreate only the project-owned prefix so a lean release never
            # retains obsolete AutoML, Torch, or analysis packages.
            Remove-ManagedEnvironment -Reason "环境规格已变化，正在重建精简且可复现的环境。"
        }
    }

    if ($needsUpdate) {
        Assert-Network
        Write-Step "创建或更新 Python 3.10 环境"
        if (Test-Path -LiteralPath $EnvPath) {
            Invoke-LoggedCommand -FilePath $MicromambaExe -Arguments @(
                "-r", $MambaRoot, "env", "update", "-y", "-p", $EnvPath,
                "-f", $EnvironmentFile, "--rc-file", $CondarcFile
            )
        }
        else {
            Invoke-LoggedCommand -FilePath $MicromambaExe -Arguments @(
                "-r", $MambaRoot, "create", "-y", "-p", $EnvPath,
                "-f", $EnvironmentFile, "--rc-file", $CondarcFile
            )
        }
        Add-HabitRuntimePath
        Write-Step "安装锁定的运行时依赖"
        Invoke-LoggedCommand -FilePath $PythonExe -Arguments @(
            "-m", "pip", "--isolated", "install", "--prefer-binary",
            "-r", $RuntimeLock, "-c", $ConstraintsLock
        )
        Write-Step "安装已校验的本地 wheel"
        Invoke-LoggedCommand -FilePath $PythonExe -Arguments @(
            "-m", "pip", "--isolated", "install", "--no-index", "--no-deps", $PyradiomicsWheel
        )
        Invoke-LoggedCommand -FilePath $PythonExe -Arguments @(
            "-m", "pip", "--isolated", "install", "--no-index", "--no-deps", "--force-reinstall", $habitWheel
        )
    }

    Write-Step "运行安装后能力自检"
    Add-HabitRuntimePath
    Invoke-LoggedCommand -FilePath $PythonExe -Arguments @("-u", $VerifyScript)
    $currentSpec | Out-File -LiteralPath $StampFile -Encoding ascii -Force
    Write-Host ""
    Write-Host "HABIT 安装完成。以后在 launchers 目录中双击「启动HABIT命令行.bat」即可。" -ForegroundColor Green
}
catch {
    Write-Log ""
    Write-Log ("Installation failed: " + $_.Exception.Message)
    Write-Host ""
    Write-Host "HABIT 安装失败：" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "详细日志：$LogFile"
    exit 1
}
