@echo off
REM Build HABIT Windows portable pack (conda pack) — maintainer entry point.
REM
REM Prerequisites:
REM   conda install -c conda-forge conda-pack
REM   conda activate habit
REM   pip install -r requirements.txt
REM   pip install pyradiomics-3.0.1-cp310-cp310-win_amd64.whl   (Windows)
REM   pip install .                                             (NOT pip install -e .)
REM   habit --version
REM
REM Usage:
REM   developer\pack_habit.bat
REM   developer\pack_habit.bat HABIT-win-py310-gpu-v0.1.0.tar.gz
REM
REM Note: In PowerShell, conda activate sets CONDA_PREFIX but cmd may not see
REM       "conda" on PATH. This script resolves conda.exe via CONDA_EXE or
REM       CONDA_PREFIX\..\..\Scripts\conda.exe .

setlocal EnableExtensions EnableDelayedExpansion

if not defined CONDA_PREFIX (
    echo [HABIT] ERROR: conda activate habit first.
    exit /b 1
)

if defined CONDA_DEFAULT_ENV (
    set "ENV_NAME=!CONDA_DEFAULT_ENV!"
) else (
    set "ENV_NAME=habit"
)

set "OUTPUT=%~1"
if "!OUTPUT!"=="" set "OUTPUT=HABIT-win-py310-gpu-v0.1.0.tar.gz"

set "DEV_DIR=%~dp0"

REM --- resolve conda executable (PowerShell activate often omits conda on PATH for cmd) ---
set "CONDA_CMD="
if defined CONDA_EXE if exist "!CONDA_EXE!" set "CONDA_CMD=!CONDA_EXE!"
if not defined CONDA_CMD if exist "!CONDA_PREFIX!\Scripts\conda.exe" (
    set "CONDA_CMD=!CONDA_PREFIX!\Scripts\conda.exe"
)
if not defined CONDA_CMD if exist "!CONDA_PREFIX!\..\..\Scripts\conda.exe" (
    for %%I in ("!CONDA_PREFIX!\..\..\Scripts\conda.exe") do set "CONDA_CMD=%%~fI"
)
if not defined CONDA_CMD (
    where conda >nul 2>&1
    if not errorlevel 1 set "CONDA_CMD=conda"
)
if not defined CONDA_CMD (
    echo [HABIT] ERROR: conda.exe not found.
    echo         Activate the env, or run from Anaconda Prompt, or set CONDA_EXE.
    exit /b 1
)

echo.
echo === HABIT conda pack build ===
echo Environment: !ENV_NAME!
echo Output:      !OUTPUT!
echo Conda:       !CONDA_CMD!
echo.

call "!DEV_DIR!stage_external_tools.bat"
if errorlevel 1 exit /b 1

echo.
echo [3/3] Running conda pack...
"!CONDA_CMD!" pack -n "!ENV_NAME!" -o "!OUTPUT!" --compress-level 9
if errorlevel 1 (
    echo [HABIT] ERROR: conda pack failed.
    exit /b 1
)

echo.
echo Done: !OUTPUT!
exit /b 0
