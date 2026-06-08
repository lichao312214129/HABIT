@echo off
REM Stage all portable-pack assets into the conda env BEFORE conda pack.
REM
REM   conda activate habit
REM   conda_pack\stage_external_tools.bat
REM   conda pack -n habit -o HABIT-win-py310-gpu-v0.1.0.tar.gz
REM
REM Pack root (%%CONDA_PREFIX%%):
REM   setup_habit.bat, install_gpu_torch.cmd, requirements-gpu-torch-only.txt
REM Scripts\:
REM   dcm2niix.exe, elastix.exe, transformix.exe, Par0040affine.txt  (from demo_data\)
REM
REM Optional argument: conda env root directory (if CONDA_PREFIX is unset).

setlocal EnableExtensions EnableDelayedExpansion

set "PACK_ROOT="
if not "%~1"=="" (
    set "PACK_ROOT=%~1"
) else if defined CONDA_PREFIX (
    set "PACK_ROOT=!CONDA_PREFIX!"
)

if not defined PACK_ROOT (
    echo [HABIT] ERROR: conda activate habit first, or pass env root:
    echo         stage_external_tools.bat "C:\path\to\env"
    exit /b 1
)

if not exist "!PACK_ROOT!\python.exe" (
    echo [HABIT] ERROR: python.exe not found in "!PACK_ROOT!"
    exit /b 1
)

set "SCRIPTS=!PACK_ROOT!\Scripts"
if not exist "!SCRIPTS!\" (
    echo [HABIT] ERROR: Scripts not found: "!SCRIPTS!"
    exit /b 1
)

set "BAT_DIR=%~dp0"
set "BAT_DIR=!BAT_DIR:~0,-1!"
set "TOOL_SRC=!BAT_DIR!\..\demo_data"
set "PACK_SRC=!BAT_DIR!"

echo.
echo === Stage conda-pack assets ===
echo Pack root:   !PACK_ROOT!
echo Tool source: !TOOL_SRC!
echo Pack source: !PACK_SRC!
echo.

set "MISSING=0"
for %%F in (dcm2niix.exe elastix.exe transformix.exe Par0040affine.txt) do (
    if not exist "!TOOL_SRC!\%%F" (
        echo [HABIT] ERROR: Missing "!TOOL_SRC!\%%F"
        set "MISSING=1"
    )
)
for %%F in (setup_habit.bat install_gpu_torch.cmd requirements-gpu-torch-only.txt) do (
    if not exist "!PACK_SRC!\%%F" (
        echo [HABIT] ERROR: Missing "!PACK_SRC!\%%F"
        set "MISSING=1"
    )
)
if "!MISSING!"=="1" exit /b 1

echo [1/2] Copy to Scripts\ ...
for %%F in (dcm2niix.exe elastix.exe transformix.exe Par0040affine.txt) do (
    copy /Y "!TOOL_SRC!\%%F" "!SCRIPTS!\%%F" >nul
    if errorlevel 1 (
        echo [HABIT] ERROR: Failed to copy %%F to Scripts
        exit /b 1
    )
    echo       Scripts\%%F
)

echo [2/2] Copy to pack root ...
for %%F in (setup_habit.bat install_gpu_torch.cmd requirements-gpu-torch-only.txt) do (
    copy /Y "!PACK_SRC!\%%F" "!PACK_ROOT!\%%F" >nul
    if errorlevel 1 (
        echo [HABIT] ERROR: Failed to copy %%F to pack root
        exit /b 1
    )
    echo       %%F
)

echo.
echo Done. Next: conda pack -n habit -o your-pack.tar.gz
exit /b 0
