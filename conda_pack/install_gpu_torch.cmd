@echo off
REM Upgrade CPU torch to GPU torch inside a conda-pack HABIT environment.
REM Place at pack root (same folder as python.exe). Paths from %%~dp0 only.

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1

set "EXIT_CODE=0"

REM Resolve pack root: bat directory, else current directory (must contain python.exe)
set "BAT_DIR=%~dp0"
set "BAT_DIR=!BAT_DIR:~0,-1!"

set "PACK_ROOT="
if exist "!BAT_DIR!\python.exe" set "PACK_ROOT=!BAT_DIR!"
if not defined PACK_ROOT if exist "%CD%\python.exe" set "PACK_ROOT=%CD%"

if not defined PACK_ROOT (
    echo [HABIT] Cannot find pack root. Copy this file next to python.exe, or cd to pack root first.
    set "EXIT_CODE=1"
    goto :finish
)

set "PYTHON=%PACK_ROOT%\python.exe"
set "REQ_FILE=%PACK_ROOT%\requirements-gpu-torch-only.txt"
set "WHEEL=%PACK_ROOT%\torch-2.4.0+cu121-cp310-cp310-win_amd64.whl"

if not exist "%PYTHON%" (
    echo [HABIT] python.exe not found: "%PYTHON%"
    echo [HABIT] Put install_gpu_torch.cmd in the pack root next to python.exe.
    set "EXIT_CODE=1"
    goto :finish
)

echo.
echo === HABIT GPU torch installer ===
echo Pack root: %PACK_ROOT%
echo.

echo Current torch:
"%PYTHON%" -c "import torch; print('  version:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo [HABIT] torch is not installed in this environment.
    set "EXIT_CODE=1"
    goto :finish
)

echo.

if /I "%~1"=="wheel" goto install_wheel
if exist "%WHEEL%" goto install_wheel

if not exist "%REQ_FILE%" (
    echo [HABIT] Missing "%REQ_FILE%" in pack root.
    echo [HABIT] Copy requirements-gpu-torch-only.txt here, or use:
    echo         install_gpu_torch.cmd wheel
    set "EXIT_CODE=1"
    goto :finish
)

echo Installing GPU torch from requirements-gpu-torch-only.txt (~2 GB download)...
"%PYTHON%" -m pip install --upgrade -r "%REQ_FILE%"
if errorlevel 1 (
    set "EXIT_CODE=1"
    goto :finish
)
goto verify

:install_wheel
if not exist "%WHEEL%" (
    echo [HABIT] Wheel not found in pack root: "%WHEEL%"
    echo [HABIT] Download from Baidu Netdisk (code nt7k^):
    echo         https://pan.baidu.com/s/1eY4lmNegCYh5KgQB640FmA?pwd=nt7k
    set "EXIT_CODE=1"
    goto :finish
)

echo Installing GPU torch from local wheel...
"%PYTHON%" -m pip install --upgrade "%WHEEL%"
if errorlevel 1 (
    set "EXIT_CODE=1"
    goto :finish
)

:verify
echo.
echo After install:
"%PYTHON%" -c "import torch; print('  version:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"
if errorlevel 1 (
    set "EXIT_CODE=1"
    goto :finish
)

echo.
echo Done. Run habit as usual; no need to re-run conda-unpack.

:finish
echo.
pause
exit /b %EXIT_CODE%
