@echo off
REM Upgrade CPU torch to GPU torch inside a conda-pack HABIT environment (Windows).
REM
REM Prerequisites:
REM   1. CPU portable pack extracted and setup_habit.bat already run once.
REM   2. Download torch-2.4.0+cu121-cp310-cp310-win_amd64.whl from Baidu Netdisk
REM      (code nt7k) and place it in this pack root (same folder as python.exe).
REM   3. Double-click this file, or run from cmd in the pack root.
REM
REM Paths are resolved from %%~dp0 only; no hard-coded install directory.

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1

set "EXIT_CODE=0"

REM Pack root = directory containing this bat (strip trailing backslash)
set "BAT_DIR=%~dp0"
set "BAT_DIR=!BAT_DIR:~0,-1!"

set "PACK_ROOT="
if exist "!BAT_DIR!\python.exe" set "PACK_ROOT=!BAT_DIR!"
if not defined PACK_ROOT if exist "%CD%\python.exe" set "PACK_ROOT=%CD%"

if not defined PACK_ROOT (
    echo [HABIT] ERROR: Cannot find pack root ^(python.exe^).
    echo         Copy this file next to python.exe, or cd to the pack root first.
    set "EXIT_CODE=1"
    goto :finish
)

set "PYTHON=!PACK_ROOT!\python.exe"
set "WHEEL=!PACK_ROOT!\torch-2.4.0+cu121-cp310-cp310-win_amd64.whl"

if not exist "!PYTHON!" (
    echo [HABIT] ERROR: python.exe not found: "!PYTHON!"
    set "EXIT_CODE=1"
    goto :finish
)

echo.
echo === HABIT GPU torch installer ===
echo Pack root: !PACK_ROOT!
echo.

echo Current torch:
"!PYTHON!" -c "import torch; print('  version:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())" 2>nul
if errorlevel 1 (
    echo [HABIT] ERROR: torch is not installed in this environment.
    set "EXIT_CODE=1"
    goto :finish
)

echo.

if not exist "!WHEEL!" (
    echo [HABIT] ERROR: GPU torch wheel not found in pack root:
    echo         !WHEEL!
    echo.
    echo Download from Baidu Netdisk ^(extract code nt7k^):
    echo   https://pan.baidu.com/s/1eY4lmNegCYh5KgQB640FmA?pwd=nt7k
    echo.
    echo File name: torch-2.4.0+cu121-cp310-cp310-win_amd64.whl  ^(~2 GB^)
    echo Copy the .whl into the pack root ^(same folder as python.exe^), then run this bat again.
    set "EXIT_CODE=1"
    goto :finish
)

echo Installing GPU torch from local wheel...
echo   !WHEEL!
"!PYTHON!" -m pip install --upgrade "!WHEEL!"
if errorlevel 1 (
    echo [HABIT] ERROR: pip install failed.
    set "EXIT_CODE=1"
    goto :finish
)

echo.
echo After install:
"!PYTHON!" -c "import torch; print('  version:', torch.__version__); print('  CUDA available:', torch.cuda.is_available())"
if errorlevel 1 (
    set "EXIT_CODE=1"
    goto :finish
)

echo.
echo Done. Run habit as usual; no need to re-run setup_habit.bat or conda-unpack.

:finish
echo.
if "!EXIT_CODE!"=="0" (
    echo Press any key to exit...
) else (
    echo Install failed. Press any key to exit...
)
pause
exit /b !EXIT_CODE!
