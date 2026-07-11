@echo off
REM Build HABIT Windows portable pack (conda pack) — maintainer entry point.
REM
REM Prerequisites:
REM   conda install -c conda-forge conda-pack
REM   conda activate habit
REM   pip install -r requirements.txt
REM   pip install pyradiomics-3.0.1-cp310-cp310-win_amd64.whl   (Windows)
REM   The script force-reinstalls HABIT from this checkout and validates the
REM   installed copy from a neutral working directory before packing.
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
for %%I in ("!DEV_DIR!..") do set "REPO_ROOT=%%~fI"
set "PYTHON_EXE=!CONDA_PREFIX!\python.exe"
set "BUILD_ROOT=!TEMP!\habit-pack-build-!RANDOM!-!RANDOM!"

if not exist "!PYTHON_EXE!" (
    echo [HABIT] ERROR: python.exe not found in "!CONDA_PREFIX!".
    exit /b 1
)

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

echo [1/4] Building and force-reinstalling HABIT from the current checkout...
mkdir "!BUILD_ROOT!\build" "!BUILD_ROOT!\bdist" "!BUILD_ROOT!\dist" >nul 2>&1
if errorlevel 1 (
    echo [HABIT] ERROR: Cannot create temporary build directory "!BUILD_ROOT!".
    exit /b 1
)
pushd "!REPO_ROOT!"
"!PYTHON_EXE!" setup.py build --build-base "!BUILD_ROOT!\build" bdist_wheel --bdist-dir "!BUILD_ROOT!\bdist" --dist-dir "!BUILD_ROOT!\dist"
if errorlevel 1 (
    popd
    echo [HABIT] ERROR: HABIT wheel build failed.
    exit /b 1
)
popd

set "HABIT_WHEEL="
for %%F in ("!BUILD_ROOT!\dist\HABIT-*.whl") do set "HABIT_WHEEL=%%~fF"
if not defined HABIT_WHEEL (
    echo [HABIT] ERROR: HABIT wheel was not created.
    exit /b 1
)
if not exist "!HABIT_WHEEL!" (
    echo [HABIT] ERROR: HABIT wheel was not created.
    exit /b 1
)
"!PYTHON_EXE!" -m pip install --force-reinstall --no-deps "!HABIT_WHEEL!"
if errorlevel 1 (
    echo [HABIT] ERROR: HABIT wheel installation failed.
    exit /b 1
)

echo.
echo [2/4] Validating the installed package from a neutral directory...
pushd "!TEMP!"
"!PYTHON_EXE!" -s -c "import pathlib, habit; env = pathlib.Path(r'!CONDA_PREFIX!').resolve(); package = pathlib.Path(habit.__file__).resolve(); assert env in package.parents, f'HABIT loaded outside target environment: {package}'; from habit.core.machine_learning.evaluation.metrics import delong_roc_test; from habit.core.habitat_analysis.clustering_features.supervoxel_cext import is_cext_available; from habit.utils.radiomics_preset_utils import get_preset_path; assert is_cext_available(), 'HABIT native C extension is unavailable'; assert pathlib.Path(get_preset_path('voxel')).is_file(), 'Bundled radiomics preset is missing'; print(f'[HABIT] Installed package verified: {package}')"
if errorlevel 1 (
    popd
    echo [HABIT] ERROR: Installed-package validation failed.
    exit /b 1
)
popd
rmdir /S /Q "!BUILD_ROOT!" >nul 2>&1

echo.
echo [3/4] Staging portable tools...
call "!DEV_DIR!stage_external_tools.bat"
if errorlevel 1 exit /b 1

echo.
echo [4/4] Running conda pack...
"!CONDA_CMD!" pack -n "!ENV_NAME!" -o "!OUTPUT!" --compress-level 9
if errorlevel 1 (
    echo [HABIT] ERROR: conda pack failed.
    exit /b 1
)

echo.
echo Done: !OUTPUT!
exit /b 0
