@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul

rem This wrapper prompts only for the HABIT wheel build interpreter. Vendored
rem release assets are validated by the PowerShell builder.
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%..") do set "REPO_ROOT=%%~fI"
set "BUILD_SCRIPT=%SCRIPT_DIR%build_lightweight_release.ps1"
set "DEFAULT_BUILD_PYTHON=E:\conda\mconda\envs\py310\python.exe"

if not exist "%BUILD_SCRIPT%" (
    echo [HABIT] Build script was not found: %BUILD_SCRIPT%
    echo.
    pause
    exit /b 1
)

echo.
echo HABIT lightweight release builder
echo The release is written to "%REPO_ROOT%\dist".
echo The current Git worktree must be clean for a formal release.
echo.

:prompt_build_python
set "BUILD_PYTHON="
if exist "%DEFAULT_BUILD_PYTHON%" (
    set /p "BUILD_PYTHON=Build Python path [%DEFAULT_BUILD_PYTHON%]: "
    if not defined BUILD_PYTHON set "BUILD_PYTHON=%DEFAULT_BUILD_PYTHON%"
) else (
    set /p "BUILD_PYTHON=Build Python path: "
)
if not exist "%BUILD_PYTHON%" (
    echo [HABIT] Build Python was not found: %BUILD_PYTHON%
    echo.
    goto prompt_build_python
)

echo.
echo [HABIT] Starting release build...
powershell -NoProfile -ExecutionPolicy Bypass -File "%BUILD_SCRIPT%" -BuildPython "%BUILD_PYTHON%"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if not "%EXIT_CODE%"=="0" (
    echo [HABIT] Release build failed with exit code %EXIT_CODE%.
    pause
    exit /b %EXIT_CODE%
)

echo [HABIT] Release build completed successfully.
pause
exit /b 0
