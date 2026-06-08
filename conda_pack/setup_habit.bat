@echo off
REM One-time setup for conda-pack HABIT (Windows).
REM Copy this file to the pack root (same folder as python.exe), then run it.
REM
REM Pack root detection (no hard-coded paths):
REM   1. Directory containing this .bat, if python.exe is there
REM   2. Else current working directory, if python.exe is there
REM
REM Steps: conda-unpack (once, silent) + prepend Scripts\ and pack root to user PATH (front).

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1

set "EXIT_CODE=0"

REM --- resolve pack root ---
set "BAT_DIR=%~dp0"
set "BAT_DIR=!BAT_DIR:~0,-1!"

set "PACK_ROOT="
if exist "!BAT_DIR!\python.exe" (
    set "PACK_ROOT=!BAT_DIR!"
)
if not defined PACK_ROOT (
    if exist "%CD%\python.exe" (
        set "PACK_ROOT=%CD%"
    )
)
if not defined PACK_ROOT (
    echo.
    echo [HABIT] ERROR: Cannot find conda-pack root ^(python.exe^).
    echo.
    echo Copy setup_habit.bat into the pack root ^(next to python.exe^), then:
    echo   - double-click setup_habit.bat, OR
    echo   - cd to pack root and run: setup_habit.bat
    echo.
    echo Do NOT run the copy under conda_pack\ in the source repo.
    set "EXIT_CODE=1"
    goto :finish
)

set "SCRIPTS=!PACK_ROOT!\Scripts"
set "HABIT_EXE=!SCRIPTS!\habit.exe"
set "CONDA_UNPACK=!SCRIPTS!\conda-unpack.exe"
set "UNPACK_MARKER=!PACK_ROOT!\.habit_unpacked"
set "SETUP_LOG=!PACK_ROOT!\.habit_setup.log"

echo.
echo === HABIT one-time setup ===
echo Pack root: !PACK_ROOT!
echo.

if not exist "!HABIT_EXE!" (
    echo [HABIT] ERROR: habit.exe not found:
    echo         "!HABIT_EXE!"
    set "EXIT_CODE=1"
    goto :finish
)

REM --- conda-unpack: run once, output hidden; only report real setup failures ---
if exist "!UNPACK_MARKER!" (
    echo [1/2] Environment paths already configured; skipping.
) else if exist "!CONDA_UNPACK!" (
    echo [1/2] Configuring environment paths...
    pushd "!PACK_ROOT!"
    "!CONDA_UNPACK!" >"!SETUP_LOG!" 2>&1
    set "UNPACK_ERR=!ERRORLEVEL!"
    popd
    if !UNPACK_ERR! equ 0 (
        echo       Done.
        echo done> "!UNPACK_MARKER!"
    ) else (
        REM Unpack may fail when torch was upgraded before packing; verify habit still works.
        "!HABIT_EXE!" --version >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            echo       Done ^(verified^).
            echo verified> "!UNPACK_MARKER!"
        ) else (
            echo [HABIT] ERROR: Environment setup failed.
            echo         Please re-extract the HABIT pack to a new folder and run setup again.
            echo         Details: "!SETUP_LOG!"
            set "EXIT_CODE=1"
            goto :finish
        )
    )
) else (
    echo [1/2] conda-unpack.exe not found; skipping path fix.
    "!HABIT_EXE!" --version >nul 2>&1
    if errorlevel 1 (
        echo [HABIT] ERROR: habit.exe cannot run. The pack may be incomplete.
        set "EXIT_CODE=1"
        goto :finish
    )
)

REM --- register user PATH: prepend pack dirs so habit.exe wins over older installs ---
echo [2/2] Registering HABIT in user PATH ^(prepend to front^)...

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$scripts='!SCRIPTS!'; $root='!PACK_ROOT!';" ^
  "$userPath = [Environment]::GetEnvironmentVariable('Path', 'User');" ^
  "if (-not $userPath) { $userPath = '' };" ^
  "$parts = @(); if ($userPath) { $parts = $userPath -split ';' | Where-Object { $_ -ne '' } };" ^
  "$scriptsL = $scripts.ToLower(); $rootL = $root.ToLower();" ^
  "$rest = @($parts | Where-Object { $_.ToLower() -ne $scriptsL -and $_.ToLower() -ne $rootL });" ^
  "$newPath = (($scripts, $root) + $rest) -join ';';" ^
  "if ($newPath -ne $userPath) {" ^
  "  [Environment]::SetEnvironmentVariable('Path', $newPath, 'User');" ^
  "  Write-Host ('      Prepended: ' + $scripts);" ^
  "  Write-Host ('      Prepended: ' + $root);" ^
  "} else {" ^
  "  Write-Host '      User PATH already has HABIT directories at the front.';" ^
  "}"
if errorlevel 1 (
    echo [HABIT] ERROR: Failed to update user PATH.
    set "EXIT_CODE=1"
    goto :finish
)

REM --- final verification ---
"!HABIT_EXE!" --version >nul 2>&1
if errorlevel 1 (
    echo [HABIT] ERROR: habit.exe failed after setup.
    set "EXIT_CODE=1"
    goto :finish
)

echo.
if "!EXIT_CODE!"=="0" (
    echo Setup complete.
    echo.
    echo Close this window, open a NEW terminal, then verify:
    echo   where habit
    echo   habit --version
    echo.
    echo Optional GPU torch upgrade ^(NVIDIA GPU only^):
    echo   install_gpu_torch.cmd
)

:finish
echo.
if "!EXIT_CODE!"=="0" (
    echo Press any key to exit...
) else (
    echo Setup finished with errors. Press any key to exit...
)
pause
exit /b !EXIT_CODE!
