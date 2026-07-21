@echo off
chcp 65001 >nul
set "ROOT=%~dp0"
set "ENV_ROOT=%ROOT%.mamba\envs\habit"
if not exist "%ENV_ROOT%\python.exe" (
    echo HABIT environment was not found.
    echo Run the one-click installer BAT in this folder first.
    echo.
    pause
    exit /b 1
)
set "PATH=%ENV_ROOT%;%ENV_ROOT%\Scripts;%ENV_ROOT%\Library\bin;%ENV_ROOT%\Library\usr\bin;%ROOT%tools\bin;%PATH%"
set "MAMBA_ROOT_PREFIX=%ROOT%.mamba"
set "PYTHONUTF8=1"
cd /d "%ROOT%"
title HABIT Python 3.10
echo HABIT environment is ready.
echo Try: habit --version
echo.
cmd /K