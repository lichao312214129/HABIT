@echo off
chcp 65001 >nul
set "ROOT=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%installer\install_optional_profile_windows.ps1" -Profile Analysis
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%