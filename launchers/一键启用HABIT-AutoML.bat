@echo off
chcp 65001 >nul
for %%I in ("%~dp0..") do set "ROOT=%%~fI\"
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%installer\install_optional_profile_windows.ps1" -Profile AutoML
set "EXIT_CODE=%ERRORLEVEL%"
echo.
pause
exit /b %EXIT_CODE%