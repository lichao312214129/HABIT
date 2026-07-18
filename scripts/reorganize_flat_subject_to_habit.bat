@echo off
REM Portable launcher: put this .bat next to the .py in ANY folder, then double-click.
REM Requires Python on PATH (python / pythonw / py).
setlocal
set "SCRIPT_DIR=%~dp0"
set "APP_PY=%SCRIPT_DIR%reorganize_flat_subject_to_habit.py"

if not exist "%APP_PY%" (
  echo ERROR: Cannot find "%APP_PY%"
  echo Keep this .bat in the same folder as reorganize_flat_subject_to_habit.py
  pause
  exit /b 1
)

REM Prefer pythonw so no black console window flashes for the GUI.
where pythonw >nul 2>&1
if %ERRORLEVEL%==0 (
  start "" pythonw "%APP_PY%"
  exit /b 0
)

where python >nul 2>&1
if %ERRORLEVEL%==0 (
  python "%APP_PY%"
  if errorlevel 1 pause
  exit /b %ERRORLEVEL%
)

where py >nul 2>&1
if %ERRORLEVEL%==0 (
  py -3 "%APP_PY%"
  if errorlevel 1 pause
  exit /b %ERRORLEVEL%
)

echo ERROR: Python not found on PATH.
echo Install Python and tick "Add python.exe to PATH", then try again.
pause
exit /b 1
