@echo off

setlocal EnableExtensions

REM Build HABIT Docker image and assemble offline distribution bundle for end users.

REM

REM   developer\export_docker_bundle.bat

REM   developer\export_docker_bundle.bat 0.1.0

REM

REM Path handling:

REM   - Normal drive paths (C:\..., D:\...) — run locally

REM   - WSL network paths (\\wsl.localhost\..., \\wsl$\...) — delegate to WSL bash script

REM   - Other UNC paths — pushd maps a temporary drive letter (cmd cannot cd to UNC)



set "VERSION=%~1"

if "%VERSION%"=="" set "VERSION=0.1.0"



set "IMAGE_TAG=habit:%VERSION%-cpu"

set "TAR_NAME=habit-%VERSION%-cpu.tar"

set "BUNDLE_NAME=HABIT-docker-v%VERSION%"



set "SCRIPT_DIR=%~dp0"

set "REPO_ROOT=%SCRIPT_DIR%.."



REM --- WSL network share: cmd.exe cannot use \\wsl...\ as current directory ---

echo %SCRIPT_DIR%| findstr /I /C:"\\wsl.localhost" /C:"\\wsl$" >nul 2>&1

if not errorlevel 1 goto RunInWsl



REM --- Other UNC paths: map to a temporary drive letter ---

echo %SCRIPT_DIR%| findstr /B "\\\\" >nul 2>&1

if not errorlevel 1 (

    echo [HABIT] UNC path detected — mapping to temporary drive letter via pushd ...

    pushd "%REPO_ROOT%"

    if errorlevel 1 (

        echo [HABIT] ERROR: Cannot access repo root via pushd.

        echo         Try: bash developer/export_docker_bundle.sh

        exit /b 1

    )

    set "REPO_ROOT=%CD%"

    set "SCRIPT_DIR=%REPO_ROOT%\developer\"

    set "_HABIT_USE_POPD=1"

    goto DoExport

)



cd /d "%REPO_ROOT%"

if errorlevel 1 (

    echo [HABIT] ERROR: Cannot cd to repo root.

    exit /b 1

)

set "REPO_ROOT=%CD%"

set "SCRIPT_DIR=%REPO_ROOT%\developer\"

goto DoExport



:RunInWsl

echo.

echo === HABIT Docker bundle export v%VERSION% ^(via WSL^) ===

echo Image tag: %IMAGE_TAG%

echo.

echo [HABIT] WSL repo path detected — running export_docker_bundle.sh inside WSL ...

where wsl >nul 2>&1

if errorlevel 1 (

    echo [HABIT] ERROR: wsl.exe not found. Open Ubuntu and run: bash developer/export_docker_bundle.sh

    exit /b 1

)

for /f "usebackq delims=" %%P in (`wsl wslpath -u "%REPO_ROOT%" 2^>nul`) do set "WSL_REPO=%%P"

if not defined WSL_REPO (

    echo [HABIT] ERROR: wslpath failed. Open Ubuntu and run: bash developer/export_docker_bundle.sh

    exit /b 1

)

wsl -e bash -lc "cd '%WSL_REPO%' && bash developer/export_docker_bundle.sh %VERSION%"

set "EXIT_CODE=%ERRORLEVEL%"

endlocal & exit /b %EXIT_CODE%



:DoExport

set "DOCKER_DIR=%REPO_ROOT%\docker"

set "OUT_DIR=%REPO_ROOT%\%BUNDLE_NAME%"



echo.

echo === HABIT Docker bundle export v%VERSION% ===

echo Image tag: %IMAGE_TAG%

echo Output:    %OUT_DIR%

echo.



python "%SCRIPT_DIR%fix_sh_line_endings.py" developer docker

if errorlevel 1 (

    echo [HABIT] WARNING: fix_sh_line_endings.py failed; bash scripts may have CRLF issues on WSL.

)



echo [1/4] Building Docker image (CPU) ...

set DOCKER_BUILDKIT=1

docker build -f docker/Dockerfile -t %IMAGE_TAG% .

if errorlevel 1 goto CleanupAndFail



echo [2/4] Saving image to tar ...

if not exist "%OUT_DIR%\images" mkdir "%OUT_DIR%\images"

docker save %IMAGE_TAG% -o "%OUT_DIR%\images\%TAR_NAME%"

if errorlevel 1 goto CleanupAndFail



echo [3/4] Copying launcher files ...

if not exist "%OUT_DIR%\data" mkdir "%OUT_DIR%\data"

if not exist "%OUT_DIR%\config" mkdir "%OUT_DIR%\config"

if not exist "%OUT_DIR%\output" mkdir "%OUT_DIR%\output"



for %%F in (

    docker-compose.yml

    start-gui.bat

    stop-gui.bat

    start-gui.sh

    stop-gui.sh

    generate-compose-override.ps1

    generate-compose-override.sh

    ensure-docker-prereqs.ps1

    README.txt

) do copy /Y "%DOCKER_DIR%\%%F" "%OUT_DIR%\%%F" >nul



python "%SCRIPT_DIR%fix_sh_line_endings.py" "%OUT_DIR%"

if errorlevel 1 (

    echo [HABIT] WARNING: fix_sh_line_endings.py failed on bundle output.

)



if exist "%REPO_ROOT%\config" (

    echo [3/4] Copying default config\ ...

    xcopy /E /I /Y "%REPO_ROOT%\config\*" "%OUT_DIR%\config\" >nul

)



echo [4/4] Done.

echo.

echo Bundle directory: %OUT_DIR%

echo User workflow:    unzip ^→ start-gui.bat ^→ http://localhost:8501

echo.



if defined _HABIT_USE_POPD popd

endlocal

exit /b 0



:CleanupAndFail

if defined _HABIT_USE_POPD popd

exit /b 1


