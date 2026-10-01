@echo off
REM Quick run script for Image Interrogator (Windows)
REM For first-time setup or CUDA issues, run: setup.bat

setlocal enabledelayedexpansion

REM Arguments are forwarded to main.py. --electron selects the opt-in Electron
REM front end; without it the PyQt6 interface opens as always.
set "USE_ELECTRON=0"
for %%a in (%*) do (
    if /i "%%~a"=="--electron" set "USE_ELECTRON=1"
)

echo ==========================================
echo Image Interrogator - Quick Launch
echo ==========================================
echo.

REM Check if virtual environment exists
if not exist "venv\" (
    echo [!] Virtual environment not found!
    echo     Please run setup.bat first for initial setup.
    echo.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat >nul 2>&1

REM Quick health check
echo [*] Performing health check...

REM Check if PyTorch is installed
python -c "import torch" >nul 2>&1
if errorlevel 1 (
    echo [X] PyTorch not installed!
    echo     Please run setup.bat to install dependencies.
    echo.
    pause
    exit /b 1
)

REM Check CUDA availability using exit code (more reliable)
python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
if not errorlevel 1 (
    REM CUDA is available
    for /f "tokens=*" %%i in ('python -c "import torch; print(torch.cuda.get_device_name(0))"') do set GPU_NAME=%%i
    echo [+] GPU Acceleration: ENABLED
    echo     GPU: !GPU_NAME!
    goto :run_app
)

REM CUDA is NOT available
echo [!] GPU Acceleration: DISABLED (CPU mode)

REM Check if NVIDIA GPU is physically available
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo.
    echo [WARNING] NVIDIA GPU detected but PyTorch cannot use it!
    echo           You have a GPU but PyTorch is using CPU mode.
    echo.
    echo     To enable GPU acceleration:
    echo     1. Close this window
    echo     2. Run setup.bat
    echo     3. Allow it to reinstall PyTorch with CUDA support
    echo.
    set /p CONTINUE="Continue in CPU mode anyway? (y/n): "
    if /i "!CONTINUE!" neq "y" (
        echo.
        echo Run setup.bat to enable GPU acceleration.
        pause
        exit /b 1
    )
)

:run_app
if "!USE_ELECTRON!"=="1" goto :check_electron

:start_app
echo.
echo [*] Starting Image Interrogator...
echo.
python main.py %*

if errorlevel 1 (
    echo.
    echo [X] Application exited with errors
    pause
)

endlocal
exit /b

REM Electron is only looked for when it was asked for. A missing install stops
REM here rather than quietly opening PyQt6.
:check_electron
echo [*] UI: Electron (--electron)
set "ELECTRON_REL="
if exist "ui_electron\node_modules\electron\path.txt" set /p ELECTRON_REL=<"ui_electron\node_modules\electron\path.txt"
if not defined ELECTRON_REL goto :electron_missing
if not exist "ui_electron\node_modules\electron\dist\!ELECTRON_REL!" goto :electron_missing
echo [+] electron found at ui_electron\node_modules
goto :start_app

:electron_missing
echo [X] --electron requested but Electron is not installed.
echo     Run setup.bat --electron once, or drop the flag
echo     to use the default PyQt6 interface.
echo.
pause
endlocal
exit /b 1
