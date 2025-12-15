@echo off
REM Quick run script for Image Interrogator (Windows)
REM For first-time setup or CUDA issues, run: setup.bat

setlocal enabledelayedexpansion

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
echo.
echo [*] Starting Image Interrogator...
echo.
python main.py

if errorlevel 1 (
    echo.
    echo [X] Application exited with errors
    pause
)

endlocal
