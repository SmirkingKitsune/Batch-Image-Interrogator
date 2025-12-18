@echo off
REM ============================================================
REM Image Interrogator - Automated Setup Script (Windows)
REM ============================================================
REM This script will:
REM - Check Python installation
REM - Create/activate virtual environment
REM - Detect NVIDIA GPU and CUDA support
REM - Install PyTorch with appropriate CUDA version
REM - Install all dependencies
REM - Verify the setup
REM ============================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================
echo IMAGE INTERROGATOR - SETUP WIZARD
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.10 or higher from python.org
    pause
    exit /b 1
)

echo [1/7] Checking Python installation...
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo       Found Python %PYTHON_VERSION%

REM Check Python version (basic check)
python -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.10 or higher is required!
    echo       Current version: %PYTHON_VERSION%
    pause
    exit /b 1
)
echo       Version check: OK
echo.

REM Create virtual environment if it doesn't exist
echo [2/7] Setting up virtual environment...
if not exist "venv\" (
    echo       Creating new virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo       Virtual environment created successfully
) else (
    echo       Virtual environment already exists
)
echo.

REM Activate virtual environment
echo [3/7] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment!
    pause
    exit /b 1
)
echo       Virtual environment activated
echo.

REM Upgrade pip first
echo       Upgrading pip...
python -m pip install --upgrade pip >nul 2>&1

REM Check for NVIDIA GPU
echo [4/7] Detecting GPU and CUDA support...
echo.
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo       [WARNING] NVIDIA GPU not detected or drivers not installed
    echo.
    echo       This application can run on CPU, but GPU is HIGHLY recommended
    echo       for performance. If you have an NVIDIA GPU, please install
    echo       drivers from: https://www.nvidia.com/download/index.aspx
    echo.
    set INSTALL_CUDA=n
    set CUDA_VERSION=cpu
) else (
    echo       NVIDIA GPU detected! Getting GPU information...
    echo.
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
    echo.

    REM Detect CUDA version - write to temp file to avoid parsing issues
    nvidia-smi > "%TEMP%\nvidia_smi_output.txt" 2>&1

    set DETECTED_CUDA=unknown
    set CUDA_INDEX=cu126

    REM Check for different CUDA versions by searching the file
    findstr /C:"CUDA Version: 13.0" "%TEMP%\nvidia_smi_output.txt" >nul 2>&1
    if not errorlevel 1 (
        set DETECTED_CUDA=13.0
        set CUDA_INDEX=cu130
        goto :cuda_found
    )

    findstr /C:"CUDA Version: 12.8" "%TEMP%\nvidia_smi_output.txt" >nul 2>&1
    if not errorlevel 1 (
        set DETECTED_CUDA=12.8
        set CUDA_INDEX=cu128
        goto :cuda_found
    )

    findstr /C:"CUDA Version: 12.6" "%TEMP%\nvidia_smi_output.txt" >nul 2>&1
    if not errorlevel 1 (
        set DETECTED_CUDA=12.6
        set CUDA_INDEX=cu126
        goto :cuda_found
    )

    findstr /C:"CUDA Version: 12.4" "%TEMP%\nvidia_smi_output.txt" >nul 2>&1
    if not errorlevel 1 (
        set DETECTED_CUDA=12.4
        set CUDA_INDEX=cu126
        goto :cuda_found
    )

    findstr /C:"CUDA Version: 12." "%TEMP%\nvidia_smi_output.txt" >nul 2>&1
    if not errorlevel 1 (
        set DETECTED_CUDA=12.x
        set CUDA_INDEX=cu126
        goto :cuda_found
    )

    findstr /C:"CUDA Version: 11.8" "%TEMP%\nvidia_smi_output.txt" >nul 2>&1
    if not errorlevel 1 (
        set DETECTED_CUDA=11.8
        set CUDA_INDEX=cu118
        goto :cuda_found
    )

    REM Default to 12.6 if we can't detect
    set DETECTED_CUDA=12.x
    set CUDA_INDEX=cu126

    :cuda_found
    del "%TEMP%\nvidia_smi_output.txt" >nul 2>&1
    echo       Detected CUDA Version: !DETECTED_CUDA!
    echo       Recommended PyTorch CUDA Version: !CUDA_INDEX!
    echo.
    set INSTALL_CUDA=y
    set CUDA_VERSION=!CUDA_INDEX!
)
echo.

REM Check current PyTorch installation
echo [5/7] Checking PyTorch installation...
python -c "import torch; print(f'      Current: PyTorch {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo       PyTorch is not installed
    set NEED_PYTORCH=y
) else (
    echo.
    python -c "import torch; print('      CUDA Available: ' + str(torch.cuda.is_available()))"

    if "!INSTALL_CUDA!"=="y" (
        REM Check if current PyTorch has CUDA support
        python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
        if errorlevel 1 (
            echo.
            echo       [WARNING] PyTorch is installed but WITHOUT CUDA support!
            echo       Your system has CUDA !DETECTED_CUDA! available.
            echo.
            set /p REINSTALL="      Reinstall PyTorch with CUDA !DETECTED_CUDA! support? (y/n): "
            if /i "!REINSTALL!"=="y" (
                set NEED_PYTORCH=y
            ) else (
                echo       Skipping PyTorch reinstall. GPU acceleration will NOT be available.
                set NEED_PYTORCH=n
            )
        ) else (
            REM Check if CUDA version matches
            for /f "tokens=*" %%i in ('python -c "import torch; v=torch.__version__; print('cu130' if 'cu130' in v else 'cu128' if 'cu128' in v else 'cu126' if 'cu126' in v else 'cpu')"') do set CURRENT_CUDA=%%i

            if not "!CURRENT_CUDA!"=="!CUDA_INDEX!" (
                echo.
                echo       [INFO] PyTorch CUDA version ^(!CURRENT_CUDA!^) does not match system CUDA ^(!CUDA_INDEX!^)
                echo       Current installation will work, but may not be optimal.
                echo.
                set /p REINSTALL="      Reinstall PyTorch with CUDA !DETECTED_CUDA! for optimal performance? (y/n): "
                if /i "!REINSTALL!"=="y" (
                    set NEED_PYTORCH=y
                ) else (
                    echo       Keeping current PyTorch installation.
                    set NEED_PYTORCH=n
                )
            ) else (
                echo       PyTorch with CUDA support is correctly installed
                set NEED_PYTORCH=n
            )
        )
    ) else (
        echo       PyTorch is installed (CPU mode)
        set NEED_PYTORCH=n
    )
)
echo.

REM Install or update PyTorch
if "!NEED_PYTORCH!"=="y" (
    echo [6/7] Installing PyTorch...
    echo.

    REM Uninstall existing PyTorch
    echo       Removing existing PyTorch installation...
    pip uninstall -y torch torchvision torchaudio >nul 2>&1

    if "!INSTALL_CUDA!"=="y" (
        echo       Installing PyTorch with CUDA !DETECTED_CUDA! support...
        echo       This may take several minutes (downloading ~2GB^)...
        echo.

        REM Install dependencies from PyPI first to avoid conflicts
        pip install typing-extensions filelock networkx jinja2 fsspec

        REM Then install PyTorch from the CUDA-specific index
        pip install torch torchvision --index-url https://download.pytorch.org/whl/!CUDA_VERSION! --no-deps

        REM Install any missing dependencies
        pip install torch torchvision --index-url https://download.pytorch.org/whl/!CUDA_VERSION!
    ) else (
        echo       Installing PyTorch (CPU-only version^)...
        pip install torch torchvision
    )

    if errorlevel 1 (
        echo.
        echo       [ERROR] Failed to install PyTorch!
        echo       Trying alternative installation method...
        echo.

        REM Alternative: Install without using index-url dependencies
        if "!INSTALL_CUDA!"=="y" (
            pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/!CUDA_VERSION!
        )
    )

    REM Verify PyTorch installed
    python -c "import torch" >nul 2>&1
    if errorlevel 1 (
        echo.
        echo       [ERROR] PyTorch installation failed!
        pause
        exit /b 1
    )

    echo.
    echo       PyTorch installed successfully!
) else (
    echo [6/7] PyTorch installation...
    echo       Skipping PyTorch installation
)
echo.

REM Install other requirements
echo [7/7] Installing remaining dependencies...
echo.

REM Install ONNX Runtime based on GPU availability
if "!INSTALL_CUDA!"=="y" (
    echo       Installing ONNX Runtime with GPU support for WD Tagger...
    pip install onnxruntime-gpu>=1.16.0
    if errorlevel 1 (
        echo.
        echo       [WARNING] Failed to install ONNX Runtime GPU. Continuing anyway...
        echo.
    )
) else (
    echo       Installing ONNX Runtime (CPU-only^)...
    pip install onnxruntime>=1.16.0
    if errorlevel 1 (
        echo.
        echo       [WARNING] Failed to install ONNX Runtime. Continuing anyway...
        echo.
    )
)

REM Install other dependencies from requirements.txt
echo       Installing packages from requirements.txt...
echo       (This may take a few minutes...)
echo.
pip install -r requirements.txt --upgrade
if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies from requirements.txt!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo       All dependencies installed successfully!
echo.

REM Verify installation
echo ============================================================
echo VERIFYING INSTALLATION
echo ============================================================
echo.
echo PyTorch:
python -c "import torch; print(f'  Version: {torch.__version__}'); print(f'  CUDA Available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.
echo ONNX Runtime:
python -c "import onnxruntime as ort; print(f'  Version: {ort.__version__}'); providers = ort.get_available_providers(); print(f'  CUDA Provider: {\"Yes\" if \"CUDAExecutionProvider\" in providers else \"No\"}')"
echo.
echo Critical Dependencies:
python -c "import PyQt6; print('  PyQt6: Installed')" 2>nul || echo   [ERROR] PyQt6: NOT INSTALLED
python -c "from clip_interrogator import Interrogator; print('  CLIP Interrogator: Installed')" 2>nul || echo   [ERROR] CLIP Interrogator: NOT INSTALLED
python -c "import PIL; print('  Pillow: Installed')" 2>nul || echo   [ERROR] Pillow: NOT INSTALLED
echo.

if "!INSTALL_CUDA!"=="y" (
    python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] PyTorch CUDA is not available!
        echo.
    )

    python -c "import onnxruntime as ort; exit(0 if 'CUDAExecutionProvider' in ort.get_available_providers() else 1)" >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] ONNX Runtime CUDA is not available!
        echo.
    ) else (
        echo [SUCCESS] GPU acceleration is fully enabled!
        echo   - PyTorch: CUDA enabled (for CLIP models)
        echo   - ONNX Runtime: CUDA enabled (for WD Tagger models)
        echo.
    )
)

echo ============================================================
echo SETUP COMPLETE!
echo ============================================================
echo.
echo You can now run the application with: run.bat
echo.
pause
