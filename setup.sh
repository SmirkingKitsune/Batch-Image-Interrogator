#!/bin/bash
# ============================================================
# Image Interrogator - Automated Setup Script (Linux/Mac)
# ============================================================
# This script will:
# - Check Python installation
# - Create/activate virtual environment
# - Detect NVIDIA GPU and CUDA support
# - Install PyTorch with appropriate CUDA version
# - Install all dependencies
# - Verify the setup
# ============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "============================================================"
echo "IMAGE INTERROGATOR - SETUP WIZARD"
echo "============================================================"
echo ""

# Check if Python is installed
echo -e "${BLUE}[1/7] Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python 3 is not installed!${NC}"
    echo "Please install Python 3.10 or higher"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo -e "      Found Python ${PYTHON_VERSION}"

# Check Python version
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MAJOR" -eq 3 -a "$PYTHON_MINOR" -lt 10 ]; then
    echo -e "${RED}[ERROR] Python 3.10 or higher is required!${NC}"
    echo "      Current version: ${PYTHON_VERSION}"
    exit 1
fi
echo -e "${GREEN}      Version check: OK${NC}"
echo ""

# Create virtual environment if it doesn't exist
echo -e "${BLUE}[2/7] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo "      Creating new virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}      Virtual environment created successfully${NC}"
else
    echo "      Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo -e "${BLUE}[3/7] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}      Virtual environment activated${NC}"
echo ""

# Detect OS
OS_TYPE=$(uname -s)
echo -e "${BLUE}[4/7] Detecting GPU and acceleration support...${NC}"
echo ""

INSTALL_CUDA="n"
INSTALL_ROCM="n"
GPU_TYPE="cpu"
CUDA_VERSION="cpu"

# Check for AMD GPU with ROCm (Linux only)
if [ "$OS_TYPE" = "Linux" ]; then
    if command -v rocm-smi &> /dev/null || command -v rocminfo &> /dev/null; then
        echo -e "${GREEN}      AMD GPU detected! Getting GPU information...${NC}"
        echo ""

        if command -v rocm-smi &> /dev/null; then
            rocm-smi --showproductname 2>/dev/null || echo "      AMD GPU Present"
        fi
        echo ""

        # Detect ROCm version
        if [ -f /opt/rocm/.info/version ]; then
            ROCM_VERSION=$(cat /opt/rocm/.info/version | cut -d'-' -f1)
            echo -e "      Detected ROCm Version: ${ROCM_VERSION}"
        elif command -v rocminfo &> /dev/null; then
            ROCM_VERSION=$(rocminfo | grep "ROCm Version" | awk '{print $3}' || echo "unknown")
            echo -e "      Detected ROCm Version: ${ROCM_VERSION}"
        else
            ROCM_VERSION="6.0"
            echo -e "      ROCm detected (version unknown, defaulting to 6.0)"
        fi

        # Determine PyTorch ROCm version
        # ROCm 6.0+ uses rocm6.0, ROCm 5.7 uses rocm5.7, etc.
        ROCM_MAJOR=$(echo "$ROCM_VERSION" | cut -d'.' -f1)
        ROCM_MINOR=$(echo "$ROCM_VERSION" | cut -d'.' -f2)
        PYTORCH_ROCM="rocm${ROCM_MAJOR}.${ROCM_MINOR}"

        echo -e "      Recommended PyTorch ROCm Version: ${PYTORCH_ROCM}"
        echo ""
        INSTALL_ROCM="y"
        GPU_TYPE="rocm"
        CUDA_VERSION="$PYTORCH_ROCM"

    # Check for NVIDIA GPU if no AMD GPU found
    elif command -v nvidia-smi &> /dev/null; then
        echo -e "${GREEN}      NVIDIA GPU detected! Getting GPU information...${NC}"
        echo ""
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
        echo ""

        # Detect CUDA version from nvidia-smi
        NVIDIA_SMI_OUTPUT=$(nvidia-smi 2>/dev/null | grep "CUDA Version" || echo "")

        if echo "$NVIDIA_SMI_OUTPUT" | grep -q "13.0"; then
            DETECTED_CUDA="13.0"
            CUDA_INDEX="cu130"
        elif echo "$NVIDIA_SMI_OUTPUT" | grep -q "12.8"; then
            DETECTED_CUDA="12.8"
            CUDA_INDEX="cu128"
        elif echo "$NVIDIA_SMI_OUTPUT" | grep -q "12.6"; then
            DETECTED_CUDA="12.6"
            CUDA_INDEX="cu126"
        elif echo "$NVIDIA_SMI_OUTPUT" | grep -q "12.4"; then
            DETECTED_CUDA="12.4"
            CUDA_INDEX="cu126"
        elif echo "$NVIDIA_SMI_OUTPUT" | grep -q "12."; then
            DETECTED_CUDA="12.x"
            CUDA_INDEX="cu126"
        elif echo "$NVIDIA_SMI_OUTPUT" | grep -q "11.8"; then
            DETECTED_CUDA="11.8"
            CUDA_INDEX="cu118"
        else
            # Default to 12.6 if we can't detect
            DETECTED_CUDA="12.x"
            CUDA_INDEX="cu126"
        fi

        echo -e "      Detected CUDA Version: ${DETECTED_CUDA}"
        echo -e "      Recommended PyTorch CUDA Version: ${CUDA_INDEX}"
        echo ""
        INSTALL_CUDA="y"
        GPU_TYPE="cuda"
        CUDA_VERSION="$CUDA_INDEX"
    else
        echo -e "${YELLOW}      [WARNING] No AMD or NVIDIA GPU detected${NC}"
        echo ""
        echo "      This application can run on CPU, but GPU is HIGHLY recommended"
        echo "      for performance."
        echo "      - AMD GPU: Install ROCm from https://rocm.docs.amd.com/"
        echo "      - NVIDIA GPU: Install drivers from https://www.nvidia.com/download/index.aspx"
        echo ""
    fi
elif [ "$OS_TYPE" = "Darwin" ]; then
    echo "      Running on macOS"
    # Check for Apple Silicon
    if [ "$(uname -m)" = "arm64" ]; then
        echo -e "${YELLOW}      Apple Silicon detected (M1/M2/M3)${NC}"
        echo "      PyTorch will use MPS (Metal Performance Shaders) for acceleration"
        # For Apple Silicon, we use the default PyTorch build
    else
        echo "      Intel Mac detected - CPU mode only"
    fi
fi
echo ""

# Check current PyTorch installation
echo -e "${BLUE}[5/7] Checking PyTorch installation...${NC}"
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(f'PyTorch {torch.__version__}')")
    echo "      Current: ${TORCH_VERSION}"
    echo ""
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
    echo "      CUDA Available: ${CUDA_AVAILABLE}"

    if [ "$INSTALL_CUDA" = "y" ] || [ "$INSTALL_ROCM" = "y" ]; then
        # Check if current PyTorch has GPU support
        if [ "$CUDA_AVAILABLE" = "False" ]; then
            echo ""
            if [ "$INSTALL_ROCM" = "y" ]; then
                echo -e "${YELLOW}      [WARNING] PyTorch is installed but WITHOUT ROCm support!${NC}"
                echo "      Your system has ROCm ${ROCM_VERSION} available."
                echo ""
                read -p "      Reinstall PyTorch with ROCm ${ROCM_VERSION} support? (y/n): " REINSTALL
            else
                echo -e "${YELLOW}      [WARNING] PyTorch is installed but WITHOUT CUDA support!${NC}"
                echo "      Your system has CUDA ${DETECTED_CUDA} available."
                echo ""
                read -p "      Reinstall PyTorch with CUDA ${DETECTED_CUDA} support? (y/n): " REINSTALL
            fi
            if [ "$REINSTALL" = "y" ] || [ "$REINSTALL" = "Y" ]; then
                NEED_PYTORCH="y"
            else
                echo "      Skipping PyTorch reinstall. GPU acceleration will NOT be available."
                NEED_PYTORCH="n"
            fi
        else
            # Check if CUDA version matches
            CURRENT_CUDA=$(python3 -c "import torch; v=torch.__version__; print('cu130' if 'cu130' in v else 'cu128' if 'cu128' in v else 'cu126' if 'cu126' in v else 'cu118' if 'cu118' in v else 'cpu')")

            if [ "$CURRENT_CUDA" != "$CUDA_INDEX" ]; then
                echo ""
                echo -e "${YELLOW}      [INFO] PyTorch CUDA version (${CURRENT_CUDA}) doesn't match system CUDA (${CUDA_INDEX})${NC}"
                echo "      Current installation will work, but may not be optimal."
                echo ""
                read -p "      Reinstall PyTorch with CUDA ${DETECTED_CUDA} for optimal performance? (y/n): " REINSTALL
                if [ "$REINSTALL" = "y" ] || [ "$REINSTALL" = "Y" ]; then
                    NEED_PYTORCH="y"
                else
                    echo "      Keeping current PyTorch installation."
                    NEED_PYTORCH="n"
                fi
            else
                echo -e "${GREEN}      PyTorch with CUDA support is correctly installed${NC}"
                NEED_PYTORCH="n"
            fi
        fi
    else
        echo "      PyTorch is installed (CPU mode)"
        NEED_PYTORCH="n"
    fi
else
    echo "      PyTorch is not installed"
    NEED_PYTORCH="y"
fi
echo ""

# Install or update PyTorch
if [ "$NEED_PYTORCH" = "y" ]; then
    echo -e "${BLUE}[6/7] Installing PyTorch...${NC}"
    echo ""

    # Uninstall existing PyTorch
    echo "      Removing existing PyTorch installation..."
    pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

    if [ "$INSTALL_ROCM" = "y" ]; then
        echo "      Installing PyTorch with ROCm ${ROCM_VERSION} support..."
        echo "      This may take several minutes (downloading ~2GB)..."
        echo ""
        pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
    elif [ "$INSTALL_CUDA" = "y" ]; then
        echo "      Installing PyTorch with CUDA ${DETECTED_CUDA} support..."
        echo "      This may take several minutes (downloading ~2GB)..."
        echo ""
        pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA_VERSION}
    else
        echo "      Installing PyTorch (CPU-only version)..."
        pip install torch torchvision
    fi

    if [ $? -ne 0 ]; then
        echo ""
        echo -e "${RED}      [ERROR] Failed to install PyTorch!${NC}"
        exit 1
    fi
    echo ""
    echo -e "${GREEN}      PyTorch installed successfully!${NC}"
else
    echo -e "${BLUE}[6/7] PyTorch installation...${NC}"
    echo "      Skipping PyTorch installation"
fi
echo ""

# Install other requirements
echo -e "${BLUE}[7/7] Installing remaining dependencies...${NC}"
echo ""

# Install ONNX Runtime based on GPU availability
if [ "$INSTALL_CUDA" = "y" ]; then
    echo "      Installing ONNX Runtime with CUDA support for WD Tagger..."
    pip install "onnxruntime-gpu>=1.16.0" || echo -e "${YELLOW}      [WARNING] Failed to install ONNX Runtime GPU. Continuing anyway...${NC}"
elif [ "$INSTALL_ROCM" = "y" ]; then
    echo "      Installing ONNX Runtime (CPU version - no ROCm pip package available)..."
    echo "      Note: WD Tagger will run on CPU. For ROCm support, build from source."
    pip install "onnxruntime>=1.16.0" || echo -e "${YELLOW}      [WARNING] Failed to install ONNX Runtime. Continuing anyway...${NC}"
else
    echo "      Installing ONNX Runtime (CPU-only)..."
    pip install "onnxruntime>=1.16.0" || echo -e "${YELLOW}      [WARNING] Failed to install ONNX Runtime. Continuing anyway...${NC}"
fi

# Install other dependencies from requirements.txt
echo "      Installing packages from requirements.txt..."
echo "      (This may take a few minutes...)"
echo ""
pip install -r requirements.txt --upgrade
if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}[ERROR] Failed to install dependencies from requirements.txt!${NC}"
    echo "Please check the error messages above."
    exit 1
fi

echo ""
echo -e "${GREEN}      All dependencies installed successfully!${NC}"
echo ""

# Verify installation
echo "============================================================"
echo "VERIFYING INSTALLATION"
echo "============================================================"
echo ""
echo "PyTorch:"
python3 << 'EOF'
import torch
print(f'  Version: {torch.__version__}')
print(f'  CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('  MPS (Apple Silicon) Available: True')
else:
    print('  GPU: N/A (CPU mode)')
EOF
echo ""
echo "ONNX Runtime:"
python3 << 'EOF'
import onnxruntime as ort
print(f'  Version: {ort.__version__}')
providers = ort.get_available_providers()
print(f'  CUDA Provider: {"Yes" if "CUDAExecutionProvider" in providers else "No"}')
EOF
echo ""
echo "Critical Dependencies:"
python3 -c "import PyQt6; print('  PyQt6: Installed')" 2>/dev/null || echo -e "${RED}  [ERROR] PyQt6: NOT INSTALLED${NC}"
python3 -c "from clip_interrogator import Interrogator; print('  CLIP Interrogator: Installed')" 2>/dev/null || echo -e "${RED}  [ERROR] CLIP Interrogator: NOT INSTALLED${NC}"
python3 -c "import PIL; print('  Pillow: Installed')" 2>/dev/null || echo -e "${RED}  [ERROR] Pillow: NOT INSTALLED${NC}"
echo ""

if [ "$INSTALL_ROCM" = "y" ]; then
    PYTORCH_OK=false

    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        PYTORCH_OK=true
        echo -e "${GREEN}[SUCCESS] ROCm GPU acceleration is enabled!${NC}"
        echo "  - PyTorch: ROCm enabled (for CLIP models)"
        echo "  - ONNX Runtime: CPU mode (no ROCm pip package available)"
        echo ""
        echo -e "${YELLOW}  Note: For ONNX Runtime ROCm support, you need to build from source${NC}"
        echo "  Visit: https://onnxruntime.ai/docs/build/eps.html#migraphx"
        echo ""
    else
        echo -e "${YELLOW}[WARNING] PyTorch ROCm is not available!${NC}"
        echo ""
    fi

elif [ "$INSTALL_CUDA" = "y" ]; then
    PYTORCH_OK=false
    ONNX_OK=false

    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        PYTORCH_OK=true
    else
        echo -e "${YELLOW}[WARNING] PyTorch CUDA is not available!${NC}"
        echo ""
    fi

    if python3 -c "import onnxruntime as ort; exit(0 if 'CUDAExecutionProvider' in ort.get_available_providers() else 1)" 2>/dev/null; then
        ONNX_OK=true
    else
        echo -e "${YELLOW}[WARNING] ONNX Runtime CUDA is not available!${NC}"
        echo ""
    fi

    if [ "$PYTORCH_OK" = true ] && [ "$ONNX_OK" = true ]; then
        echo -e "${GREEN}[SUCCESS] GPU acceleration is fully enabled!${NC}"
        echo "  - PyTorch: CUDA enabled (for CLIP models)"
        echo "  - ONNX Runtime: CUDA enabled (for WD Tagger models)"
        echo ""
    fi
fi

echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "You can now run the application with: ./run.sh"
echo ""
