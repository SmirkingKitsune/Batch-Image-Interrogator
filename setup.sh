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
echo -e "${BLUE}[4/7] Detecting GPU and CUDA support...${NC}"
echo ""

INSTALL_CUDA="n"
CUDA_VERSION="cpu"

# Check for NVIDIA GPU (Linux only, Mac doesn't support NVIDIA CUDA)
if [ "$OS_TYPE" = "Linux" ]; then
    if command -v nvidia-smi &> /dev/null; then
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
        CUDA_VERSION="$CUDA_INDEX"
    else
        echo -e "${YELLOW}      [WARNING] NVIDIA GPU not detected or drivers not installed${NC}"
        echo ""
        echo "      This application can run on CPU, but GPU is HIGHLY recommended"
        echo "      for performance. If you have an NVIDIA GPU, please install"
        echo "      drivers from: https://www.nvidia.com/download/index.aspx"
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

    if [ "$INSTALL_CUDA" = "y" ]; then
        # Check if current PyTorch has CUDA support
        if [ "$CUDA_AVAILABLE" = "False" ]; then
            echo ""
            echo -e "${YELLOW}      [WARNING] PyTorch is installed but WITHOUT CUDA support!${NC}"
            echo "      Your system has CUDA ${DETECTED_CUDA} available."
            echo ""
            read -p "      Reinstall PyTorch with CUDA ${DETECTED_CUDA} support? (y/n): " REINSTALL
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

    if [ "$INSTALL_CUDA" = "y" ]; then
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
pip install -r requirements.txt --upgrade
if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}[ERROR] Failed to install dependencies!${NC}"
    exit 1
fi
echo ""

# Verify installation
echo "============================================================"
echo "VERIFYING INSTALLATION"
echo "============================================================"
echo ""
python3 << 'EOF'
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('MPS (Apple Silicon) Available: True')
else:
    print('GPU: N/A (CPU mode)')
EOF
echo ""

if [ "$INSTALL_CUDA" = "y" ]; then
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        echo -e "${GREEN}[SUCCESS] GPU acceleration is enabled!${NC}"
        echo ""
    else
        echo -e "${YELLOW}[WARNING] CUDA is still not available!${NC}"
        echo "Please check the installation and try running this script again."
        echo ""
    fi
fi

echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "You can now run the application with: ./run.sh"
echo ""
