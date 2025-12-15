#!/bin/bash
# Quick run script for Image Interrogator (Linux/Mac)
# For first-time setup or CUDA issues, run: ./setup.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Image Interrogator - Quick Launch"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${RED}[!] Virtual environment not found!${NC}"
    echo "    Please run ./setup.sh first for initial setup."
    echo ""
    exit 1
fi

# Activate virtual environment
source venv/bin/activate 2>/dev/null

# Quick health check
echo -e "${BLUE}[*] Performing health check...${NC}"

# Check if PyTorch is installed
if ! python3 -c "import torch" 2>/dev/null; then
    echo -e "${RED}[X] PyTorch not installed!${NC}"
    echo "    Please run ./setup.sh to install dependencies."
    echo ""
    exit 1
fi

# Check CUDA/MPS availability
CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
MPS_AVAILABLE=$(python3 -c "import torch; print(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())" 2>/dev/null || echo "False")

if [ "$CUDA_AVAILABLE" = "True" ]; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    echo -e "${GREEN}[+] GPU Acceleration: ENABLED${NC}"
    echo "    GPU: ${GPU_NAME}"
elif [ "$MPS_AVAILABLE" = "True" ]; then
    echo -e "${GREEN}[+] GPU Acceleration: ENABLED (Apple Silicon MPS)${NC}"
else
    echo -e "${YELLOW}[!] GPU Acceleration: DISABLED (CPU mode)${NC}"

    # Check if NVIDIA GPU is available (Linux only)
    if [ "$(uname -s)" = "Linux" ] && command -v nvidia-smi &> /dev/null; then
        echo ""
        echo -e "${YELLOW}[WARNING] NVIDIA GPU detected but PyTorch cannot use it!${NC}"
        echo "          You have a GPU but PyTorch is using CPU mode."
        echo ""
        echo "    To enable GPU acceleration:"
        echo "    1. Press Ctrl+C to cancel"
        echo "    2. Run ./setup.sh"
        echo "    3. Allow it to reinstall PyTorch with CUDA support"
        echo ""
        read -p "Continue in CPU mode anyway? (y/n): " CONTINUE
        if [ "$CONTINUE" != "y" ] && [ "$CONTINUE" != "Y" ]; then
            echo ""
            echo "Run ./setup.sh to enable GPU acceleration."
            exit 1
        fi
    fi
fi

echo ""
echo -e "${BLUE}[*] Starting Image Interrogator...${NC}"
echo ""
python3 main.py

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}[X] Application exited with errors${NC}"
fi
