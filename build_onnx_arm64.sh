#!/bin/bash
# ============================================================
# ONNX Runtime GPU Build Script for ARM64
# ============================================================
# This script builds ONNX Runtime from source with CUDA support
# for ARM64 systems (e.g., NVIDIA Jetson, Grace Blackwell).
#
# WARNING: This build process takes 30-60+ minutes and requires
# several GB of disk space.
#
# Prerequisites:
#   - NVIDIA GPU with CUDA support
#   - CUDA Toolkit installed
#   - cuDNN installed
#   - CMake 3.26+
#   - Python 3.10+
#   - Build tools (gcc, g++, make)
# ============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/onnxruntime_build"
ONNX_VERSION="v1.23.2"  # Latest stable release

echo ""
echo "============================================================"
echo "ONNX RUNTIME GPU BUILD FOR ARM64"
echo "============================================================"
echo ""
echo -e "${YELLOW}WARNING: This process takes 30-60+ minutes and requires${NC}"
echo -e "${YELLOW}several GB of disk space. Only proceed if you need GPU${NC}"
echo -e "${YELLOW}acceleration for WD Tagger models on ARM64.${NC}"
echo ""

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    echo -e "${RED}[ERROR] This script is for ARM64 systems only.${NC}"
    echo "        Detected architecture: $ARCH"
    echo "        For x86_64, use: pip install onnxruntime-gpu"
    exit 1
fi
echo -e "${GREEN}[OK] ARM64 architecture detected${NC}"

# Check for virtual environment
if [ ! -d "${SCRIPT_DIR}/venv" ]; then
    echo -e "${RED}[ERROR] Virtual environment not found.${NC}"
    echo "        Please run ./setup.sh first."
    exit 1
fi
echo -e "${GREEN}[OK] Virtual environment found${NC}"

# Activate virtual environment
source "${SCRIPT_DIR}/venv/bin/activate"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
echo -e "${GREEN}[OK] Python ${PYTHON_VERSION}${NC}"

# Check for CUDA
echo ""
echo -e "${BLUE}Checking prerequisites...${NC}"
echo ""

if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}[ERROR] CUDA Toolkit not found (nvcc not in PATH).${NC}"
    echo ""
    echo "Please install CUDA Toolkit:"
    echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"
    echo "  Or download from: https://developer.nvidia.com/cuda-downloads"
    echo ""
    echo "After installation, ensure nvcc is in your PATH:"
    echo "  export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo -e "${GREEN}[OK] CUDA Toolkit ${CUDA_VERSION}${NC}"

# Detect CUDA home
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME="/usr/local/cuda"
    elif [ -d "/opt/cuda" ]; then
        export CUDA_HOME="/opt/cuda"
    else
        CUDA_HOME=$(dirname $(dirname $(which nvcc)))
    fi
fi
echo "    CUDA_HOME: ${CUDA_HOME}"

# Check for cuDNN
CUDNN_FOUND=false
CUDNN_INCLUDE=""
CUDNN_LIB=""

# Check common cuDNN locations (including ARM64-specific paths)
CUDNN_SEARCH_PATHS=(
    "${CUDA_HOME}"
    "/usr"
    "/usr/local/cudnn"
    "/opt/cudnn"
)

CUDNN_INCLUDE_SUBDIRS=(
    "include"
    "include/aarch64-linux-gnu"
    "include/x86_64-linux-gnu"
)

for cudnn_path in "${CUDNN_SEARCH_PATHS[@]}"; do
    for include_subdir in "${CUDNN_INCLUDE_SUBDIRS[@]}"; do
        if [ -f "${cudnn_path}/${include_subdir}/cudnn.h" ] || [ -f "${cudnn_path}/${include_subdir}/cudnn_version.h" ]; then
            CUDNN_INCLUDE="${cudnn_path}/${include_subdir}"
            # Find matching library path
            if [ -d "${cudnn_path}/lib64" ]; then
                CUDNN_LIB="${cudnn_path}/lib64"
            elif [ -d "${cudnn_path}/lib/aarch64-linux-gnu" ]; then
                CUDNN_LIB="${cudnn_path}/lib/aarch64-linux-gnu"
            elif [ -d "${cudnn_path}/lib/x86_64-linux-gnu" ]; then
                CUDNN_LIB="${cudnn_path}/lib/x86_64-linux-gnu"
            elif [ -d "${cudnn_path}/lib" ]; then
                CUDNN_LIB="${cudnn_path}/lib"
            fi
            if [ -n "$CUDNN_LIB" ]; then
                CUDNN_FOUND=true
                break 2
            fi
        fi
    done
done

if [ "$CUDNN_FOUND" = false ]; then
    echo -e "${RED}[ERROR] cuDNN not found.${NC}"
    echo ""
    echo "Please install cuDNN:"
    echo "  Ubuntu/Debian: sudo apt install libcudnn8-dev"
    echo "  Or download from: https://developer.nvidia.com/cudnn"
    exit 1
fi
echo -e "${GREEN}[OK] cuDNN found${NC}"
echo "    Include: ${CUDNN_INCLUDE}"
echo "    Library: ${CUDNN_LIB}"

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo -e "${RED}[ERROR] CMake not found.${NC}"
    echo "Please install CMake 3.26+:"
    echo "  Ubuntu/Debian: sudo apt install cmake"
    echo "  Or: pip install cmake"
    exit 1
fi
CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
echo -e "${GREEN}[OK] CMake ${CMAKE_VERSION}${NC}"

# Check for build tools
if ! command -v gcc &> /dev/null; then
    echo -e "${RED}[ERROR] GCC not found.${NC}"
    echo "Please install build tools:"
    echo "  Ubuntu/Debian: sudo apt install build-essential"
    exit 1
fi
GCC_VERSION=$(gcc --version | head -n1 | awk '{print $NF}')
echo -e "${GREEN}[OK] GCC ${GCC_VERSION}${NC}"

# Check for git
if ! command -v git &> /dev/null; then
    echo -e "${RED}[ERROR] Git not found.${NC}"
    echo "Please install git:"
    echo "  Ubuntu/Debian: sudo apt install git"
    exit 1
fi
echo -e "${GREEN}[OK] Git installed${NC}"

# Check for Eigen (required to avoid download issues)
if [ ! -f "/usr/share/eigen3/cmake/Eigen3Config.cmake" ]; then
    echo -e "${RED}[ERROR] Eigen3 not found.${NC}"
    echo "Please install Eigen3:"
    echo "  Ubuntu/Debian: sudo apt install libeigen3-dev"
    exit 1
fi
echo -e "${GREEN}[OK] Eigen3 installed${NC}"

# Check available disk space (need at least 10GB)
AVAILABLE_SPACE=$(df -BG "${SCRIPT_DIR}" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 10 ]; then
    echo -e "${RED}[ERROR] Insufficient disk space.${NC}"
    echo "        Available: ${AVAILABLE_SPACE}GB, Required: 10GB+"
    exit 1
fi
echo -e "${GREEN}[OK] Disk space: ${AVAILABLE_SPACE}GB available${NC}"

# Check available memory (recommend at least 8GB)
AVAILABLE_MEM=$(free -g | awk '/^Mem:/{print $7}')
if [ "$AVAILABLE_MEM" -lt 4 ]; then
    echo -e "${YELLOW}[WARNING] Low available memory: ${AVAILABLE_MEM}GB${NC}"
    echo "          Build may fail or be very slow. 8GB+ recommended."
fi

echo ""
echo "============================================================"
echo "All prerequisites satisfied. Ready to build."
echo "============================================================"
echo ""
echo "Build configuration:"
echo "  - ONNX Runtime version: ${ONNX_VERSION}"
echo "  - CUDA version: ${CUDA_VERSION}"
echo "  - Python version: ${PYTHON_VERSION}"
echo "  - Build directory: ${BUILD_DIR}"
echo ""
read -p "Continue with build? This will take 30-60+ minutes. (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Build cancelled."
    exit 0
fi

echo ""
echo -e "${BLUE}[1/5] Preparing build directory...${NC}"

# Clean up any previous build
if [ -d "${BUILD_DIR}" ]; then
    echo "      Removing previous build directory..."
    rm -rf "${BUILD_DIR}"
fi
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo ""
echo -e "${BLUE}[2/5] Cloning ONNX Runtime repository...${NC}"
echo "      This may take a few minutes..."
git clone --recursive --branch ${ONNX_VERSION} --depth 1 https://github.com/microsoft/onnxruntime.git
cd onnxruntime

echo ""
echo -e "${BLUE}[3/5] Installing Python build dependencies...${NC}"
pip install numpy packaging wheel

echo ""
echo -e "${BLUE}[4/5] Building ONNX Runtime with CUDA support...${NC}"
echo ""
echo -e "${YELLOW}This will take 30-60+ minutes. Please be patient.${NC}"
echo "You can monitor CPU/memory usage in another terminal with: htop"
echo ""

# Determine number of parallel jobs (use half of available cores to avoid OOM)
NUM_CORES=$(nproc)
PARALLEL_JOBS=$((NUM_CORES / 2))
if [ "$PARALLEL_JOBS" -lt 1 ]; then
    PARALLEL_JOBS=1
fi
echo "Using ${PARALLEL_JOBS} parallel jobs (of ${NUM_CORES} cores)"
echo ""

# Build ONNX Runtime
# Using --config Release for optimized build
# --build_wheel to create pip-installable wheel
# --skip_tests to save time
# --parallel for faster compilation
# Use system Eigen to avoid download issues with hash mismatches
./build.sh \
    --config Release \
    --build_wheel \
    --skip_tests \
    --parallel ${PARALLEL_JOBS} \
    --use_cuda \
    --cuda_home "${CUDA_HOME}" \
    --cudnn_home "$(dirname ${CUDNN_INCLUDE})" \
    --cmake_extra_defines CMAKE_CUDA_ARCHITECTURES="80;86;87;89;90" \
    --cmake_extra_defines Eigen3_DIR=/usr/share/eigen3/cmake \
    --cmake_extra_defines onnxruntime_USE_PREINSTALLED_EIGEN=ON \
    --cmake_extra_defines onnxruntime_ENABLE_CPUINFO=OFF \
    --cmake_extra_defines onnxruntime_DEV_MODE=OFF \
    --cmake_extra_defines CMAKE_CXX_FLAGS="-Wno-error=deprecated-declarations -I/usr/local/cuda/targets/sbsa-linux/include/cccl" \
    --cmake_extra_defines CMAKE_CUDA_FLAGS="-Wno-deprecated-declarations -I/usr/local/cuda/targets/sbsa-linux/include/cccl"

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}[ERROR] Build failed!${NC}"
    echo ""
    echo "Common issues:"
    echo "  - Out of memory: Try closing other applications or add swap"
    echo "  - Missing dependencies: Check error messages above"
    echo "  - CUDA/cuDNN mismatch: Ensure compatible versions"
    echo ""
    echo "For help, see: https://onnxruntime.ai/docs/build/eps.html#cuda"
    exit 1
fi

echo ""
echo -e "${BLUE}[5/5] Installing built wheel...${NC}"

# Find and install the wheel
WHEEL_PATH=$(find "${BUILD_DIR}/onnxruntime/build/Linux/Release/dist" -name "*.whl" | head -1)
if [ -z "$WHEEL_PATH" ]; then
    echo -e "${RED}[ERROR] Could not find built wheel.${NC}"
    exit 1
fi

echo "      Found wheel: $(basename ${WHEEL_PATH})"

# Uninstall existing onnxruntime
pip uninstall -y onnxruntime onnxruntime-gpu 2>/dev/null || true

# Install the new wheel
pip install "${WHEEL_PATH}"

if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] Failed to install wheel.${NC}"
    exit 1
fi

echo ""
echo "============================================================"
echo -e "${GREEN}BUILD COMPLETE!${NC}"
echo "============================================================"
echo ""

# Verify installation
echo "Verifying installation..."
python3 << 'EOF'
import onnxruntime as ort
print(f"ONNX Runtime Version: {ort.__version__}")
providers = ort.get_available_providers()
print(f"Available Providers: {providers}")
if "CUDAExecutionProvider" in providers:
    print("\n[SUCCESS] CUDA support is enabled!")
else:
    print("\n[WARNING] CUDA provider not found. Build may have issues.")
EOF

echo ""
echo "You can now use WD Tagger with GPU acceleration."
echo ""

# Offer to clean up build directory
echo "The build directory uses several GB of disk space."
read -p "Delete build directory to free space? (y/n): " CLEANUP
if [ "$CLEANUP" = "y" ] || [ "$CLEANUP" = "Y" ]; then
    rm -rf "${BUILD_DIR}"
    echo "Build directory removed."
fi

echo ""
echo "Done!"
