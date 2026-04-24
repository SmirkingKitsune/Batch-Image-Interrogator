#!/bin/bash
# ============================================================
# llama.cpp CUDA Build Script for ARM64
# ============================================================
# Builds llama.cpp from source with CUDA enabled for ARM64
# systems (for example DGX Spark / Jetson / Grace Blackwell),
# then installs llama-server into cache/llama_cpp/bin so the
# app can use it directly.
# ============================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_ROOT="${SCRIPT_DIR}/llama_cpp_build"
REPO_DIR="${BUILD_ROOT}/llama.cpp"
INSTALL_DIR="${SCRIPT_DIR}/cache/llama_cpp/bin"
VERSION_FILE="${INSTALL_DIR}/llama-server.version"

LLAMA_CPP_REPO="${LLAMA_CPP_REPO:-https://github.com/ggml-org/llama.cpp}"
LLAMA_CPP_VERSION="${LLAMA_CPP_VERSION:-}"
LLAMA_CPP_CUDA_ARCH="${LLAMA_CPP_CUDA_ARCH:-}"
LLAMA_CPP_JOBS="${LLAMA_CPP_JOBS:-}"
LLAMA_CPP_CURL="${LLAMA_CPP_CURL:-OFF}"

detect_cuda_architecture() {
    local cap=""
    local arch=""

    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]' || true)"
    if [ -z "$cap" ]; then
        cap="$(nvidia-smi -q 2>/dev/null | grep -oE 'CUDA Compute Capability[[:space:]]*:[[:space:]]*[0-9]+\.[0-9]+' | head -n1 | awk -F: '{print $2}' | tr -d '[:space:]' || true)"
    fi

    if [[ "$cap" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
        arch="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
        echo "$arch"
        return
    fi

    # DGX Spark (GB10) default from NVIDIA guidance.
    echo "121"
}

echo ""
echo "============================================================"
echo "LLAMA.CPP CUDA BUILD FOR ARM64"
echo "============================================================"
echo ""
echo -e "${YELLOW}This script compiles llama.cpp from source and installs${NC}"
echo -e "${YELLOW}llama-server into cache/llama_cpp/bin for this project.${NC}"
echo ""

ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ] && [ "$ARCH" != "arm64" ]; then
    echo -e "${RED}[ERROR] This script targets ARM64 systems only.${NC}"
    echo "        Detected architecture: ${ARCH}"
    exit 1
fi
echo -e "${GREEN}[OK] ARM64 architecture detected${NC}"

echo ""
echo -e "${BLUE}Checking prerequisites...${NC}"
echo ""

for cmd in git cmake make nvcc nvidia-smi; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo -e "${RED}[ERROR] Missing required command: ${cmd}${NC}"
        echo "Install required toolchain before running this build."
        exit 1
    fi
done

echo -e "${GREEN}[OK] git installed${NC}"
echo -e "${GREEN}[OK] cmake installed${NC}"
echo -e "${GREEN}[OK] make installed${NC}"
echo -e "${GREEN}[OK] nvcc installed${NC}"
echo -e "${GREEN}[OK] nvidia-smi available${NC}"

CUDA_VERSION="$(nvcc --version | grep -oE 'release[[:space:]]+[0-9]+(\.[0-9]+)?' | awk '{print $2}' | head -n1)"
if [ -z "$CUDA_VERSION" ]; then
    CUDA_VERSION="unknown"
fi
echo "    CUDA Toolkit: ${CUDA_VERSION}"

if [ -z "${CUDA_HOME:-}" ]; then
    if [ -d "/usr/local/cuda" ]; then
        CUDA_HOME="/usr/local/cuda"
    elif [ -d "/opt/cuda" ]; then
        CUDA_HOME="/opt/cuda"
    else
        CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
    fi
fi
echo "    CUDA_HOME: ${CUDA_HOME}"

if [ -z "$LLAMA_CPP_CUDA_ARCH" ]; then
    LLAMA_CPP_CUDA_ARCH="$(detect_cuda_architecture)"
fi
echo "    CUDA Architecture: ${LLAMA_CPP_CUDA_ARCH}"

if [ -z "$LLAMA_CPP_JOBS" ]; then
    CPU_CORES="$(nproc)"
    if [ "$CPU_CORES" -gt 8 ]; then
        LLAMA_CPP_JOBS=8
    else
        LLAMA_CPP_JOBS="$CPU_CORES"
    fi
fi
echo "    Parallel Jobs: ${LLAMA_CPP_JOBS}"

if [ -n "$LLAMA_CPP_VERSION" ]; then
    echo "    llama.cpp version: ${LLAMA_CPP_VERSION}"
else
    echo "    llama.cpp version: latest default branch"
fi

AVAILABLE_SPACE="$(df -BG "${SCRIPT_DIR}" | tail -1 | awk '{print $4}' | sed 's/G//')"
if [ -n "$AVAILABLE_SPACE" ] && [ "$AVAILABLE_SPACE" -lt 5 ]; then
    echo -e "${RED}[ERROR] Insufficient disk space.${NC}"
    echo "        Available: ${AVAILABLE_SPACE}GB, Required: 5GB+"
    exit 1
fi
echo -e "${GREEN}[OK] Disk space check passed (${AVAILABLE_SPACE}GB available)${NC}"

echo ""
echo "Build configuration:"
echo "  - Repository: ${LLAMA_CPP_REPO}"
echo "  - Install path: ${INSTALL_DIR}"
echo "  - CMake flags: -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=${LLAMA_CPP_CUDA_ARCH} -DLLAMA_CURL=${LLAMA_CPP_CURL}"
echo ""
read -p "Continue with llama.cpp source build? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
    echo "Build cancelled."
    exit 0
fi

echo ""
echo -e "${BLUE}[1/5] Preparing build directory...${NC}"
rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}"

echo ""
echo -e "${BLUE}[2/5] Cloning llama.cpp...${NC}"
if [ -n "$LLAMA_CPP_VERSION" ]; then
    git clone --recursive --branch "${LLAMA_CPP_VERSION}" --depth 1 "${LLAMA_CPP_REPO}" "${REPO_DIR}"
else
    git clone --recursive --depth 1 "${LLAMA_CPP_REPO}" "${REPO_DIR}"
fi

echo ""
echo -e "${BLUE}[3/5] Configuring CMake...${NC}"
mkdir -p "${REPO_DIR}/build"
cd "${REPO_DIR}/build"
cmake .. \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${LLAMA_CPP_CUDA_ARCH}" \
    -DLLAMA_CURL="${LLAMA_CPP_CURL}" \
    -DCMAKE_BUILD_TYPE=Release

echo ""
echo -e "${BLUE}[4/5] Building llama.cpp (this can take several minutes)...${NC}"
cmake --build . --config Release -j "${LLAMA_CPP_JOBS}"

LLAMA_SERVER_PATH="${REPO_DIR}/build/bin/llama-server"
if [ ! -f "${LLAMA_SERVER_PATH}" ]; then
    LLAMA_SERVER_PATH="$(find "${REPO_DIR}/build" -type f -name llama-server | head -n1 || true)"
fi

if [ -z "${LLAMA_SERVER_PATH}" ] || [ ! -f "${LLAMA_SERVER_PATH}" ]; then
    echo -e "${RED}[ERROR] Build completed but llama-server was not found.${NC}"
    exit 1
fi

echo ""
echo -e "${BLUE}[5/5] Installing llama-server runtime bundle...${NC}"
mkdir -p "${INSTALL_DIR}"
cp -af "${REPO_DIR}/build/bin/." "${INSTALL_DIR}/"
chmod +x "${INSTALL_DIR}/llama-server"

COMMIT_SHA="$(git -C "${REPO_DIR}" rev-parse --short HEAD)"
DESCRIBE_TAG="$(git -C "${REPO_DIR}" describe --tags --always --dirty 2>/dev/null || echo "${COMMIT_SHA}")"
echo "${DESCRIBE_TAG} (source build, cuda-arch=${LLAMA_CPP_CUDA_ARCH})" > "${VERSION_FILE}"

echo ""
echo "============================================================"
echo -e "${GREEN}BUILD COMPLETE!${NC}"
echo "============================================================"
echo ""
echo "Installed llama-server: ${INSTALL_DIR}/llama-server"
echo "Version marker: ${VERSION_FILE}"
echo ""

echo "Verifying binary..."
"${INSTALL_DIR}/llama-server" --version || true

echo ""
echo "Use this binary in the app:"
echo "  Inquiry -> llama-server Path -> ${INSTALL_DIR}/llama-server"
echo ""

read -p "Delete temporary source/build directory (${BUILD_ROOT}) to free space? (y/n): " CLEANUP
if [ "$CLEANUP" = "y" ] || [ "$CLEANUP" = "Y" ]; then
    rm -rf "${BUILD_ROOT}"
    echo "Temporary build directory removed."
fi

echo ""
echo "Done!"
