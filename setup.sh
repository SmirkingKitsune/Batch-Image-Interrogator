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
# - Provision llama.cpp llama-server from GitHub releases
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
DEFAULT_LLAMA_BIN="$SCRIPT_DIR/cache/llama_cpp/bin/llama-server"
LLAMA_BINARY_PATH="$DEFAULT_LLAMA_BIN"
LLAMA_STATUS="not_checked"
LLAMA_VERSION=""
LLAMA_MESSAGE=""

detect_cuda_version() {
    local detected=""
    local source="unknown"
    local nvcc_output=""
    local smi_output=""
    local cuda_root=""
    local version_json=""

    nvcc_output="$(nvcc --version 2>/dev/null || true)"
    if [ -n "$nvcc_output" ]; then
        detected="$(echo "$nvcc_output" | grep -oE 'release[[:space:]]+[0-9]+(\.[0-9]+)?' | awk '{print $2}' | head -n1)"
        if [ -n "$detected" ]; then
            source="nvcc"
        fi
    fi

    if [ -z "$detected" ]; then
        for cuda_root in "${CUDA_HOME:-}" "${CUDA_PATH:-}" /usr/local/cuda /opt/cuda; do
            [ -n "$cuda_root" ] || continue
            version_json="${cuda_root}/version.json"
            if [ -f "$version_json" ]; then
                detected="$(grep -oE '"cuda"[[:space:]]*:[[:space:]]*"[0-9]+(\.[0-9]+)?' "$version_json" | head -n1 | grep -oE '[0-9]+(\.[0-9]+)?')"
                if [ -n "$detected" ]; then
                    source="version.json"
                    break
                fi
            fi
        done
    fi

    if [ -z "$detected" ]; then
        smi_output="$(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version:[[:space:]]*[0-9]+(\.[0-9]+)?' || true)"
        if [ -n "$smi_output" ]; then
            detected="$(echo "$smi_output" | grep -oE '[0-9]+(\.[0-9]+)?' | head -n1)"
            if [ -n "$detected" ]; then
                source="nvidia-smi"
            fi
        fi
    fi

    echo "${detected}|${source}"
}

cuda_index_for_version() {
    local version="$1"
    local major="0"
    local minor="0"

    if [[ "$version" =~ ^([0-9]+)(\.([0-9]+))?$ ]]; then
        major="${BASH_REMATCH[1]}"
        if [ -n "${BASH_REMATCH[3]}" ]; then
            minor="${BASH_REMATCH[3]}"
        fi
    fi

    if [ "$major" -ge 13 ]; then
        echo "cu130"
    elif [ "$major" -eq 12 ] && [ "$minor" -ge 8 ]; then
        echo "cu128"
    elif [ "$major" -eq 12 ]; then
        echo "cu126"
    elif [ "$major" -eq 11 ] && [ "$minor" -ge 8 ]; then
        echo "cu118"
    else
        echo "cu126"
    fi
}

build_cuda_candidates() {
    local preferred="$1"
    local all_candidates=("$preferred" "cu130" "cu128" "cu126" "cu118")
    local item=""
    local result=()

    for item in "${all_candidates[@]}"; do
        if [[ " ${result[*]} " != *" $item "* ]]; then
            result+=("$item")
        fi
    done

    echo "${result[*]}"
}

echo ""
echo "============================================================"
echo "IMAGE INTERROGATOR - SETUP WIZARD"
echo "============================================================"
echo ""

# Check if Python is installed
echo -e "${BLUE}[1/8] Checking Python installation...${NC}"
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
echo -e "${BLUE}[2/8] Setting up virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo "      Creating new virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}      Virtual environment created successfully${NC}"
else
    echo "      Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo -e "${BLUE}[3/8] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}      Virtual environment activated${NC}"
echo ""

# Detect OS
OS_TYPE=$(uname -s)
echo -e "${BLUE}[4/8] Detecting GPU and acceleration support...${NC}"
echo ""

INSTALL_CUDA="n"
INSTALL_ROCM="n"
GPU_TYPE="cpu"
CUDA_VERSION="cpu"
DETECTED_CUDA="unknown"
CUDA_INDEX="cu126"
CUDA_DETECTION_SOURCE="unknown"
CUDA_CANDIDATES=()

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

        # Detect CUDA toolkit version (nvcc/version.json first, nvidia-smi fallback)
        CUDA_DETECTION_RESULT="$(detect_cuda_version)"
        DETECTED_CUDA="${CUDA_DETECTION_RESULT%%|*}"
        CUDA_DETECTION_SOURCE="${CUDA_DETECTION_RESULT##*|}"
        if [ -z "$DETECTED_CUDA" ]; then
            DETECTED_CUDA="unknown"
            CUDA_INDEX="cu130"
        else
            CUDA_INDEX="$(cuda_index_for_version "$DETECTED_CUDA")"
        fi
        read -r -a CUDA_CANDIDATES <<< "$(build_cuda_candidates "$CUDA_INDEX")"

        echo -e "      Detected CUDA Version: ${DETECTED_CUDA}"
        echo -e "      Detection Source: ${CUDA_DETECTION_SOURCE}"
        echo -e "      Recommended PyTorch CUDA Version: ${CUDA_INDEX}"
        echo -e "      Fallback PyTorch CUDA Versions: ${CUDA_CANDIDATES[*]}"
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
echo -e "${BLUE}[5/8] Checking PyTorch installation...${NC}"
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
            if [ "$INSTALL_CUDA" = "y" ]; then
                # Check whether installed torch CUDA build matches detected toolkit level
                CURRENT_CUDA=$(python3 -c "import re, torch; v=getattr(torch.version, 'cuda', None) or ''; m=re.match(r'^(\d+)(?:\.(\d+))?', v); major=int(m.group(1)) if m else 0; minor=int(m.group(2) or 0) if m else 0; print('cu130' if major >= 13 else 'cu128' if major == 12 and minor >= 8 else 'cu126' if major == 12 else 'cu118' if major == 11 and minor >= 8 else 'cpu')")

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
            else
                CURRENT_ROCM=$(python3 -c "import torch; print('rocm' if getattr(torch.version, 'hip', None) else 'unknown')")
                if [ "$CURRENT_ROCM" = "rocm" ]; then
                    echo -e "${GREEN}      PyTorch with ROCm support is correctly installed${NC}"
                    NEED_PYTORCH="n"
                else
                    echo -e "${YELLOW}      [WARNING] PyTorch is installed but ROCm metadata is missing.${NC}"
                    read -p "      Reinstall PyTorch with ROCm ${ROCM_VERSION} support? (y/n): " REINSTALL
                    if [ "$REINSTALL" = "y" ] || [ "$REINSTALL" = "Y" ]; then
                        NEED_PYTORCH="y"
                    else
                        NEED_PYTORCH="n"
                    fi
                fi
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
    echo -e "${BLUE}[6/8] Installing PyTorch...${NC}"
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
        if [ ${#CUDA_CANDIDATES[@]} -eq 0 ]; then
            read -r -a CUDA_CANDIDATES <<< "$(build_cuda_candidates "$CUDA_VERSION")"
        fi

        PYTORCH_INSTALLED=false
        for CANDIDATE_INDEX in "${CUDA_CANDIDATES[@]}"; do
            echo "      Trying PyTorch index: ${CANDIDATE_INDEX}"
            if pip install torch torchvision --index-url "https://download.pytorch.org/whl/${CANDIDATE_INDEX}"; then
                CUDA_VERSION="${CANDIDATE_INDEX}"
                PYTORCH_INSTALLED=true
                break
            fi
            echo -e "${YELLOW}      [WARNING] PyTorch install failed for ${CANDIDATE_INDEX}.${NC}"
        done

        if [ "$PYTORCH_INSTALLED" = false ]; then
            echo ""
            echo -e "${RED}      [ERROR] Failed to install any CUDA-enabled PyTorch build.${NC}"
            echo "      Tried indices: ${CUDA_CANDIDATES[*]}"
            echo "      On DGX Spark, verify network access and consider using NVIDIA NGC containers for pre-validated stacks."
            exit 1
        fi
        echo "      Installed PyTorch from index: ${CUDA_VERSION}"
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
    echo -e "${BLUE}[6/8] PyTorch installation...${NC}"
    echo "      Skipping PyTorch installation"
fi
echo ""

# Install other requirements
echo -e "${BLUE}[7/8] Installing remaining dependencies...${NC}"
echo ""

# Install ONNX Runtime based on GPU availability
ARCH=$(uname -m)
if [ "$INSTALL_CUDA" = "y" ]; then
    echo "      Installing ONNX Runtime with CUDA support for WD Tagger..."

    # Check if ARM64 - onnxruntime-gpu doesn't provide ARM wheels on PyPI
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        if python3 -c "import onnxruntime as ort; exit(0 if 'CUDAExecutionProvider' in ort.get_available_providers() else 1)" 2>/dev/null; then
            echo -e "${GREEN}      Existing ONNX Runtime CUDA provider detected on ARM64; keeping current install.${NC}"
        else
            echo -e "${YELLOW}      [INFO] ARM64 detected - onnxruntime-gpu wheels not available on PyPI${NC}"
            echo "      Installing CPU-only ONNX Runtime (WD Tagger will run on CPU)..."
            echo "      Tip: Run ./build_onnx_arm64.sh to build ONNX Runtime with GPU support."
            pip install "onnxruntime>=1.16.0" || echo -e "${YELLOW}      [WARNING] Failed to install ONNX Runtime. Continuing anyway...${NC}"
        fi
    else
        # For x86_64, try GPU version
        ONNX_INSTALLED=false
        if pip install "onnxruntime-gpu>=1.16.0" --extra-index-url https://pypi.nvidia.com 2>/dev/null; then
            ONNX_INSTALLED=true
        elif pip install "onnxruntime-gpu>=1.16.0" 2>/dev/null; then
            ONNX_INSTALLED=true
        fi

        if [ "$ONNX_INSTALLED" = false ]; then
            echo -e "${YELLOW}      [WARNING] Could not install ONNX Runtime GPU.${NC}"
            echo "      Falling back to CPU-only ONNX Runtime..."
            pip install "onnxruntime>=1.16.0" || echo -e "${YELLOW}      [WARNING] Failed to install ONNX Runtime. Continuing anyway...${NC}"
        fi
    fi
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

echo -e "${BLUE}[8/8] Provisioning llama.cpp llama-server...${NC}"
echo ""
if [ -x "$DEFAULT_LLAMA_BIN" ] && [ -z "${LLAMA_CPP_VERSION:-}" ]; then
    LLAMA_STATUS="existing"
    LLAMA_MESSAGE="Existing llama-server binary found; skipping download."
    if [ -f "$SCRIPT_DIR/cache/llama_cpp/bin/llama-server.version" ]; then
        LLAMA_VERSION="$(cat "$SCRIPT_DIR/cache/llama_cpp/bin/llama-server.version" 2>/dev/null || true)"
    fi
else
    LLAMA_ARGS=(--cache-dir "$SCRIPT_DIR/cache/llama_cpp")
    if [ "$INSTALL_CUDA" = "y" ]; then
        LLAMA_ARGS+=(--prefer-cuda)
        if [ -n "${DETECTED_CUDA:-}" ]; then
            LLAMA_ARGS+=(--cuda-version "${DETECTED_CUDA}")
        fi
    fi
    if [ "$INSTALL_ROCM" = "y" ]; then
        LLAMA_ARGS+=(--prefer-rocm)
    fi
    if [ -n "${LLAMA_CPP_VERSION:-}" ]; then
        LLAMA_ARGS+=(--tag "${LLAMA_CPP_VERSION}")
    fi

    LLAMA_OUTPUT="$(python3 "$SCRIPT_DIR/provision_llama_server.py" "${LLAMA_ARGS[@]}")"
    while IFS='=' read -r key value; do
        case "$key" in
            LLAMA_STATUS) LLAMA_STATUS="$value" ;;
            LLAMA_BINARY_PATH) LLAMA_BINARY_PATH="$value" ;;
            LLAMA_VERSION) LLAMA_VERSION="$value" ;;
            LLAMA_MESSAGE) LLAMA_MESSAGE="$value" ;;
        esac
    done <<< "$LLAMA_OUTPUT"
fi

if [ "$LLAMA_STATUS" = "installed" ]; then
    echo -e "${GREEN}      llama-server installed: $LLAMA_BINARY_PATH${NC}"
elif [ "$LLAMA_STATUS" = "existing" ]; then
    echo -e "${GREEN}      llama-server ready: $LLAMA_BINARY_PATH${NC}"
else
    echo -e "${YELLOW}      [WARNING] Failed to provision llama-server automatically.${NC}"
    if [ -n "$LLAMA_MESSAGE" ]; then
        echo -e "${YELLOW}      $LLAMA_MESSAGE${NC}"
    fi
    echo "      Manual fallback: https://github.com/ggml-org/llama.cpp/releases"
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        echo "      Tip: Run ./build_llama_cpp_arm64.sh to compile llama.cpp with CUDA from source."
    fi
fi
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
print(f'  TensorRT Provider: {"Yes" if "TensorrtExecutionProvider" in providers else "No (install TensorRT for faster inference)"}')
EOF
echo ""
echo "Critical Dependencies:"
python3 -c "import PyQt6; print('  PyQt6: Installed')" 2>/dev/null || echo -e "${RED}  [ERROR] PyQt6: NOT INSTALLED${NC}"
python3 -c "from clip_interrogator import Interrogator; print('  CLIP Interrogator: Installed')" 2>/dev/null || echo -e "${RED}  [ERROR] CLIP Interrogator: NOT INSTALLED${NC}"
python3 -c "import PIL; print('  Pillow: Installed')" 2>/dev/null || echo -e "${RED}  [ERROR] Pillow: NOT INSTALLED${NC}"
echo ""
echo "llama.cpp runtime:"
if [ -x "$LLAMA_BINARY_PATH" ]; then
    echo "  Binary: $LLAMA_BINARY_PATH"
    if [ -n "$LLAMA_VERSION" ]; then
        echo "  Release: $LLAMA_VERSION"
    fi
else
    echo "  Binary: NOT INSTALLED (set manually in LlamaCpp MM config)"
    if [ -n "$LLAMA_MESSAGE" ]; then
        echo "  Note: $LLAMA_MESSAGE"
    fi
fi
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
    fi

    # Handle ARM64 vs x86_64 differently for success messages
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        if [ "$PYTORCH_OK" = true ] && [ "$ONNX_OK" = true ]; then
            echo -e "${GREEN}[SUCCESS] GPU acceleration is fully enabled!${NC}"
            echo "  - PyTorch: CUDA enabled (for CLIP models)"
            echo "  - ONNX Runtime: CUDA enabled (for WD Tagger models)"
            echo ""
        elif [ "$PYTORCH_OK" = true ]; then
            echo -e "${GREEN}[SUCCESS] GPU acceleration is enabled!${NC}"
            echo "  - PyTorch: CUDA enabled (for CLIP models)"
            echo "  - ONNX Runtime: CPU mode (no ARM64 GPU wheels on PyPI)"
            echo ""
            echo -e "${YELLOW}  Tip: Run ./build_onnx_arm64.sh to build ONNX Runtime with GPU support${NC}"
            echo ""
        fi
    else
        if [ "$ONNX_OK" = false ]; then
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
fi

echo "============================================================"
echo "SETUP COMPLETE!"
echo "============================================================"
echo ""
echo "You can now run the application with: ./run.sh"
echo ""
