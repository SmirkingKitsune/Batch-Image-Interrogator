"""
CUDA Diagnostic Script
Checks PyTorch installation and CUDA availability
"""
import sys

print("=" * 60)
print("CUDA DIAGNOSTIC REPORT")
print("=" * 60)

# Check Python version
print(f"\n1. Python Version: {sys.version}")

# Check PyTorch installation
try:
    import torch
    print(f"\n2. PyTorch Version: {torch.__version__}")
    print(f"   PyTorch Install Path: {torch.__file__}")

    # Check if PyTorch was built with CUDA
    print(f"\n3. PyTorch Built with CUDA: {torch.cuda.is_available()}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   CUDA Version (PyTorch): {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
        print(f"   Number of CUDA Devices: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\n   Device {i}: {torch.cuda.get_device_name(i)}")
            print(f"   - Compute Capability: {torch.cuda.get_device_capability(i)}")
            print(f"   - Total Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        print("\n   ⚠️  CUDA is NOT available!")
        print("   This usually means:")
        print("   - PyTorch was installed without CUDA support (CPU-only version)")
        print("   - Or NVIDIA drivers are not properly installed")

        # Check if this is CPU-only build
        if hasattr(torch.version, 'cuda') and torch.version.cuda is None:
            print("\n   ❌ This is a CPU-ONLY build of PyTorch!")
            print("   You need to reinstall PyTorch with CUDA support.")

except ImportError:
    print("\n2. ❌ PyTorch is NOT installed!")
    sys.exit(1)

# Check torchvision
try:
    import torchvision
    print(f"\n4. TorchVision Version: {torchvision.__version__}")
except ImportError:
    print("\n4. ❌ TorchVision is NOT installed!")

# Check NVIDIA driver
print("\n5. Checking NVIDIA Driver...")
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   ✓ NVIDIA Driver is installed")
        # Extract driver version and CUDA version from nvidia-smi
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Driver Version' in line or 'CUDA Version' in line:
                print(f"   {line.strip()}")
    else:
        print("   ❌ nvidia-smi failed to run")
except FileNotFoundError:
    print("   ❌ nvidia-smi not found - NVIDIA drivers may not be installed")
except Exception as e:
    print(f"   ❌ Error running nvidia-smi: {e}")

# Check CUDA in PATH
import os
print("\n6. CUDA in System PATH:")
cuda_paths = [p for p in os.environ.get('PATH', '').split(os.pathsep) if 'cuda' in p.lower()]
if cuda_paths:
    print("   ✓ CUDA paths found:")
    for path in cuda_paths:
        print(f"   - {path}")
else:
    print("   ⚠️  No CUDA paths found in PATH")

print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

if not torch.cuda.is_available():
    print("\nTo fix CUDA support, you need to:")
    print("1. Uninstall current PyTorch:")
    print("   pip uninstall torch torchvision torchaudio")
    print("\n2. Reinstall with CUDA support:")
    print("   For CUDA 12.6:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
    print("\n   For CUDA 12.8:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128")
    print("\n   For CUDA 13.0:")
    print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130")
else:
    print("\n✓ PyTorch with CUDA is properly configured!")

print("\n" + "=" * 60)
