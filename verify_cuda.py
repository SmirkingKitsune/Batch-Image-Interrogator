"""
Quick CUDA Verification Script
Tests that PyTorch can access the GPU
"""
import torch

print("=" * 60)
print("PYTORCH CUDA VERIFICATION")
print("=" * 60)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

    # Test GPU computation
    print("\nTesting GPU computation...")
    x = torch.rand(5, 3).cuda()
    print(f"Created tensor on GPU: {x.device}")
    y = x * 2
    print(f"Computation successful!")
    print(f"Result:\n{y}")

    print("\n" + "=" * 60)
    print("SUCCESS! CUDA is working correctly!")
    print("=" * 60)
else:
    print("\nERROR: CUDA is still not available!")
    print("Please check:")
    print("1. NVIDIA drivers are installed")
    print("2. PyTorch was installed with correct CUDA version")
    print("=" * 60)
