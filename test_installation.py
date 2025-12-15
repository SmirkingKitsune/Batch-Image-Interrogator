#!/usr/bin/env python3
"""
Test script to verify Image Interrogator installation.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    try:
        from core import InterrogationDatabase, FileManager, hash_image_content
        print("  ✓ Core modules")
    except ImportError as e:
        print(f"  ✗ Core modules: {e}")
        return False
    
    try:
        from interrogators import CLIPInterrogator, WDInterrogator
        print("  ✓ Interrogators")
    except ImportError as e:
        print(f"  ✗ Interrogators: {e}")
        return False
    
    try:
        from ui import MainWindow
        print("  ✓ UI components")
    except ImportError as e:
        print(f"  ✗ UI components: {e}")
        return False
    
    return True


def test_dependencies():
    """Test that required packages are installed."""
    print("\nTesting dependencies...")
    
    packages = [
        ('PyQt6', 'PyQt6.QtWidgets'),
        ('Pillow', 'PIL'),
        ('NumPy', 'numpy'),
        ('ONNX Runtime', 'onnxruntime'),
        ('HuggingFace Hub', 'huggingface_hub'),
        ('Pandas', 'pandas'),
    ]
    
    all_installed = True
    for name, module in packages:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - Not installed")
            all_installed = False
    
    # Optional dependencies
    print("\nOptional dependencies:")
    
    optional = [
        ('CLIP Interrogator', 'clip_interrogator'),
        ('PyTorch', 'torch'),
        ('TorchVision', 'torchvision'),
    ]
    
    for name, module in optional:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ○ {name} - Not installed (optional)")
    
    return all_installed


def test_database():
    """Test database creation."""
    print("\nTesting database...")
    
    try:
        from core import InterrogationDatabase
        
        db_path = Path(__file__).parent / "test_db.db"
        db = InterrogationDatabase(str(db_path))
        
        # Test basic operations
        model_id = db.register_model("test_model", "TEST", version="1.0")
        stats = db.get_statistics()
        
        db.close()
        
        # Cleanup
        db_path.unlink()
        
        print("  ✓ Database operations")
        return True
        
    except Exception as e:
        print(f"  ✗ Database operations: {e}")
        return False


def test_gpu_availability():
    """Test GPU availability."""
    print("\nChecking GPU availability...")
    
    # CUDA for PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ○ CUDA not available (CPU mode will be used)")
    except ImportError:
        print("  ○ PyTorch not installed")
    
    # ONNX Runtime
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            print("  ✓ ONNX Runtime GPU support")
        else:
            print("  ○ ONNX Runtime CPU only")
    except ImportError:
        print("  ○ ONNX Runtime not installed")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Image Interrogator - Installation Test")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("Dependencies", test_dependencies()))
    results.append(("Database", test_database()))
    
    # GPU check (informational)
    test_gpu_availability()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary:")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! Installation is complete.")
        print("\nYou can now run: python main.py")
    else:
        print("Some tests failed. Please install missing dependencies:")
        print("  pip install -r requirements.txt")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
