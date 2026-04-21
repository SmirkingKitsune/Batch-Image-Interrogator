#!/usr/bin/env python3
"""
Image Interrogator - Batch Image Tagging Tool

A PyQt6-based application for batch image interrogation and tagging using
CLIP and Waifu Diffusion models with SQLite caching.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    """Main application entry point."""
    # CRITICAL: Detect devices BEFORE Qt initialization
    # This allows PyTorch to claim CUDA before Qt touches GPU resources
    print("Detecting compute devices...")
    from core.device_detector import detect_devices_early
    device_status = detect_devices_early()

    # Log detection results to console (use ASCII-safe characters for Windows)
    if device_status['pytorch_cuda_available']:
        print(f"[OK] PyTorch CUDA available - GPU acceleration enabled")
    else:
        print(f"[WARN] PyTorch CUDA not available - using CPU")
        if device_status['pytorch_error']:
            print(f"  Reason: {device_status['pytorch_error']}")

    if device_status['onnx_cuda_available']:
        print(f"[OK] ONNX Runtime CUDA available - GPU acceleration enabled")
    else:
        print(f"[WARN] ONNX Runtime CUDA not available - using CPU")
        if device_status['onnx_error']:
            print(f"  Reason: {device_status['onnx_error']}")

    print()  # Blank line for readability

    # NOW safe to import and initialize Qt (PyTorch already claimed CUDA)
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import Qt, QTimer
    from ui import MainWindow

    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Image Interrogator")
    app.setOrganizationName("ImageInterrogator")

    # Create main window (pass device status for config defaults)
    window = MainWindow(device_status=device_status)
    window.show()

    # Defer model list population until event loop starts
    # This ensures window is fully rendered before heavy operations
    QTimer.singleShot(100, window.populate_model_lists)

    # Run event loop
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
