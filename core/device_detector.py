"""
Device detection utility for PyTorch and ONNX Runtime.

CRITICAL: This module must be imported and initialized BEFORE any Qt imports
to ensure PyTorch can properly initialize CUDA.
"""

import logging
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


class DeviceType(Enum):
    """Available device types."""
    CUDA = "cuda"
    CPU = "cpu"


class DeviceDetector:
    """
    Detects available compute devices for PyTorch and ONNX Runtime.

    This class performs detection once and caches results to avoid
    repeated import overhead.
    """

    def __init__(self):
        self._pytorch_cuda_available: Optional[bool] = None
        self._pytorch_device: Optional[str] = None
        self._onnx_cuda_available: Optional[bool] = None
        self._detection_performed = False
        self._pytorch_error: Optional[str] = None
        self._onnx_error: Optional[str] = None

    def detect_all(self) -> Dict:
        """
        Perform full device detection for both PyTorch and ONNX Runtime.

        MUST be called before Qt initialization to prevent CUDA conflicts.

        Returns:
            Dict with detection results and recommended device
        """
        if self._detection_performed:
            return self.get_status()

        self._detect_pytorch()
        self._detect_onnx()
        self._detection_performed = True

        return self.get_status()

    def _detect_pytorch(self):
        """Detect PyTorch CUDA availability."""
        try:
            import torch
            self._pytorch_cuda_available = torch.cuda.is_available()

            if self._pytorch_cuda_available:
                self._pytorch_device = "cuda"
                # Log GPU info
                gpu_name = torch.cuda.get_device_name(0)
                logger.info(f"PyTorch CUDA available - GPU: {gpu_name}")
            else:
                self._pytorch_device = "cpu"
                logger.info("PyTorch CUDA not available - using CPU")

        except ImportError:
            self._pytorch_cuda_available = False
            self._pytorch_device = "cpu"
            self._pytorch_error = "PyTorch not installed"
            logger.warning("PyTorch not installed")
        except Exception as e:
            self._pytorch_cuda_available = False
            self._pytorch_device = "cpu"
            self._pytorch_error = str(e)
            logger.error(f"Error detecting PyTorch CUDA: {e}")

    def _detect_onnx(self):
        """Detect ONNX Runtime CUDA availability."""
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            self._onnx_cuda_available = 'CUDAExecutionProvider' in providers

            if self._onnx_cuda_available:
                logger.info("ONNX Runtime CUDA available")
            else:
                logger.info("ONNX Runtime CUDA not available")

        except ImportError:
            self._onnx_cuda_available = False
            self._onnx_error = "ONNX Runtime not installed"
            logger.warning("ONNX Runtime not installed")
        except Exception as e:
            self._onnx_cuda_available = False
            self._onnx_error = str(e)
            logger.error(f"Error detecting ONNX CUDA: {e}")

    def get_pytorch_device(self) -> str:
        """
        Get recommended PyTorch device.

        Returns:
            'cuda' if available, otherwise 'cpu'
        """
        if not self._detection_performed:
            self.detect_all()
        return self._pytorch_device or "cpu"

    def get_onnx_device(self) -> str:
        """
        Get recommended ONNX Runtime device.

        Returns:
            'cuda' if available, otherwise 'cpu'
        """
        if not self._detection_performed:
            self.detect_all()
        return "cuda" if self._onnx_cuda_available else "cpu"

    def is_pytorch_cuda_available(self) -> bool:
        """Check if PyTorch CUDA is available."""
        if not self._detection_performed:
            self.detect_all()
        return self._pytorch_cuda_available or False

    def is_onnx_cuda_available(self) -> bool:
        """Check if ONNX Runtime CUDA is available."""
        if not self._detection_performed:
            self.detect_all()
        return self._onnx_cuda_available or False

    def get_status(self) -> Dict:
        """
        Get comprehensive device status.

        Returns:
            Dict containing all detection results
        """
        return {
            'pytorch_cuda_available': self._pytorch_cuda_available,
            'pytorch_device': self._pytorch_device,
            'pytorch_error': self._pytorch_error,
            'onnx_cuda_available': self._onnx_cuda_available,
            'onnx_error': self._onnx_error,
            'any_cuda_available': (
                self._pytorch_cuda_available or self._onnx_cuda_available
            ),
            'detection_performed': self._detection_performed
        }


# Global singleton instance
_device_detector = None


def get_device_detector() -> DeviceDetector:
    """Get or create the global device detector instance."""
    global _device_detector
    if _device_detector is None:
        _device_detector = DeviceDetector()
    return _device_detector


def detect_devices_early() -> Dict:
    """
    Perform early device detection before Qt initialization.

    This function MUST be called in main.py before creating QApplication.

    Returns:
        Device detection status dict
    """
    detector = get_device_detector()
    return detector.detect_all()
