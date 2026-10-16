"""
Device detection utility for PyTorch and ONNX Runtime.

CRITICAL: This module must be imported and initialized BEFORE any Qt imports
to ensure PyTorch can properly initialize CUDA.
"""

import logging
import os
import shutil
import sys
import sysconfig
from typing import Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)


def _triton_build_blocker() -> Optional[str]:
    """
    Say why Triton cannot compile its CUDA launcher here, or None if it can.

    Triton builds a small C extension before its first kernel launch, which
    needs a C compiler and the Python development headers. The lookups
    mirror triton.runtime.build._build.
    """
    scheme = sysconfig.get_default_scheme()
    if scheme == 'posix_local':  # Debian's scheme; Triton swaps it the same way
        scheme = 'posix_prefix'
    include_dir = sysconfig.get_paths(scheme=scheme)['include']
    if not os.path.exists(os.path.join(include_dir, 'Python.h')):
        package = f"python{sys.version_info.major}.{sys.version_info.minor}-dev"
        return f"Python.h is missing from {include_dir} (install {package})"
    if not (os.environ.get('CC') or shutil.which('gcc') or shutil.which('clang')):
        return "no C compiler was found (install gcc)"
    return None


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
                self._avoid_unbuildable_triton()
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

    def _avoid_unbuildable_triton(self):
        """
        Send PyTorch's Triton-backed CUDA ops back to its stock kernels when
        Triton cannot build on this machine.

        PyTorch 2.14+ routes some eager ops through Triton -- bmm on batched
        outer products is one, and BLIP-2's T5 decoder reaches it via
        scaled_dot_product_attention. Without a compiler or Python.h every
        such call raises, failing CLIP interrogation at its caption step on
        every image.
        """
        reason = _triton_build_blocker()
        if reason is None:
            return
        try:
            from torch._native import registry, triton_utils
            if not registry.get_dsl_operations("triton"):
                return
            triton_utils.deregister_op_overrides()
        except (ImportError, AttributeError):
            return  # this PyTorch doesn't route eager ops through Triton
        except Exception as e:
            # Best effort: never let this cost the CUDA detection itself.
            logger.warning(f"Could not turn off Triton-backed PyTorch ops ({reason}): {e}")
            return
        logger.warning(
            f"Triton can't build its CUDA launcher: {reason}. "
            "PyTorch will use its standard CUDA kernels instead."
        )

    def _detect_onnx(self):
        """Detect ONNX Runtime CUDA availability."""
        try:
            import onnxruntime as ort

            # The pip nvidia-* packages ship only SONAME-versioned libraries
            # (libcudnn.so.9), but ONNX Runtime dlopens the unversioned
            # libcudnn.so. Without a preload the CUDA provider still *lists* as
            # available and then fails at the first Conv node with
            # "cuDNN is unavailable or disabled". preload_dlls() resolves the
            # libraries out of site-packages up front. This process is
            # single-threaded Qt (all workers are QThreads), so preloading once
            # here covers every session created later.
            if hasattr(ort, 'preload_dlls'):
                try:
                    ort.preload_dlls()
                except Exception as e:
                    logger.warning(f"ONNX Runtime dependency preload failed: {e}")

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
