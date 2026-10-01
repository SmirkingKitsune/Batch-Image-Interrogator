"""Hardware facts for the always-on status chrome.

The design keeps device, GPU memory and model state visible at all times, so
the bridge samples GPU memory once a second while a window is connected.
nvidia-smi is preferred because it needs no CUDA context in this process.
Unified-memory GPUs (for example the GB10 in a DGX Spark) report their memory
as "[N/A]"; there the pool is system memory, so /proc/meminfo is reported and
flagged as unified.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

NVIDIA_SMI_TIMEOUT = 4.0


def _run(argv, timeout: float = NVIDIA_SMI_TIMEOUT) -> str:
    try:
        result = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout if result.returncode == 0 else ""


def _meminfo() -> Optional[Dict[str, int]]:
    """Total and used system memory in bytes, from /proc/meminfo."""
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return None
    values: Dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            values[key.strip()] = int(parts[0]) * 1024
    total = values.get("MemTotal")
    available = values.get("MemAvailable")
    if not total or available is None:
        return None
    return {"total": total, "used": max(0, total - available)}


def _parse_number(value: str) -> Optional[float]:
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return None  # "[N/A]" on unified-memory GPUs


class HardwareMonitor:
    """Static device description plus a cheap live memory sample."""

    def __init__(self, device_status: Optional[Dict[str, Any]] = None):
        self.device_status = dict(device_status or {})
        self._has_nvidia_smi = shutil.which("nvidia-smi") is not None
        self._static: Optional[Dict[str, Any]] = None

    def static_info(self) -> Dict[str, Any]:
        """Versions and device identity. Computed once; none of it changes."""
        if self._static is not None:
            return self._static

        info: Dict[str, Any] = {
            "platform": platform.system().lower(),
            "arch": platform.machine().lower(),
            "gpu_name": "",
            "driver": "",
            "cuda_version": "",
            "torch_version": "",
            "torch_cuda": bool(self.device_status.get("pytorch_cuda_available")),
            "torch_error": self.device_status.get("pytorch_error") or "",
            "onnx_version": "",
            "onnx_providers": [],
            "onnx_cuda": bool(self.device_status.get("onnx_cuda_available")),
            "onnx_error": self.device_status.get("onnx_error") or "",
            "tensorrt_engines": 0,
        }

        try:
            import torch

            info["torch_version"] = torch.__version__
            info["cuda_version"] = torch.version.cuda or ""
            if info["torch_cuda"]:
                info["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass

        try:
            import onnxruntime as ort

            info["onnx_version"] = ort.__version__
            info["onnx_providers"] = list(ort.get_available_providers())
        except Exception:
            pass

        if self._has_nvidia_smi:
            line = _run([
                "nvidia-smi",
                "--query-gpu=name,driver_version",
                "--format=csv,noheader,nounits",
            ]).strip().splitlines()
            if line:
                fields = [field.strip() for field in line[0].split(",")]
                if fields and not info["gpu_name"]:
                    info["gpu_name"] = fields[0]
                if len(fields) > 1:
                    info["driver"] = fields[1]

        try:
            from core.model_cache import TRT_CACHE_DIR

            if TRT_CACHE_DIR.exists():
                info["tensorrt_engines"] = sum(1 for _ in TRT_CACHE_DIR.iterdir())
        except Exception:
            pass

        self._static = info
        return info

    def sample(self) -> Dict[str, Any]:
        """Current GPU memory use. Keys: used, total (bytes), unified, util."""
        sample: Dict[str, Any] = {"used": None, "total": None, "unified": False, "util": None}
        if self._has_nvidia_smi:
            line = _run([
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ]).strip().splitlines()
            if line:
                fields = line[0].split(",")
                used = _parse_number(fields[0]) if fields else None
                total = _parse_number(fields[1]) if len(fields) > 1 else None
                util = _parse_number(fields[2]) if len(fields) > 2 else None
                sample["util"] = util
                if used is not None and total:
                    sample["used"] = int(used * 1024 * 1024)
                    sample["total"] = int(total * 1024 * 1024)
                    return sample
                # A GPU that reports no dedicated memory shares system RAM.
                sample["unified"] = True

        if sample["unified"]:
            memory = _meminfo()
            if memory:
                sample["used"] = memory["used"]
                sample["total"] = memory["total"]
        return sample
