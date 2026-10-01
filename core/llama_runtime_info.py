"""What the managed llama.cpp runtime actually is, for display.

Both front ends need to answer "what engine am I actually running?": the
PyQt6 runtime card (ui/llama_runtime.py) and the Electron bridge. The answer
comes from provisioner state rather than a configured path. A path tells you
where a file is, while `active-runtime.json` tells you whether it is a CUDA
build or the CPU fallback the ladder settled for.

Nothing here imports Qt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from core.llama_provisioner import (
    ProvisionConfig,
    ProvisionError,
    detect_accelerator,
    find_managed_executable,
    read_active_runtime,
)

DEFAULT_PROVISION_DIR = Path(__file__).resolve().parents[1] / "cache" / "llama_cpp"

RUNTIME_MODE_MANAGED = "managed"
RUNTIME_MODE_CUSTOM = "custom"

# Backends that actually use the GPU. Anything else running while one of these
# was asked for means the acquisition ladder fell back.
GPU_ACCELERATORS = ("cuda", "rocm", "vulkan", "metal", "sycl-fp32", "sycl-fp16")


def managed_config(provision_dir: Optional[Path] = None) -> ProvisionConfig:
    """Config describing what this host should be running."""
    return ProvisionConfig(provision_dir=Path(provision_dir or DEFAULT_PROVISION_DIR))


def managed_executable(provision_dir: Optional[Path] = None) -> Optional[Path]:
    """Path to the installed managed runtime, or None when absent."""
    try:
        return find_managed_executable(managed_config(provision_dir).normalized())
    except (ProvisionError, OSError):
        return None


def runtime_mode(llama_config: Dict[str, Any]) -> str:
    """Managed or custom, defaulting to managed for a fresh configuration.

    A saved custom path from before this setting existed keeps working: it is
    read as custom mode rather than being silently ignored.
    """
    mode = str(llama_config.get("llama_runtime_mode", "") or "").strip().lower()
    if mode in (RUNTIME_MODE_MANAGED, RUNTIME_MODE_CUSTOM):
        return mode
    saved_path = str(llama_config.get("llama_binary_path", "") or "").strip()
    if not saved_path:
        return RUNTIME_MODE_MANAGED
    managed = managed_executable()
    if managed is not None and Path(saved_path) == managed:
        return RUNTIME_MODE_MANAGED
    return RUNTIME_MODE_CUSTOM


def resolve_runtime_binary(
    llama_config: Dict[str, Any],
    provision_dir: Optional[Path] = None,
) -> str:
    """The llama-server path to launch for this configuration.

    Managed mode resolves from disk on every call rather than trusting a stored
    absolute path, so reprovisioning to a different backend takes effect without
    the user re-picking anything.
    """
    if runtime_mode(llama_config) == RUNTIME_MODE_CUSTOM:
        return str(llama_config.get("llama_binary_path", "") or "").strip()
    managed = managed_executable(provision_dir)
    return str(managed) if managed else ""


def summarize_runtime(
    provision_dir: Optional[Path] = None,
    detect: Callable[[], str] = detect_accelerator,
) -> Dict[str, Any]:
    """Describe the installed managed runtime for display.

    Keys: installed, executable, version, method, accelerator, cuda_architectures,
    installed_at, is_gpu, fallback_from, ladder.

    `detect` reports what the hardware supports; it is a parameter so callers
    can substitute their own probe.
    """
    executable = managed_executable(provision_dir)
    active = read_active_runtime(Path(provision_dir or DEFAULT_PROVISION_DIR))
    accelerator = str(active.get("accelerator", "") or "")
    ladder = active.get("ladder")

    summary: Dict[str, Any] = {
        "installed": executable is not None,
        "executable": str(executable) if executable else "",
        "version": str(active.get("version", "") or ""),
        "method": str(active.get("method", "") or ""),
        "accelerator": accelerator,
        "cuda_architectures": str(active.get("cuda_architectures", "") or ""),
        "installed_at": str(active.get("installed_at", "") or ""),
        "is_gpu": accelerator in GPU_ACCELERATORS,
        "fallback_from": "",
        "ladder": list(ladder) if isinstance(ladder, list) else [],
    }

    # A CPU runtime on a machine whose hardware supports something better is the
    # ladder's last rung, not a choice. Say so, permanently — the provisioning
    # dialog's one-time warning is gone by the time a slow batch is running.
    if summary["installed"] and accelerator and not summary["is_gpu"]:
        detected = detect()
        if detected in GPU_ACCELERATORS:
            summary["fallback_from"] = detected
    return summary


def short_version(version: str) -> str:
    """Just the build identity.

    The provisioner records `b10828 (source, cuda, cuda-arch=121a-real)`, and
    the runtime card already shows method and architecture in their own fields,
    so the parenthetical would repeat both back a second and third time.
    """
    return version.split(" (", 1)[0].strip() or version


def short_timestamp(value: str) -> str:
    """ISO timestamp trimmed to minutes, without the T separator."""
    trimmed = value.replace("T", " ").strip()
    parts = trimmed.split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else trimmed
