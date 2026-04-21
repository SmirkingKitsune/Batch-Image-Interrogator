#!/usr/bin/env python3
"""Provision llama-server from official ggml-org/llama.cpp GitHub releases."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_BASE = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
USER_AGENT = "batch-image-interrogator-llama-setup/1.0"
CUDA_HINTS = ("cuda", "cublas", "cu11", "cu12", "cu13")
ROCM_HINTS = ("rocm", "hip", "hip-radeon")
BLOCK_HINTS = ("sha256", "checksum", ".txt", ".sig", "source", "src", "xcframework")
ARCH_X64_HINTS = ("x86_64", "amd64", "x64", "win64")
ARCH_X86_HINTS = ("x86", "i386", "i686")
ARCH_ARM64_HINTS = ("arm64", "aarch64")
CUDA_VER_RE = re.compile(r"cuda-([0-9]+(?:\.[0-9]+)?)")
ROCM_VER_RE = re.compile(r"rocm[-_]?([0-9]+(?:\.[0-9]+)?)")
GENERIC_VER_RE = re.compile(r"([0-9]+(?:\.[0-9]+)?)")


def _emit(status: str, binary_path: str = "", version: str = "", message: str = "") -> None:
    clean_message = " ".join(str(message).splitlines()).strip()
    print(f"LLAMA_STATUS={status}")
    print(f"LLAMA_BINARY_PATH={binary_path}")
    print(f"LLAMA_VERSION={version}")
    print(f"LLAMA_MESSAGE={clean_message}")


def _normalize_os() -> str:
    system = platform.system().lower()
    if system.startswith("win"):
        return "windows"
    if system == "darwin":
        return "macos"
    if system == "linux":
        return "linux"
    return system


def _normalize_arch() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64", "x64"):
        return "x64"
    if machine in ("arm64", "aarch64"):
        return "arm64"
    return machine


def _os_hints(os_id: str) -> Tuple[str, ...]:
    if os_id == "windows":
        return ("windows", "win")
    if os_id == "linux":
        return ("linux", "ubuntu")
    if os_id == "macos":
        return ("macos", "darwin", "osx", "mac", "metal")
    return (os_id,)


def _is_archive(name: str) -> bool:
    lower = name.lower()
    return lower.endswith(".zip") or lower.endswith(".tar.gz") or lower.endswith(".tgz") or lower.endswith(".tar.xz")


def _has_any(text: str, hints: Iterable[str]) -> bool:
    return any(hint in text for hint in hints)


def _is_arch_compatible(name: str, arch_id: str) -> bool:
    lower = name.lower()
    has_x64 = _has_any(lower, ARCH_X64_HINTS)
    has_x86 = _has_any(lower, ARCH_X86_HINTS)
    has_arm64 = _has_any(lower, ARCH_ARM64_HINTS)
    if arch_id == "x64" and has_x86 and not has_x64:
        return False
    if arch_id == "x64" and has_arm64 and not has_x64:
        return False
    if arch_id == "x86" and has_x64 and not has_x86:
        return False
    if arch_id == "arm64" and has_x64 and not has_arm64:
        return False
    if arch_id == "arm64" and has_x86 and not has_arm64:
        return False
    return True


def _runtime_flavor(name: str) -> str:
    lower = name.lower()
    if _has_any(lower, CUDA_HINTS):
        return "cuda"
    if _has_any(lower, ROCM_HINTS):
        return "rocm"
    if "vulkan" in lower:
        return "vulkan"
    if "opencl" in lower:
        return "opencl"
    if "sycl" in lower or "openvino" in lower:
        return "sycl"
    if "cpu" in lower:
        return "cpu"
    return "generic"


def _parse_cuda_version(text: str) -> Optional[float]:
    if not text:
        return None
    match = GENERIC_VER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_cuda_version_from_asset(name: str) -> Optional[float]:
    match = CUDA_VER_RE.search(name.lower())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_rocm_version(text: str) -> Optional[float]:
    if not text:
        return None
    match = GENERIC_VER_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_rocm_version_from_asset(name: str) -> Optional[float]:
    match = ROCM_VER_RE.search(name.lower())
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _parse_version_list(text: str) -> List[float]:
    if not text:
        return []
    versions: List[float] = []
    for token in re.split(r"[\s,;|]+", text.strip()):
        if not token:
            continue
        parsed = _parse_cuda_version(token)
        if parsed is not None:
            versions.append(parsed)
    return versions


def _dedupe_sort_versions(versions: Sequence[float]) -> List[float]:
    unique = {round(float(version), 2) for version in versions}
    return sorted(unique, reverse=True)


def _run_command_text(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=8,
            check=False,
            text=True,
        )
    except Exception:
        return ""
    output = f"{completed.stdout}\n{completed.stderr}".strip()
    return output


def _detect_cuda_versions() -> List[float]:
    detected: List[float] = []

    for env_name in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        env_value = os.environ.get(env_name, "")
        parsed = _parse_cuda_version(env_value)
        if parsed is not None:
            detected.append(parsed)

    if platform.system().lower().startswith("win"):
        toolkit_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        if toolkit_root.exists():
            for child in toolkit_root.iterdir():
                if child.is_dir():
                    parsed = _parse_cuda_version(child.name)
                    if parsed is not None:
                        detected.append(parsed)
    else:
        for root_dir, pattern in (("/usr/local", "cuda-*"), ("/opt", "cuda-*")):
            base = Path(root_dir)
            if not base.exists():
                continue
            for path in base.glob(pattern):
                parsed = _parse_cuda_version(path.name)
                if parsed is not None:
                    detected.append(parsed)

    nvcc_text = _run_command_text(("nvcc", "--version"))
    if nvcc_text:
        release_match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)?)", nvcc_text, re.IGNORECASE)
        if release_match:
            parsed = _parse_cuda_version(release_match.group(1))
            if parsed is not None:
                detected.append(parsed)

    # `nvidia-smi` reports driver CUDA capability, not installed toolkit versions.
    # Use it only as a fallback when toolkit-oriented probes found nothing.
    if not detected:
        nvidia_smi_text = _run_command_text(("nvidia-smi",))
        if nvidia_smi_text:
            smi_match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", nvidia_smi_text, re.IGNORECASE)
            if smi_match:
                parsed = _parse_cuda_version(smi_match.group(1))
                if parsed is not None:
                    detected.append(parsed)

    return _dedupe_sort_versions(detected)


def _detect_rocm_versions() -> List[float]:
    detected: List[float] = []

    for env_name in ("ROCM_PATH", "ROCM_HOME"):
        env_value = os.environ.get(env_name, "")
        parsed = _parse_rocm_version(env_value)
        if parsed is not None:
            detected.append(parsed)

    rocm_info_file = Path("/opt/rocm/.info/version")
    if rocm_info_file.exists():
        parsed = _parse_rocm_version(_read_text(rocm_info_file))
        if parsed is not None:
            detected.append(parsed)

    rocminfo_text = _run_command_text(("rocminfo",))
    if rocminfo_text:
        rocm_match = re.search(r"ROCm Version:\s*([0-9]+(?:\.[0-9]+)?)", rocminfo_text, re.IGNORECASE)
        if rocm_match:
            parsed = _parse_rocm_version(rocm_match.group(1))
            if parsed is not None:
                detected.append(parsed)

    hipcc_text = _run_command_text(("hipcc", "--version"))
    if hipcc_text:
        hip_match = re.search(r"HIP version:\s*([0-9]+(?:\.[0-9]+)?)", hipcc_text, re.IGNORECASE)
        if hip_match:
            parsed = _parse_rocm_version(hip_match.group(1))
            if parsed is not None:
                detected.append(parsed)

    return _dedupe_sort_versions(detected)


def _pick_highest_exact_then_lower(
    candidates: Sequence[Dict[str, str]],
    candidate_version_key: str,
    installed_versions: Sequence[float],
) -> Tuple[Optional[Dict[str, str]], str]:
    if not candidates:
        return None, ""

    installed_sorted = _dedupe_sort_versions(installed_versions)
    if not installed_sorted:
        return None, ""

    versioned = [c for c in candidates if c.get(candidate_version_key) is not None]
    if not versioned:
        return None, ""

    installed_set = {round(v, 2) for v in installed_sorted}
    exact = [c for c in versioned if round(float(c[candidate_version_key]), 2) in installed_set]
    if exact:
        selected = max(exact, key=lambda item: (float(item[candidate_version_key]), int(item["_score"])))
        return selected, "exact"

    highest_installed = installed_sorted[0]
    compatible = [c for c in versioned if float(c[candidate_version_key]) <= highest_installed + 1e-6]
    if compatible:
        selected = max(compatible, key=lambda item: (float(item[candidate_version_key]), int(item["_score"])))
        return selected, "lower"

    return None, ""


def _score_asset(
    name: str,
    os_id: str,
    arch_id: str,
    prefer_cuda: bool,
    prefer_rocm: bool,
    target_cuda_version: Optional[float],
) -> int:
    lower = name.lower()
    score = 0

    if _has_any(lower, _os_hints(os_id)):
        score += 100
    else:
        score -= 120

    if _is_arch_compatible(lower, arch_id):
        score += 25
    else:
        score -= 50

    if "-bin-" in lower:
        score += 12
    if os_id == "windows" and "-bin-win-" in lower:
        score += 20
    if os_id == "linux" and "-bin-ubuntu-" in lower:
        score += 20
    if os_id == "macos" and "-bin-macos-" in lower:
        score += 20
    if lower.startswith("llama-"):
        score += 8
    if lower.startswith("cudart-"):
        score -= 12

    if "llama-server" in lower:
        score += 20
    elif "server" in lower:
        score += 10

    flavor = _runtime_flavor(lower)
    if prefer_cuda and os_id in ("windows", "linux"):
        if flavor == "cuda":
            score += 55
            asset_cuda = _parse_cuda_version_from_asset(lower)
            if target_cuda_version is not None and asset_cuda is not None:
                delta = abs(asset_cuda - target_cuda_version)
                score += max(0, 25 - int(delta * 20))
                if asset_cuda <= target_cuda_version + 0.001:
                    score += 5
            elif target_cuda_version is not None:
                score -= 5
        elif flavor == "cpu":
            score -= 20
        else:
            score -= 12
    elif prefer_rocm and os_id == "linux":
        if flavor == "rocm":
            score += 55
        elif flavor == "cpu":
            score += 5
        else:
            score -= 10
    elif os_id == "macos":
        if "metal" in lower:
            score += 25
        if flavor == "cuda":
            score -= 10
    else:
        if flavor == "cpu":
            score += 12

    if lower.endswith(".zip"):
        score += 3
    if lower.endswith(".tar.gz") or lower.endswith(".tgz"):
        score += 2

    return score


def _select_asset(
    assets: List[Dict[str, str]],
    os_id: str,
    arch_id: str,
    prefer_cuda: bool,
    prefer_rocm: bool,
    target_cuda_version: Optional[float],
    cuda_versions: Sequence[float],
    rocm_versions: Sequence[float],
) -> Optional[Dict[str, str]]:
    candidates: List[Dict[str, object]] = []
    for asset in assets:
        name = str(asset.get("name", ""))
        if not name or not _is_archive(name):
            continue
        lower = name.lower()
        if _has_any(lower, BLOCK_HINTS):
            continue
        if not any(token in lower for token in ("llama", "server", "bin")):
            continue
        scored: Dict[str, object] = dict(asset)
        scored["_score"] = _score_asset(
            name=name,
            os_id=os_id,
            arch_id=arch_id,
            prefer_cuda=prefer_cuda,
            prefer_rocm=prefer_rocm,
            target_cuda_version=target_cuda_version,
        )
        scored["_os_match"] = _has_any(lower, _os_hints(os_id))
        scored["_arch_match"] = _is_arch_compatible(lower, arch_id)
        scored["_flavor"] = _runtime_flavor(lower)
        scored["_cuda_version"] = _parse_cuda_version_from_asset(lower)
        scored["_rocm_version"] = _parse_rocm_version_from_asset(lower)
        candidates.append(scored)

    if not candidates:
        return None

    primary = [c for c in candidates if bool(c["_os_match"]) and bool(c["_arch_match"])]
    cuda_primary = [c for c in primary if c["_flavor"] == "cuda"]
    rocm_primary = [c for c in primary if c["_flavor"] == "rocm"]
    cpu_primary = [c for c in primary if c["_flavor"] in ("cpu", "generic", "vulkan", "opencl", "sycl")]

    if prefer_cuda and os_id in ("windows", "linux"):
        cuda_selected, cuda_match = _pick_highest_exact_then_lower(
            candidates=cuda_primary,
            candidate_version_key="_cuda_version",
            installed_versions=cuda_versions,
        )
        if cuda_selected is not None:
            selected = dict(cuda_selected)
            cuda_asset_ver = selected.get("_cuda_version")
            if cuda_match == "exact" and cuda_asset_ver is not None:
                selected["_selection_note"] = f"Selected CUDA build {cuda_asset_ver} (exact toolkit match)."
            elif cuda_match == "lower" and cuda_asset_ver is not None and cuda_versions:
                selected["_selection_note"] = (
                    f"Selected CUDA build {cuda_asset_ver} (highest <= detected toolkit {cuda_versions[0]})."
                )
            return selected  # type: ignore[return-value]

        if cuda_primary and not cuda_versions:
            selected = dict(max(cuda_primary, key=lambda item: int(item["_score"])))
            selected["_selection_note"] = "Selected CUDA build (toolkit version not detected)."
            selected["_selection_warning"] = "cuda_version_not_detected"
            return selected  # type: ignore[return-value]

        if cuda_primary and cuda_versions:
            available_cuda_versions = _dedupe_sort_versions(
                float(c["_cuda_version"]) for c in cuda_primary if c.get("_cuda_version") is not None
            )
            highest_detected_cuda = cuda_versions[0]
            if available_cuda_versions and highest_detected_cuda < available_cuda_versions[-1]:
                if cpu_primary:
                    selected = dict(max(cpu_primary, key=lambda item: int(item["_score"])))
                    selected["_selection_note"] = (
                        f"Fell back to CPU build: detected CUDA {highest_detected_cuda} is below minimum "
                        f"available CUDA build {available_cuda_versions[-1]}."
                    )
                    selected["_selection_warning"] = "cuda_too_old_for_available_builds"
                    return selected  # type: ignore[return-value]

    if prefer_rocm and os_id == "linux":
        rocm_selected, rocm_match = _pick_highest_exact_then_lower(
            candidates=rocm_primary,
            candidate_version_key="_rocm_version",
            installed_versions=rocm_versions,
        )
        if rocm_selected is not None:
            selected = dict(rocm_selected)
            rocm_asset_ver = selected.get("_rocm_version")
            if rocm_match == "exact" and rocm_asset_ver is not None:
                selected["_selection_note"] = f"Selected ROCm build {rocm_asset_ver} (exact toolkit match)."
            elif rocm_match == "lower" and rocm_asset_ver is not None and rocm_versions:
                selected["_selection_note"] = (
                    f"Selected ROCm build {rocm_asset_ver} (highest <= detected toolkit {rocm_versions[0]})."
                )
            return selected  # type: ignore[return-value]

        if rocm_primary:
            selected = dict(max(rocm_primary, key=lambda item: int(item["_score"])))
            selected["_selection_note"] = "Selected ROCm build."
            if not rocm_versions:
                selected["_selection_warning"] = "rocm_version_not_detected"
            return selected  # type: ignore[return-value]

    if primary:
        return max(primary, key=lambda item: int(item["_score"]))  # type: ignore[return-value]

    os_only = [c for c in candidates if bool(c["_os_match"])]
    if os_only:
        return max(os_only, key=lambda item: int(item["_score"]))  # type: ignore[return-value]

    return max(candidates, key=lambda item: int(item["_score"]))  # type: ignore[return-value]


def _download_json(url: str) -> Dict:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/vnd.github+json"})
    with urlopen(request, timeout=30) as response:
        body = response.read().decode("utf-8")
    return json.loads(body)


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream"})
    with urlopen(request, timeout=120) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def _extract_archive(archive_path: Path, extract_dir: Path) -> None:
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    lower = archive_path.name.lower()
    if lower.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive.extractall(extract_dir)
        return

    with tarfile.open(archive_path, "r:*") as archive:
        archive.extractall(extract_dir)


def _find_binary(extract_dir: Path, os_id: str) -> Optional[Path]:
    def rank(path: Path) -> int:
        lower_name = path.name.lower()
        lower_path = str(path).lower()
        if os_id == "windows" and not lower_name.endswith(".exe"):
            return -1

        score = 0
        if lower_name in ("llama-server.exe", "llama-server"):
            score += 200
        elif "llama-server" in lower_name:
            score += 140
        elif "llama" in lower_name and "server" in lower_name:
            score += 100
        elif "server" in lower_name:
            score += 30

        if "test" in lower_path or "benchmark" in lower_path:
            score -= 20
        if os_id != "windows" and os.access(path, os.X_OK):
            score += 10
        return score

    candidates: List[Tuple[int, Path]] = []
    for path in extract_dir.rglob("*"):
        if not path.is_file():
            continue
        candidate_score = rank(path)
        if candidate_score > 0:
            candidates.append((candidate_score, path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _validate_binary(binary_path: Path) -> bool:
    commands = [
        [str(binary_path), "--version"],
        [str(binary_path), "--help"],
    ]
    for command in commands:
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
                check=False,
            )
            if completed.returncode == 0:
                return True
        except Exception:
            continue
    return False


def _install_runtime_bundle(discovered_binary: Path, bin_dir: Path, os_id: str) -> Tuple[Path, int]:
    """Install executable plus companion runtime files into stable bin directory."""
    binary_name = "llama-server.exe" if os_id == "windows" else "llama-server"
    target_binary = bin_dir / binary_name
    source_dir = discovered_binary.parent

    bin_dir.mkdir(parents=True, exist_ok=True)

    installed_count = 0
    # Copy all sibling files from runtime directory to preserve dynamic library dependencies.
    for source_file in source_dir.iterdir():
        if not source_file.is_file():
            continue
        destination = bin_dir / source_file.name
        shutil.copy2(source_file, destination)
        installed_count += 1

    # Ensure the stable target binary name exists even when upstream names differ.
    discovered_name = discovered_binary.name
    discovered_target = bin_dir / discovered_name
    if discovered_name != target_binary.name:
        shutil.copy2(discovered_target, target_binary)
        installed_count += 1

    if os_id != "windows":
        mode = target_binary.stat().st_mode
        target_binary.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return target_binary, installed_count


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.strip(), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/install llama-server from llama.cpp releases.")
    parser.add_argument("--cache-dir", required=True, help="Cache directory root (e.g. cache/llama_cpp).")
    parser.add_argument("--tag", default="", help="Release tag override. If omitted, latest is used.")
    parser.add_argument("--prefer-cuda", action="store_true", help="Prefer CUDA assets on Linux/Windows.")
    parser.add_argument("--prefer-rocm", action="store_true", help="Prefer ROCm assets on Linux.")
    parser.add_argument("--cuda-version", default="", help="Detected CUDA version (example: 12.4).")
    parser.add_argument(
        "--cuda-versions",
        default="",
        help="Comma-separated CUDA toolkit versions (example: 13.1,12.4).",
    )
    parser.add_argument(
        "--rocm-version",
        default="",
        help="Detected ROCm version (example: 6.0).",
    )
    parser.add_argument(
        "--rocm-versions",
        default="",
        help="Comma-separated ROCm versions (example: 6.2,6.0).",
    )
    args = parser.parse_args()

    os_id = _normalize_os()
    arch_id = _normalize_arch()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    bin_dir = cache_dir / "bin"
    releases_dir = cache_dir / "releases"
    binary_name = "llama-server.exe" if os_id == "windows" else "llama-server"
    target_binary = bin_dir / binary_name
    version_file = bin_dir / "llama-server.version"

    desired_tag = (args.tag or os.environ.get("LLAMA_CPP_VERSION", "")).strip()
    installed_tag = _read_text(version_file)
    explicit_cuda_versions: List[float] = _parse_version_list(args.cuda_versions)
    legacy_cuda_version = _parse_cuda_version(args.cuda_version)
    if legacy_cuda_version is not None:
        explicit_cuda_versions.append(legacy_cuda_version)
    detected_cuda_versions = _detect_cuda_versions() if os_id in ("windows", "linux") else []
    cuda_versions = _dedupe_sort_versions([*explicit_cuda_versions, *detected_cuda_versions])
    target_cuda_version = cuda_versions[0] if cuda_versions else None

    explicit_rocm_versions: List[float] = _parse_version_list(args.rocm_versions)
    legacy_rocm_version = _parse_rocm_version(args.rocm_version)
    if legacy_rocm_version is not None:
        explicit_rocm_versions.append(legacy_rocm_version)
    detected_rocm_versions = _detect_rocm_versions() if os_id == "linux" else []
    rocm_versions = _dedupe_sort_versions([*explicit_rocm_versions, *detected_rocm_versions])

    if args.prefer_rocm and not args.prefer_cuda:
        effective_prefer_cuda = False
        effective_prefer_rocm = os_id == "linux"
    elif args.prefer_cuda and not args.prefer_rocm:
        effective_prefer_cuda = os_id in ("windows", "linux")
        effective_prefer_rocm = False
    else:
        effective_prefer_cuda = os_id in ("windows", "linux") and bool(cuda_versions)
        effective_prefer_rocm = os_id == "linux" and bool(rocm_versions) and not effective_prefer_cuda
    windows_runtime_missing = (
        os_id == "windows"
        and target_binary.exists()
        and not (bin_dir / "llama-common.dll").exists()
    )

    if target_binary.exists() and not windows_runtime_missing:
        if not desired_tag:
            _emit(
                status="existing",
                binary_path=str(target_binary),
                version=installed_tag,
                message="Existing llama-server binary found; skipping download.",
            )
            return 0
        if installed_tag == desired_tag:
            _emit(
                status="existing",
                binary_path=str(target_binary),
                version=installed_tag,
                message="Requested llama-server version already installed.",
            )
            return 0

    release_url = f"{API_BASE}/latest" if not desired_tag else f"{API_BASE}/tags/{desired_tag}"
    try:
        release_data = _download_json(release_url)
    except HTTPError as exc:
        _emit(
            status="failed",
            message=f"GitHub release query failed ({exc.code}). Use {API_BASE} for manual download.",
        )
        return 0
    except URLError as exc:
        _emit(
            status="failed",
            message=f"Network error while querying GitHub releases: {exc.reason}.",
        )
        return 0
    except Exception as exc:
        _emit(
            status="failed",
            message=f"Failed to parse GitHub release metadata: {exc}",
        )
        return 0

    resolved_tag = str(release_data.get("tag_name", "")).strip()
    assets = release_data.get("assets") or []
    selected = _select_asset(
        assets=assets,
        os_id=os_id,
        arch_id=arch_id,
        prefer_cuda=effective_prefer_cuda,
        prefer_rocm=effective_prefer_rocm,
        target_cuda_version=target_cuda_version,
        cuda_versions=cuda_versions,
        rocm_versions=rocm_versions,
    )
    if not selected:
        detail_bits: List[str] = []
        if effective_prefer_cuda:
            detail_bits.append(f"cuda_versions={cuda_versions or 'none'}")
        if effective_prefer_rocm:
            detail_bits.append(f"rocm_versions={rocm_versions or 'none'}")
        detail_suffix = f" ({', '.join(detail_bits)})" if detail_bits else ""
        _emit(
            status="failed",
            version=resolved_tag,
            message=f"No compatible llama-server release asset found for this platform{detail_suffix}.",
        )
        return 0

    asset_name = str(selected.get("name", ""))
    asset_url = str(selected.get("browser_download_url", ""))
    if not asset_name or not asset_url:
        _emit(
            status="failed",
            version=resolved_tag,
            message="Selected release asset is missing name or download URL.",
        )
        return 0

    release_dir = releases_dir / (resolved_tag or "latest")
    archive_path = release_dir / asset_name
    extract_dir = release_dir / "extracted"

    try:
        if not archive_path.exists():
            _download_file(asset_url, archive_path)
        _extract_archive(archive_path, extract_dir)
        discovered_binary = _find_binary(extract_dir, os_id=os_id)
        if not discovered_binary:
            _emit(
                status="failed",
                version=resolved_tag,
                message=f"Downloaded asset '{asset_name}' does not contain a llama-server executable.",
            )
            return 0

        target_binary, installed_count = _install_runtime_bundle(
            discovered_binary=discovered_binary,
            bin_dir=bin_dir,
            os_id=os_id,
        )

        if not _validate_binary(target_binary):
            _emit(
                status="failed",
                binary_path=str(target_binary),
                version=resolved_tag,
                message="Installed binary failed validation (--version/--help).",
            )
            return 0

        _write_text(version_file, resolved_tag)
        selection_note = str(selected.get("_selection_note", "")).strip()
        selection_warning = str(selected.get("_selection_warning", "")).strip()
        message_parts = [f"Installed from asset: {asset_name} ({installed_count} files)"]
        if selection_note:
            message_parts.append(selection_note)
        if selection_warning:
            message_parts.append(f"warning={selection_warning}")

        _emit(
            status="installed",
            binary_path=str(target_binary),
            version=resolved_tag,
            message=" ".join(message_parts),
        )
    except Exception as exc:
        _emit(
            status="failed",
            version=resolved_tag,
            message=f"Failed to install llama-server: {exc}",
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
