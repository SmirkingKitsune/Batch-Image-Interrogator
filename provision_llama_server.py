#!/usr/bin/env python3
"""Provision llama-server for setup.sh / setup.bat.

A thin CLI over `core.llama_provisioner`, which installs a matched official
release and falls back to a source build for the platform/accelerator
combinations upstream does not publish. The `LLAMA_*` stdout contract is what
the setup scripts parse, so it is kept stable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.llama_provisioner import (  # noqa: E402
    LlamaProvisioner,
    ProvisionConfig,
    read_active_runtime,
)


def _emit(status: str, binary_path: str = "", version: str = "", message: str = "") -> None:
    """Print the key=value block setup.sh and setup.bat read back."""
    clean_message = " ".join(str(message).splitlines()).strip()
    print(f"LLAMA_STATUS={status}")
    print(f"LLAMA_BINARY_PATH={binary_path}")
    print(f"LLAMA_VERSION={version}")
    print(f"LLAMA_MESSAGE={clean_message}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install llama-server from a llama.cpp release, or build it from source."
    )
    parser.add_argument("--cache-dir", required=True, help="Cache directory root (e.g. cache/llama_cpp).")
    parser.add_argument("--tag", default="", help="Release tag / git ref. Defaults to latest.")
    parser.add_argument(
        "--accelerator",
        default="",
        help="cuda|rocm|vulkan|metal|sycl-fp32|sycl-fp16|openvino|cpu. Detected when omitted.",
    )
    parser.add_argument(
        "--method",
        default="auto",
        choices=("auto", "release", "source"),
        help="auto tries a matched release, then falls back to a source build.",
    )
    parser.add_argument("--cuda-arch", default="", help="CUDA compute capability, e.g. 121. Detected when omitted.")
    parser.add_argument("--jobs", type=int, default=0, help="Parallel compile jobs. 0 selects a safe default.")
    parser.add_argument("--verbose", action="store_true", help="Stream build output to stderr.")
    parser.add_argument("--force", action="store_true", help="Reinstall even when a runtime is already present.")

    # Accepted for compatibility with the existing setup scripts. The
    # accelerator is now selected directly, and CUDA/ROCm versions are probed by
    # the provisioner itself.
    parser.add_argument("--prefer-cuda", action="store_true", help="Shorthand for --accelerator cuda.")
    parser.add_argument("--prefer-rocm", action="store_true", help="Shorthand for --accelerator rocm.")
    parser.add_argument("--cuda-version", default="", help=argparse.SUPPRESS)
    parser.add_argument("--cuda-versions", default="", help=argparse.SUPPRESS)
    parser.add_argument("--rocm-version", default="", help=argparse.SUPPRESS)
    parser.add_argument("--rocm-versions", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()

    accelerator = args.accelerator.strip().lower()
    if not accelerator and args.prefer_cuda:
        accelerator = "cuda"
    elif not accelerator and args.prefer_rocm:
        accelerator = "rocm"

    try:
        config = ProvisionConfig(
            provision_dir=Path(args.cache_dir),
            install_method=args.method,
            version=args.tag.strip() or "latest",
            accelerator=accelerator,
            cuda_arch=args.cuda_arch.strip(),
            build_jobs=args.jobs,
            force_reinstall=args.force,
        ).normalized()
    except Exception as exc:
        _emit(status="failed", message=str(exc))
        return 0

    def log(line: str, is_stderr: bool) -> None:
        if args.verbose:
            print(line, file=sys.stderr, flush=True)

    provisioner = LlamaProvisioner(config=config, log_sink=log)
    try:
        binary = provisioner.ensure_runtime()
    except Exception as exc:
        detail = str(exc)
        if provisioner.build_log_path:
            detail += f" (build log: {provisioner.build_log_path})"
        # Exit 0 regardless: the setup scripts branch on LLAMA_STATUS, and a
        # non-zero exit would abort the whole setup run over an optional engine.
        _emit(status="failed", message=detail)
        return 0

    active = read_active_runtime(config.provision_dir)
    if provisioner.used_existing:
        _emit(
            status="existing",
            binary_path=str(binary),
            version=str(active.get("version", "")),
            message="Existing validated llama-server binary found; skipping install.",
        )
        return 0

    message = (
        f"Installed via {active.get('method', config.install_method)} "
        f"for {config.platform}/{config.arch}/{active.get('accelerator', config.accelerator)}."
    )
    if provisioner.target_mismatch:
        message += f" WARNING: {provisioner.target_mismatch}"
    _emit(
        status="installed",
        binary_path=str(binary),
        version=str(active.get("version", config.version)),
        message=message,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
