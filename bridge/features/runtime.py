"""The managed llama.cpp runtime, for the bridge.

Status comes from provisioner state (core.llama_runtime_info), which says what
is installed rather than what was configured, including a CPU fallback on
hardware that supports a GPU backend.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from bridge.server import RpcError
from core.llama_provisioner import (
    ACCELERATORS,
    LlamaProvisioner,
    ProvisionCancelled,
    ProvisionConfig,
    ProvisionError,
    UpdateStatus,
    check_for_update,
    current_arch,
    current_platform,
    cuda_cmake_architectures,
    detect_accelerator,
    detect_cuda_arch,
    validate_server,
)
from core.llama_runtime_info import (
    DEFAULT_PROVISION_DIR,
    managed_config,
    short_timestamp,
    short_version,
    summarize_runtime,
)


class RuntimeFeature:
    """Install, update, and check the llama-server the Inquiry tab launches."""

    def _init_runtime(self) -> None:
        self._provision_cancelled = False
        self._provisioning = False
        self._last_health: Optional[Dict[str, Any]] = None
        self._last_update: Optional[Dict[str, Any]] = None

    def rpc_runtime_summary(self) -> Dict[str, Any]:
        summary = summarize_runtime()
        summary["short_version"] = short_version(summary["version"]) if summary["version"] else ""
        summary["installed_short"] = short_timestamp(summary["installed_at"]) if summary["installed_at"] else ""
        summary["detected"] = detect_accelerator()
        summary["platform"] = current_platform()
        summary["arch"] = current_arch()
        summary["provisioning"] = self._provisioning
        summary["log_dir"] = str(DEFAULT_PROVISION_DIR / "logs")
        summary["last_health"] = self._last_health
        summary["last_update"] = self._last_update
        return summary

    def _publish_runtime(self) -> None:
        self.bus.publish("runtime.summary", self.rpc_runtime_summary())

    def rpc_runtime_environment(self, accelerator: str = "") -> Dict[str, Any]:
        """What the provisioning form pre-fills."""
        chosen = accelerator or detect_accelerator()
        cuda_arch = detect_cuda_arch() if chosen == "cuda" else ""
        return {
            "platform": current_platform(),
            "arch": current_arch(),
            "detected": detect_accelerator(),
            "accelerator": chosen,
            "accelerators": list(ACCELERATORS),
            "cuda_arch": cuda_arch,
            "cuda_cmake": cuda_cmake_architectures(cuda_arch) if cuda_arch else "",
        }

    def rpc_runtime_cuda_hint(self, cuda_arch: str) -> Dict[str, Any]:
        mapped = cuda_cmake_architectures(cuda_arch or "")
        return {"ok": bool(mapped), "cmake": mapped}

    def rpc_runtime_check_updates(self) -> Dict[str, Any]:
        summary = summarize_runtime()
        if not summary["installed"]:
            raise RpcError("Install a runtime before checking for updates.", code="no_runtime")

        config = managed_config()
        if summary["accelerator"]:
            config.accelerator = summary["accelerator"]
        if summary["method"] in ("release", "source"):
            # A source build tuned for this machine is never "updated" into
            # a generic archive.
            config.install_method = summary["method"]
        installed = summary["version"]

        def check() -> None:
            try:
                status = check_for_update(config, installed)
            except Exception as exc:  # noqa: BLE001
                status = UpdateStatus(current_version=installed, error=str(exc))
            self._last_update = _update_payload(status)
            self.bus.publish("runtime.update", self._last_update)

        self._spawn("runtime-update", check)
        return {"checking": True}

    def rpc_runtime_health(self) -> Dict[str, Any]:
        summary = summarize_runtime()
        if not summary["installed"]:
            self._last_health = {"ok": False, "message": "No runtime installed."}
            return self._last_health

        executable = summary["executable"]

        def check() -> None:
            started = time.monotonic()
            try:
                detail = validate_server(Path(executable))
                self._last_health = {
                    "ok": True,
                    "message": f"llama-server --version exited 0 in {time.monotonic() - started:.1f}s",
                    "detail": detail,
                }
            except (ProvisionError, OSError) as exc:
                self._last_health = {"ok": False, "message": str(exc)}
            except Exception as exc:  # noqa: BLE001
                self._last_health = {"ok": False, "message": f"Runtime health check failed: {exc}"}
            self.bus.publish("runtime.health", self._last_health)

        self._spawn("runtime-health", check)
        return {"checking": True}

    def rpc_runtime_provision(
        self,
        accelerator: str = "",
        method: str = "auto",
        version: str = "latest",
        cuda_arch: str = "",
        bypass_checks: bool = False,
    ) -> Dict[str, Any]:
        """Install or rebuild the managed runtime.

        A source build can run for twenty minutes, so the UI only calls this
        after the user has explicitly confirmed.
        """
        if self._provisioning:
            raise RpcError("Provisioning is already running.", code="busy")
        if self.llama is not None:
            raise RpcError("Unload the llama model before reinstalling its runtime.", code="busy")
        config = ProvisionConfig(
            provision_dir=DEFAULT_PROVISION_DIR,
            install_method=method if method in ("auto", "release", "source") else "auto",
            version=(version or "latest").strip() or "latest",
            accelerator=accelerator or "",
            cuda_arch=(cuda_arch or "").strip(),
            bypass_environment_checks=bool(bypass_checks),
        )
        try:
            config.normalized()
        except ProvisionError as exc:
            raise RpcError(str(exc)) from exc

        self._provisioning = True
        self._provision_cancelled = False
        self.bus.publish("runtime.provision.started", {"method": config.install_method})
        self._spawn("runtime-provision", lambda: self._provision(config))
        return {"started": True}

    def _provision(self, config: ProvisionConfig) -> None:
        provisioner = None
        binary = ""
        error = ""
        try:
            provisioner = LlamaProvisioner(
                config=config,
                log_sink=lambda line, is_stderr: self.bus.publish(
                    "runtime.provision.log", {"line": line, "stderr": bool(is_stderr)}
                ),
                progress_sink=lambda p: self.bus.publish(
                    "runtime.provision.progress",
                    {"step": p.step, "total": p.total, "label": p.label, "fraction": p.fraction},
                ),
                cancel_check=lambda: self._provision_cancelled,
            )
            binary = str(provisioner.ensure_runtime())
        except ProvisionCancelled:
            error = "Provisioning cancelled."
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
        finally:
            self._provisioning = False
            self.bus.publish(
                "runtime.provision.finished",
                {
                    "binary": binary,
                    "error": error,
                    "mismatch": getattr(provisioner, "target_mismatch", "") if provisioner else "",
                    "log_path": str(getattr(provisioner, "build_log_path", "") or "") if provisioner else "",
                },
            )
            self._last_update = None
            self._publish_runtime()

    def rpc_runtime_provision_cancel(self) -> Dict[str, Any]:
        self._provision_cancelled = True
        return {"cancelling": self._provisioning}


def _update_payload(status: UpdateStatus) -> Dict[str, Any]:
    """An update result plus the sentence the design shows for it."""
    payload = {
        "error": status.error,
        "current": status.current_version,
        "latest": status.latest_version,
        "behind": status.behind,
        "action": status.action,
        "warning": status.warning,
        "is_current": status.is_current,
        "below_threshold": status.below_threshold,
        "update_available": status.update_available,
    }
    if status.error:
        payload.update(level="error", text=status.error)
    elif status.is_current:
        payload.update(level="ok", text=f"Up to date ({status.latest_version}).")
    elif status.below_threshold:
        builds = "builds" if status.behind != 1 else "build"
        payload.update(
            level="muted",
            text=f"Close to current — {status.latest_version} is {status.behind} {builds} ahead.",
        )
    else:
        cost = {
            "release": "downloads a matched release",
            "compile": "requires a source rebuild",
            "unavailable": "has no installable build for this configuration",
        }.get(status.action, "")
        text = f"Update available: {status.latest_version}"
        if status.behind:
            text += f" — {status.behind} builds behind"
        if cost:
            text += f" — {cost}"
        payload.update(level="error" if status.action == "unavailable" else "warn", text=text)
    return payload
