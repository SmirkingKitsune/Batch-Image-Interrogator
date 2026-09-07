"""Shared UI for the managed llama.cpp runtime.

The Inquiry tab and the Settings tab both need to answer the same question —
"what engine am I actually running?" — so the card that answers it lives here
rather than in either of them.

The card reads provisioner state rather than a configured path. That distinction
is the point of the redesign: a path tells you where a file is, while
`active-runtime.json` tells you whether it is a CUDA build or the CPU fallback
the ladder settled for, which is what determines whether a 7900-image batch
takes an evening or a week.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.llama_provisioner import (
    ProvisionConfig,
    ProvisionError,
    UpdateStatus,
    detect_accelerator,
    find_managed_executable,
    read_active_runtime,
    validate_server,
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


def runtime_summary(provision_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Describe the installed managed runtime for display.

    Keys: installed, executable, version, method, accelerator, cuda_architectures,
    installed_at, is_gpu, fallback_from.
    """
    executable = managed_executable(provision_dir)
    active = read_active_runtime(Path(provision_dir or DEFAULT_PROVISION_DIR))
    accelerator = str(active.get("accelerator", "") or "")

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
    }

    # A CPU runtime on a machine whose hardware supports something better is the
    # ladder's last rung, not a choice. Say so, permanently — the provisioning
    # dialog's one-time warning is gone by the time a slow batch is running.
    if summary["installed"] and accelerator and not summary["is_gpu"]:
        detected = detect_accelerator()
        if detected in GPU_ACCELERATORS:
            summary["fallback_from"] = detected
    return summary


def _short_version(version: str) -> str:
    """Just the build identity.

    The provisioner records `b10828 (source, cuda, cuda-arch=121a-real)`, and
    the card already shows method and architecture in their own fields, so the
    parenthetical would repeat both back a second and third time.
    """
    return version.split(" (", 1)[0].strip() or version


def _short_timestamp(value: str) -> str:
    """ISO timestamp trimmed to minutes, without the T separator."""
    trimmed = value.replace("T", " ").strip()
    parts = trimmed.split(":")
    return ":".join(parts[:2]) if len(parts) >= 2 else trimmed


class LlamaRuntimeCard(QWidget):
    """Status of the managed llama.cpp runtime, with provisioning actions.

    A plain widget rather than a group box: both call sites already sit inside
    a titled section, and nesting frames just draws a box in a box.
    """

    provision_requested = pyqtSignal()
    manage_requested = pyqtSignal()

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        provision_dir: Optional[Path] = None,
        show_manage: bool = False,
    ):
        super().__init__(parent)
        self.provision_dir = Path(provision_dir or DEFAULT_PROVISION_DIR)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.headline_label = QLabel("")
        self.headline_label.setWordWrap(True)
        layout.addWidget(self.headline_label)

        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet("color: #666;")
        layout.addWidget(self.detail_label)

        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: #b36b00;")
        self.warning_label.setVisible(False)
        layout.addWidget(self.warning_label)

        self.update_label = QLabel("")
        self.update_label.setWordWrap(True)
        self.update_label.setVisible(False)
        layout.addWidget(self.update_label)

        actions = QHBoxLayout()
        self.provision_button = QPushButton("Install Runtime")
        self.provision_button.clicked.connect(self.provision_requested.emit)
        actions.addWidget(self.provision_button)

        self.update_button = QPushButton("Check for Updates")
        self.update_button.clicked.connect(self.check_for_updates)
        actions.addWidget(self.update_button)

        self.manage_button = QPushButton("Manage...")
        self.manage_button.clicked.connect(self.manage_requested.emit)
        self.manage_button.setVisible(show_manage)
        actions.addWidget(self.manage_button)
        actions.addStretch()
        layout.addLayout(actions)

        self._update_worker = None
        self.update_status: Optional[UpdateStatus] = None
        self.refresh()

    def check_for_updates(self) -> None:
        """Ask upstream whether a newer runtime exists, off the UI thread."""
        summary = runtime_summary(self.provision_dir)
        if not summary["installed"]:
            self._show_update_text("Install a runtime before checking for updates.", "#b36b00")
            return
        if self._update_worker is not None and self._update_worker.isRunning():
            return

        # Imported here: ui.workers imports ui.dialogs, which imports this
        # module, so a module-level import would close the cycle.
        from ui.workers import LlamaUpdateCheckWorker

        self.update_button.setEnabled(False)
        self._show_update_text("Checking for updates...", "#666")

        config = managed_config(self.provision_dir)
        if summary["accelerator"]:
            config.accelerator = summary["accelerator"]
        if summary["method"] in ("release", "source"):
            # Preserve how this runtime was acquired. In particular, silently
            # replacing a source build tuned for this machine with a generic
            # archive is not the update the user asked to check for.
            config.install_method = summary["method"]
        self._update_worker = LlamaUpdateCheckWorker(config, summary["version"])
        self._update_worker.completed.connect(self._on_update_checked)
        self._update_worker.finished.connect(self._on_update_worker_finished)
        self._update_worker.start()

    def _on_update_worker_finished(self) -> None:
        self._update_worker = None

    def _on_update_checked(self, status: "UpdateStatus") -> None:
        self.update_button.setEnabled(True)
        self.update_status = status

        if status.error:
            self._show_update_text(status.error, "#b00020")
            return
        if status.is_current:
            self._show_update_text(f"Up to date ({status.latest_version}).", "#1a7f37")
            return
        if status.below_threshold:
            # Behind, but not enough to be worth acting on. Saying "up to date"
            # here would be false, and reporting it as an update would make the
            # signal meaningless — upstream tags a dozen builds a day.
            self._show_update_text(
                f"Close to current — {status.latest_version} is "
                f"{status.behind} build{'s' if status.behind != 1 else ''} ahead.",
                "#666",
            )
            return

        # Say what taking the update costs, not merely that one exists: a
        # release download is a minute and a source rebuild is twenty.
        cost = {
            "release": "downloads a matched release",
            "compile": "requires a source rebuild",
            "unavailable": "has no installable build for this configuration",
        }.get(status.action, "")
        headline = f"Update available: {status.latest_version}"
        if status.behind:
            headline += f" — {status.behind} builds behind"
        if cost:
            headline += f" — {cost}"
        text = headline if not status.warning else f"{headline}\n{status.warning}"
        self._show_update_text(text, "#b00020" if status.action == "unavailable" else "#b36b00")

    def _show_update_text(self, text: str, color: str) -> None:
        self.update_label.setText(text)
        self.update_label.setStyleSheet(f"color: {color};")
        self.update_label.setVisible(True)

    def refresh(self) -> None:
        """Re-read provisioner state and repaint.

        Any previous update result is discarded: refresh follows a provisioning
        run, after which "update available" describes a version that may no
        longer be the installed one.
        """
        self.update_status = None
        self.update_label.setVisible(False)
        summary = runtime_summary(self.provision_dir)

        if not summary["installed"]:
            detected = detect_accelerator()
            self.headline_label.setText("No runtime installed")
            self.headline_label.setStyleSheet("color: #b00020; font-weight: bold;")
            self.detail_label.setText(
                f"Detected accelerator: {detected}. Installing fetches a matched "
                "release, or builds from source when none is published."
            )
            self.warning_label.setVisible(False)
            self.update_label.setVisible(False)
            self.update_button.setEnabled(False)
            self.provision_button.setText("Install Runtime")
            return

        self.update_button.setEnabled(True)

        accelerator = summary["accelerator"] or "unknown"
        self.headline_label.setText(f"Ready — {accelerator}")
        self.headline_label.setStyleSheet(
            "color: #1a7f37; font-weight: bold;" if summary["is_gpu"]
            else "color: #b36b00; font-weight: bold;"
        )

        details = []
        if summary["version"]:
            details.append(_short_version(summary["version"]))
        if summary["method"]:
            details.append(f"via {summary['method']}")
        if summary["cuda_architectures"]:
            details.append(f"arch {summary['cuda_architectures']}")
        if summary["installed_at"]:
            details.append(f"installed {_short_timestamp(summary['installed_at'])}")
        self.detail_label.setText(" · ".join(details) or summary["executable"])
        self.detail_label.setToolTip(summary["executable"])

        if summary["fallback_from"]:
            self.warning_label.setText(
                f"Running on CPU although this machine supports {summary['fallback_from']}. "
                "Inference will be far slower. Reinstall to retry the GPU build."
            )
            self.warning_label.setVisible(True)
        else:
            self.warning_label.setVisible(False)

        self.provision_button.setText("Reinstall / Update")

    def check_health(self) -> str:
        """Run the executable to confirm it starts. Returns '' when healthy."""
        summary = runtime_summary(self.provision_dir)
        if not summary["installed"]:
            return "No runtime installed."
        try:
            validate_server(Path(summary["executable"]))
        except ProvisionError as exc:
            return str(exc)
        return ""
