"""The headless backend behind the Electron front end.

Owns the same shared objects MainWindow does (database, tag filters, ONNX
provider settings, Inquiry settings) and exposes them as RPC methods. The
method "group.name" dispatches to `rpc_group_name`; every feature lives in a
mixin under bridge/features.

Long operations run on background threads and report through the event bus,
so an RPC never blocks on a model load or a batch.
"""

from __future__ import annotations

import re
import threading
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from bridge.events import EventBus
from bridge.features.inquiry import InquiryFeature
from bridge.features.interrogation import InterrogationFeature
from bridge.features.library import LibraryFeature
from bridge.features.runtime import RuntimeFeature
from bridge.features.settings import SettingsFeature
from bridge.hardware import HardwareMonitor
from bridge.server import RpcError
from bridge.ui_settings import UiSettings
from core.database import InterrogationDatabase
from core.inquiry_settings import InquirySettings
from core.onnx_providers import ONNXProviderSettings
from core.tag_filters import TagFilterSettings

METHOD_PATTERN = re.compile(r"^[a-z]+\.[a-z_]+$")
TELEMETRY_INTERVAL = 1.0
DATABASE_BUSY_TIMEOUT = 300.0


class BridgeService(
    LibraryFeature,
    InterrogationFeature,
    InquiryFeature,
    RuntimeFeature,
    SettingsFeature,
):
    def __init__(
        self,
        device_status: Optional[Dict[str, Any]] = None,
        bus: Optional[EventBus] = None,
        database: Optional[InterrogationDatabase] = None,
        tag_filters: Optional[TagFilterSettings] = None,
        provider_settings: Optional[ONNXProviderSettings] = None,
        inquiry_settings: Optional[InquirySettings] = None,
        ui_settings: Optional[UiSettings] = None,
        telemetry: bool = True,
    ):
        self.device_status = dict(device_status or {})
        self.bus = bus or EventBus()
        self._lock = threading.RLock()
        self._stopping = threading.Event()
        self._threads: set = set()
        self._threads_lock = threading.Lock()
        self.started_at = time.monotonic()
        # Set by start_bridge just before Electron is launched.
        self.spawned_at: Optional[float] = None
        self._ready_reported = False

        self.database = database or InterrogationDatabase()
        self.tag_filters = tag_filters or TagFilterSettings()
        self.provider_settings = provider_settings or ONNXProviderSettings()
        self.inquiry_settings = inquiry_settings or InquirySettings()
        self.ui_settings = ui_settings or UiSettings()
        self.hardware = HardwareMonitor(self.device_status)

        self._init_library()
        self._init_interrogation()
        self._init_inquiry()
        self._init_runtime()
        self._init_settings()

        self._busy_requests: Dict[str, Dict[str, Any]] = {}
        self.database.set_busy_callback(self._on_database_busy)

        self._restore_last_directory()

        self._telemetry_thread: Optional[threading.Thread] = None
        if telemetry:
            self._telemetry_thread = threading.Thread(
                target=self._telemetry_loop, name="bridge-telemetry", daemon=True
            )
            self._telemetry_thread.start()

    def _restore_last_directory(self) -> None:
        """Reopen the directory from the previous session, when it still exists."""
        last = self.ui_settings.get("last_directory")
        if last and Path(last).is_dir():
            try:
                self.rpc_dir_open(last, bool(self.ui_settings.get("recursive")))
            except Exception:  # noqa: BLE001 - a stale path just starts empty
                pass

    # ------------------------------------------------------------------
    # Dispatch and background work
    # ------------------------------------------------------------------

    def dispatch(self, method: str, params: Dict[str, Any]) -> Any:
        if not METHOD_PATTERN.match(method or ""):
            raise RpcError(f"Unknown method: {method}", code="unknown_method")
        handler: Optional[Callable[..., Any]] = getattr(self, "rpc_" + method.replace(".", "_"), None)
        if not callable(handler):
            raise RpcError(f"Unknown method: {method}", code="unknown_method")
        try:
            return handler(**params)
        except TypeError as exc:
            # Wrong or missing parameters are a caller bug; say which.
            if "argument" in str(exc):
                raise RpcError(f"{method}: {exc}", code="bad_params") from exc
            raise

    def _spawn(self, name: str, target: Callable[[], None]) -> Optional[threading.Thread]:
        def run() -> None:
            try:
                target()
            except Exception as exc:  # noqa: BLE001
                traceback.print_exc()
                self.bus.publish("toast", {"level": "error", "message": f"{name} failed: {exc}"})
            finally:
                with self._threads_lock:
                    self._threads.discard(thread)

        thread = threading.Thread(target=run, name=f"bridge-{name}", daemon=True)
        # Checked under the lock shutdown() reads the set with, so every
        # tracked thread has started and none begins after shutdown.
        with self._threads_lock:
            if self._stopping.is_set():
                return None  # e.g. the rescan a finishing organize asks for
            self._threads.add(thread)
            thread.start()
        return thread

    # ------------------------------------------------------------------
    # App-level methods
    # ------------------------------------------------------------------

    def rpc_app_state(self) -> Dict[str, Any]:
        """Everything the window needs to draw itself after a (re)connect."""
        return {
            "device": self.device_status,
            "hardware": {**self.hardware.static_info(), **self.hardware.sample()},
            "settings": self.ui_settings.all(),
            "directory": self._dir_payload(),
            "interrogation": self.rpc_interrogate_state(),
            "inquiry": {
                "model": self.llama_info,
                "loading": self.llama_loading,
                "busy": self.inquiry_busy(),
            },
            "runtime": self.rpc_runtime_summary(),
            "filters": self.rpc_filters_get(),
            "queue": self.database.get_queue_status(),
            "cwd": str(Path.cwd()),
        }

    def rpc_app_ready(self) -> Dict[str, Any]:
        """The window reports its first paint; printed as in the launch spec."""
        if not self._ready_reported:
            self._ready_reported = True
            seconds = time.monotonic() - (self.spawned_at or self.started_at)
            print(f"[bridge] window ready in {seconds:.1f}s", flush=True)
        return {"ok": True}

    def rpc_app_hardware(self) -> Dict[str, Any]:
        return {**self.hardware.static_info(), **self.hardware.sample()}

    def rpc_app_complete_first_run(self) -> Dict[str, Any]:
        self.ui_settings.set("first_run_done", True)
        return {"ok": True}

    def on_client_connected(self) -> None:
        """A window (re)connected to the event stream."""
        self.bus.publish("hardware", self.hardware.sample())

    def _telemetry_loop(self) -> None:
        """GPU memory once a second, only while a window is listening."""
        while not self._stopping.wait(TELEMETRY_INTERVAL):
            if self.bus.subscriber_count == 0:
                continue
            try:
                self.bus.publish("hardware", self.hardware.sample())
            except Exception:  # noqa: BLE001 - telemetry must never take the bridge down
                pass

    # ------------------------------------------------------------------
    # Tag filters (shared by the rail sheet and the Settings page)
    # ------------------------------------------------------------------

    def rpc_filters_get(self) -> Dict[str, Any]:
        filters = self.tag_filters
        return {
            "prefix": filters.get_prefix_tags(),
            "remove": filters.get_remove_list(),
            "replace": filters.get_replace_dict(),
            "keep": filters.get_keep_list(),
            "underscores": filters.get_replace_underscores(),
            "stats": filters.get_statistics(),
        }

    def rpc_filters_add(self, kind: str, tag: str, replacement: str = "") -> Dict[str, Any]:
        tag = (tag or "").strip()
        if not tag:
            raise RpcError("Enter a tag.")
        filters = self.tag_filters
        if kind == "prefix":
            filters.add_prefix_tag(tag)
        elif kind == "remove":
            filters.add_remove_tag(tag)
        elif kind == "keep":
            filters.add_keep_tag(tag)
        elif kind == "replace":
            replacement = (replacement or "").strip()
            if not replacement:
                raise RpcError("Enter a replacement.")
            filters.add_replace_rule(tag, replacement)
        else:
            raise RpcError(f"Unknown filter: {kind}")
        return self._filters_changed()

    def rpc_filters_remove(self, kind: str, tag: str) -> Dict[str, Any]:
        filters = self.tag_filters
        removers = {
            "prefix": filters.remove_prefix_tag,
            "remove": filters.remove_remove_tag,
            "keep": filters.remove_keep_tag,
            "replace": filters.remove_replace_rule,
        }
        if kind not in removers:
            raise RpcError(f"Unknown filter: {kind}")
        removers[kind](tag)
        return self._filters_changed()

    def rpc_filters_clear(self, kind: str) -> Dict[str, Any]:
        filters = self.tag_filters
        clearers = {
            "prefix": filters.clear_prefix_tags,
            "remove": filters.clear_remove_list,
            "keep": filters.clear_keep_list,
            "replace": filters.clear_replace_dict,
        }
        if kind not in clearers:
            raise RpcError(f"Unknown filter: {kind}")
        clearers[kind]()
        return self._filters_changed()

    def rpc_filters_set_underscores(self, enabled: bool) -> Dict[str, Any]:
        self.tag_filters.set_replace_underscores(bool(enabled))
        return self._filters_changed()

    def _filters_changed(self) -> Dict[str, Any]:
        state = self.rpc_filters_get()
        self.bus.publish("filters.changed", state)
        return state

    # ------------------------------------------------------------------
    # Database busy: request/reply with the window
    # ------------------------------------------------------------------

    def _on_database_busy(self, operation: str, params: Dict[str, Any], retry_count: int) -> str:
        """Called from a worker thread when SQLite stays locked.

        Mirrors MainWindow: the window is asked to choose retry, queue or
        abort, and the worker waits (up to five minutes) for the answer.
        """
        request_id = uuid.uuid4().hex
        waiter = {"event": threading.Event(), "response": "abort"}
        self._busy_requests[request_id] = waiter
        queue_status = self.database.get_queue_status()
        try:
            queueable = self.database._operation_queue.is_queueable(operation)
        except Exception:  # noqa: BLE001
            queueable = False
        self.bus.publish(
            "database_busy",
            {
                "id": request_id,
                "operation": operation,
                "retry_count": retry_count,
                "queueable": bool(queueable),
                "queued": queue_status.get("operations", [])[:10],
            },
        )
        try:
            if waiter["event"].wait(timeout=DATABASE_BUSY_TIMEOUT):
                return waiter["response"]
            return "abort"
        finally:
            self._busy_requests.pop(request_id, None)

    def rpc_db_busy_reply(self, id: str, response: str) -> Dict[str, Any]:  # noqa: A002
        waiter = self._busy_requests.get(id)
        if waiter is None:
            return {"ok": False}
        waiter["response"] = response if response in ("retry", "queue", "abort") else "abort"
        waiter["event"].set()
        return {"ok": True}

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Stop background work and release models, as MainWindow.closeEvent does."""
        self._stopping.set()
        if self._runner is not None:
            self._runner.cancel()
        if self._llama_batch is not None:
            self._llama_batch.cancel()
        self._provision_cancelled = True
        self._scan_generation += 1
        self._sources_generation += 1
        for request in list(self._busy_requests.values()):
            request["event"].set()

        # Join until none are left: a thread that was mid-way through can
        # have started a follow-up before the stop flag was set.
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            with self._threads_lock:
                pending = list(self._threads)
            if not pending:
                break
            for thread in pending:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))

        for unload in (
            lambda: self.interrogator and self.interrogator.unload_model(),
            lambda: self.llama and self.llama.unload_model(),
        ):
            try:
                unload()
            except Exception:  # noqa: BLE001
                traceback.print_exc()
        self._meta_pool.shutdown(wait=False, cancel_futures=True)
        try:
            self.database.close()
        except Exception:  # noqa: BLE001
            pass
