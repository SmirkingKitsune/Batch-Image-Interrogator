"""Preferences that only the Electron front end has.

Model, filter, provider and Inquiry settings are shared with PyQt6 through
their existing files. This file holds what only the Electron UI needs:
whether first run has been completed, the last directory, and view choices.
The renderer cannot keep these in browser storage because the bridge port,
and therefore the page origin, changes on every launch.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

from core.atomic_write import write_json_atomic

DEFAULTS: Dict[str, Any] = {
    "first_run_done": False,
    "last_directory": "",
    "recursive": False,
    "auto_unload": True,
    "txt_mode": "merge",
    "interrogation_model": "WD",
    "interrogation_configs": {},
    "gallery_thumb": 200,
    "active_tab": "interrogation",
}

VALID_TXT_MODES = {"none", "merge", "overwrite"}
VALID_TABS = {"interrogation", "inquiry", "gallery", "settings"}


class UiSettings:
    """Small JSON-backed key/value store with validation."""

    def __init__(self, settings_file: str = "electron_settings.json"):
        self.settings_file = Path(settings_file)
        self._lock = threading.Lock()
        self._values: Dict[str, Any] = dict(DEFAULTS)
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.settings_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(data, dict):
            for key, value in data.items():
                if key in DEFAULTS:
                    try:
                        self._values[key] = self._validate(key, value)
                    except ValueError:
                        continue

    def all(self) -> Dict[str, Any]:
        with self._lock:
            return json.loads(json.dumps(self._values))

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values.get(key, DEFAULTS.get(key))

    def set(self, key: str, value: Any) -> Any:
        if key not in DEFAULTS:
            raise KeyError(f"Unknown setting: {key}")
        value = self._validate(key, value)
        with self._lock:
            self._values[key] = value
            snapshot = dict(self._values)
        try:
            write_json_atomic(self.settings_file, snapshot, ensure_ascii=False)
        except Exception as exc:  # noqa: BLE001 - a lost preference is not fatal
            print(f"Error saving Electron UI settings: {exc}")
        return value

    @staticmethod
    def _validate(key: str, value: Any) -> Any:
        default = DEFAULTS[key]
        if isinstance(default, bool):
            return bool(value)
        if key == "txt_mode":
            if value not in VALID_TXT_MODES:
                raise ValueError(f"Invalid txt mode: {value}")
            return value
        if key == "active_tab":
            if value not in VALID_TABS:
                raise ValueError(f"Invalid tab: {value}")
            return value
        if key == "interrogation_model":
            if value not in {"WD", "CLIP", "Camie"}:
                raise ValueError(f"Invalid model type: {value}")
            return value
        if key == "gallery_thumb":
            return max(100, min(400, int(value)))
        if isinstance(default, dict):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object")
            return value
        if isinstance(default, str):
            return str(value)
        return value
