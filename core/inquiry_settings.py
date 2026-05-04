"""Persistence for Inquiry tab options."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class InquirySettings:
    """Manages persisted llama.cpp inquiry options."""

    DEFAULT_OPTIONS: Dict[str, Any] = {
        "llama_config": {},
        "single_task": "describe",
        "single_prompt": "",
        "batch_task": "describe",
        "batch_prompt": "",
        "txt_output_mode": "merge",
        "active_tab": 0,
    }

    VALID_TXT_OUTPUT_MODES = {"none", "merge", "overwrite"}

    def __init__(self, settings_file: str = "inquiry_settings.json"):
        self.settings_file = Path(settings_file)
        self.options: Dict[str, Any] = dict(self.DEFAULT_OPTIONS)
        self.load_settings()

    def load_settings(self):
        """Load inquiry options from disk."""
        if not self.settings_file.exists():
            return

        try:
            with open(self.settings_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Error loading inquiry settings: {exc}")
            return

        if not isinstance(data, dict):
            return

        self.options.update(self._normalize_options(data))

    def save_settings(self):
        """Save inquiry options to disk."""
        try:
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(self.options, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"Error saving inquiry settings: {exc}")

    def get_options(self) -> Dict[str, Any]:
        """Return all persisted inquiry options."""
        return dict(self.options)

    def get_llama_config(self) -> Dict[str, Any]:
        """Return persisted llama.cpp config values."""
        config = self.options.get("llama_config", {})
        return dict(config) if isinstance(config, dict) else {}

    def update_options(self, options: Dict[str, Any]):
        """Merge and persist inquiry options."""
        self.options.update(self._normalize_options(options))
        self.save_settings()

    def update_llama_config(self, config: Dict[str, Any]):
        """Merge and persist llama.cpp config values."""
        saved_config = self.get_llama_config()
        saved_config.update(dict(config or {}))
        self.options["llama_config"] = saved_config
        self.save_settings()

    def _normalize_options(self, data: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}

        llama_config = data.get("llama_config")
        if isinstance(llama_config, dict):
            normalized["llama_config"] = dict(llama_config)

        for key in ("single_task", "single_prompt", "batch_task", "batch_prompt"):
            value = data.get(key)
            if isinstance(value, str):
                normalized[key] = value

        txt_output_mode = data.get("txt_output_mode")
        if txt_output_mode in self.VALID_TXT_OUTPUT_MODES:
            normalized["txt_output_mode"] = txt_output_mode

        active_tab = self._coerce_int(data.get("active_tab"))
        if active_tab is not None and active_tab >= 0:
            normalized["active_tab"] = active_tab

        return normalized

    @staticmethod
    def _coerce_int(value: Any) -> Optional[int]:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
