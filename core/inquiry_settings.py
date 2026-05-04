"""Persistence for Inquiry tab options."""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class InquirySettings:
    """Manages persisted llama.cpp inquiry options."""

    CONTEXT_CONFIG_KEYS = {
        "include_prior_tables": "batch_include_prior_tables",
        "included_model_types": "batch_included_model_types",
        "carry_batch_context": "batch_carry_context",
    }

    DEFAULT_OPTIONS: Dict[str, Any] = {
        "llama_config": {},
        "single_task": "describe",
        "single_prompt": "",
        "batch_task": "describe",
        "batch_prompt": "",
        "batch_include_prior_tables": False,
        "single_include_prior_transcripts": False,
        "batch_include_prior_transcripts": False,
        "batch_included_model_types": ["CLIP", "WD", "Camie"],
        "batch_context_source_keys": [],
        "batch_carry_context": False,
        "batch_use_cache": None,
        "txt_output_mode": "merge",
        "active_tab": 0,
    }

    VALID_TXT_OUTPUT_MODES = {"none", "merge", "overwrite"}

    def __init__(self, settings_file: str = "inquiry_settings.json"):
        self.settings_file = Path(settings_file)
        self.options: Dict[str, Any] = dict(self.DEFAULT_OPTIONS)
        self._saved_keys = set()
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

        normalized = self._normalize_options(data)
        self.options.update(normalized)
        self._saved_keys.update(normalized.keys())

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

    def has_saved_option(self, key: str) -> bool:
        """Return True when an option was explicitly loaded or saved."""
        return key in self._saved_keys

    def update_options(self, options: Dict[str, Any]):
        """Merge and persist inquiry options."""
        normalized = self._normalize_options(options)
        self.options.update(normalized)
        self._saved_keys.update(normalized.keys())
        self.save_settings()

    def update_llama_config(self, config: Dict[str, Any]):
        """Merge and persist llama.cpp config values."""
        saved_config = self.get_llama_config()
        saved_config.update(
            {
                key: value
                for key, value in dict(config or {}).items()
                if key not in self.CONTEXT_CONFIG_KEYS
            }
        )
        self.options["llama_config"] = saved_config
        self.save_settings()

    def _normalize_options(self, data: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}

        llama_config = data.get("llama_config")
        if isinstance(llama_config, dict):
            normalized["llama_config"] = {
                key: value
                for key, value in llama_config.items()
                if key not in self.CONTEXT_CONFIG_KEYS
            }
            for old_key, new_key in self.CONTEXT_CONFIG_KEYS.items():
                if new_key not in data and old_key in llama_config:
                    normalized[new_key] = llama_config[old_key]

        for key in ("single_task", "single_prompt", "batch_task", "batch_prompt"):
            value = data.get(key)
            if isinstance(value, str):
                normalized[key] = value

        for key in (
            "batch_include_prior_tables",
            "single_include_prior_transcripts",
            "batch_include_prior_transcripts",
            "batch_carry_context",
            "batch_use_cache",
        ):
            value = data.get(key)
            if isinstance(value, bool):
                normalized[key] = value

        included_model_types = data.get("batch_included_model_types")
        if isinstance(included_model_types, list):
            normalized["batch_included_model_types"] = [
                model_type for model_type in included_model_types if isinstance(model_type, str)
            ]

        context_source_keys = data.get("batch_context_source_keys")
        if isinstance(context_source_keys, list):
            normalized["batch_context_source_keys"] = [
                source_key for source_key in context_source_keys if isinstance(source_key, str)
            ]

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
