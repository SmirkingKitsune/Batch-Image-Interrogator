"""Database, providers, model cache and app preferences for the bridge."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from bridge.server import RpcError
from core.model_cache import ModelCacheManager
from core.onnx_providers import ProviderPreference


class SettingsFeature:
    """What the Database / Settings tab reads and changes."""

    def _init_settings(self) -> None:
        self._queue_processing = False

    # -- database -----------------------------------------------------------

    def rpc_db_stats(self) -> Dict[str, Any]:
        stats = self.database.get_statistics()
        location = self.database.get_db_location()
        try:
            size = Path(location).stat().st_size
        except OSError:
            size = 0
        return {
            **stats,
            "location": location,
            "size": size,
            "local": bool(self.database.use_local_db),
        }

    def rpc_db_set_mode(self, local: bool) -> Dict[str, Any]:
        self.database.set_local_mode(bool(local))
        if self.directory:
            self.database.switch_to_directory(str(self.directory))
        return self.rpc_db_stats()

    def rpc_db_vacuum(self) -> Dict[str, Any]:
        location = Path(self.database.get_db_location())
        before = location.stat().st_size if location.exists() else 0
        self.database.vacuum()
        after = location.stat().st_size if location.exists() else 0
        return {"before": before, "after": after, "saved": max(0, before - after)}

    def rpc_db_export(self, path: str) -> Dict[str, Any]:
        """Write images, models and interrogations to a JSON file."""
        target = Path(path or "")
        if not target.name:
            raise RpcError("Choose a file to export to.")
        location = self.database.get_db_location()
        # Read-only connection: an export must never lock the database for writers.
        uri = Path(location).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            images = [dict(row) for row in conn.execute(
                "SELECT id, file_path, file_hash, width, height, file_size FROM images"
            )]
            models = [dict(row) for row in conn.execute("SELECT * FROM models")]
            interrogations = []
            for row in conn.execute("SELECT * FROM interrogations"):
                entry = dict(row)
                for key in ("tags", "confidence_scores"):
                    if isinstance(entry.get(key), str):
                        try:
                            entry[key] = json.loads(entry[key])
                        except ValueError:
                            pass
                interrogations.append(entry)
        finally:
            conn.close()
        payload = {
            "exported_from": location,
            "images": images,
            "models": models,
            "interrogations": interrogations,
        }
        target.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return {"path": str(target), "images": len(images), "interrogations": len(interrogations)}

    # -- operation queue ----------------------------------------------------

    def rpc_db_queue(self) -> Dict[str, Any]:
        status = self.database.get_queue_status()
        return {**status, "processing": self._queue_processing}

    def rpc_db_process_queue(self) -> Dict[str, Any]:
        if self._queue_processing:
            raise RpcError("The queue is already being processed.", code="busy")
        self._queue_processing = True

        def run() -> None:
            try:
                success, failed = self.database.process_queued_operations()
            except Exception as exc:  # noqa: BLE001
                success, failed = 0, -1
                self.bus.publish("toast", {"level": "error", "message": f"Queue processing failed: {exc}"})
            finally:
                self._queue_processing = False
            self.bus.publish("db.queue_processed", {"success": success, "failed": failed, **self.rpc_db_queue()})

        self._spawn("db-queue", run)
        return {"processing": True}

    def rpc_db_clear_queue(self) -> Dict[str, Any]:
        cleared = self.database.clear_operation_queue()
        return {"cleared": cleared, **self.rpc_db_queue()}

    # -- providers -----------------------------------------------------------

    def _provider_status(self) -> Dict[str, Any]:
        status = self.provider_settings.get_status_info()
        preference = status["current_preference"]
        return {
            "tensorrt": status["tensorrt_available"],
            "cuda": status["cuda_available"],
            "available": status["all_providers"],
            "preference": preference.value,
            "preference_display": status["preference_display"],
            "description": status["preference_description"],
            "chain": self.provider_settings.get_provider_chain("cuda"),
            "options": [
                {"value": pref.value, "label": pref.display_name(), "description": pref.description()}
                for pref in ProviderPreference
            ],
        }

    def rpc_providers_state(self) -> Dict[str, Any]:
        return self._provider_status()

    def rpc_providers_refresh(self) -> Dict[str, Any]:
        self.provider_settings.get_available_providers(force_refresh=True)
        return self._provider_status()

    def rpc_providers_set(self, preference: str) -> Dict[str, Any]:
        self.provider_settings.set_preference(ProviderPreference.from_string(preference))
        return self._provider_status()

    # -- model cache ----------------------------------------------------------

    def rpc_cache_models(self) -> Dict[str, Any]:
        manager = ModelCacheManager()
        models: List[Dict[str, Any]] = []
        total_hf = total_trt = 0
        for model_type, infos in manager.get_all_models().items():
            for info in infos:
                total_hf += info.cache_size_bytes
                total_trt += info.tensorrt_engine_size
                models.append({
                    "id": info.model_id,
                    "type": model_type,
                    "name": info.display_name,
                    "cached": info.is_cached,
                    "size": info.cache_size_bytes,
                    "trt": info.has_tensorrt_engine,
                    "trt_size": info.tensorrt_engine_size,
                })
        gguf = []
        config = self._llama_config()
        for key in ("llama_model_path", "llama_mmproj_path"):
            path = config.get(key)
            if path:
                try:
                    gguf.append({"path": path, "name": Path(path).name, "size": Path(path).stat().st_size})
                except OSError:
                    pass
        return {
            "models": models,
            "huggingface": total_hf,
            "tensorrt": total_trt,
            "total": total_hf + total_trt,
            "gguf": gguf,
        }

    def rpc_cache_delete(self, model_id: str, tensorrt: bool = False) -> Dict[str, Any]:
        manager = ModelCacheManager()
        if model_id not in set(manager.WD_MODELS) | set(manager.CAMIE_MODELS):
            raise RpcError(f"Unknown model: {model_id}")
        loaded = (self.model_info or {}).get("label")
        if loaded == model_id:
            raise RpcError("Unload this model before deleting its cache.", code="busy")
        deleted = manager.delete_hf_model_cache(model_id)
        trt_deleted = manager.delete_tensorrt_engine(model_id) if tensorrt else False
        return {"deleted": deleted, "tensorrt_deleted": trt_deleted, **self.rpc_cache_models()}

    # -- app preferences ------------------------------------------------------

    def rpc_app_set_setting(self, key: str, value: Any) -> Dict[str, Any]:
        try:
            stored = self.ui_settings.set(key, value)
        except (KeyError, ValueError) as exc:
            raise RpcError(str(exc)) from exc
        return {"key": key, "value": stored}
