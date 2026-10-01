"""Classic tagger interrogation (CLIP, WD, Camie) for the bridge.

Mirrors the PyQt6 Interrogation tab: the same interrogators, the same load
parameters, and the batch loop from core.pipelines. What the Electron UI adds
is per-image state, throughput and ETA, which are derived here from the
runner's callbacks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from bridge.server import RpcError
from core.device_detector import get_device_detector
from core.file_manager import FileManager
from core.model_catalog import (
    CAMIE_CATEGORIES,
    CAMIE_DEFAULT_CATEGORY_THRESHOLDS,
    CAMIE_MODELS,
    CAMIE_THRESHOLD_PROFILES,
    CAPTION_MODELS,
    CLIP_MODES,
    DEFAULT_CAMIE_MODEL,
    DEFAULT_CLIP_MODEL,
    DEFAULT_WD_MODEL,
    FALLBACK_CLIP_MODELS,
    WD_MODEL_NOTES,
    WD_MODELS,
)
from core.pipelines import InterrogationBatchRunner

# Discovered-tag table refresh cap, matching the PyQt6 tab's 250 ms debounce.
TAGS_PUBLISH_INTERVAL = 0.25
TOP_DISCOVERED_TAGS = 100


class InterrogationFeature:
    """Model lifecycle and batch runs for the classic taggers."""

    def _init_interrogation(self) -> None:
        self.interrogator = None
        self.model_info: Optional[Dict[str, Any]] = None
        self.model_loading = False
        self._clip_models: Optional[Dict[str, List[str]]] = None
        self._clip_models_loading = False
        self._runner: Optional[InterrogationBatchRunner] = None
        self._job: Optional[Dict[str, Any]] = None
        self._discovered: Dict[str, List[Any]] = {}
        self._tags_published_at = 0.0

    def interrogation_busy(self) -> bool:
        return self._runner is not None

    # ------------------------------------------------------------------
    # Options
    # ------------------------------------------------------------------

    def rpc_interrogate_options(self) -> Dict[str, Any]:
        detector = get_device_detector()
        self._ensure_clip_models()
        default_pytorch = detector.get_pytorch_device()
        default_onnx = detector.get_onnx_device()
        saved = self.ui_settings.get("interrogation_configs") or {}
        return {
            "wd_models": WD_MODELS,
            "wd_notes": WD_MODEL_NOTES,
            "camie_models": CAMIE_MODELS,
            "caption_models": CAPTION_MODELS,
            "clip_modes": CLIP_MODES,
            "camie_profiles": CAMIE_THRESHOLD_PROFILES,
            "camie_categories": CAMIE_CATEGORIES,
            "clip_models": self._clip_models,
            "devices": {
                "pytorch_cuda": detector.is_pytorch_cuda_available(),
                "onnx_cuda": detector.is_onnx_cuda_available(),
                "pytorch_error": detector.get_status().get("pytorch_error") or "",
                "onnx_error": detector.get_status().get("onnx_error") or "",
            },
            "defaults": {
                "CLIP": {
                    "clip_model": DEFAULT_CLIP_MODEL,
                    "caption_model": "None",
                    "mode": "best",
                    "device": default_pytorch,
                    **(saved.get("CLIP") or {}),
                },
                "WD": {
                    "wd_model": DEFAULT_WD_MODEL,
                    "threshold": 0.35,
                    "device": default_onnx,
                    **(saved.get("WD") or {}),
                },
                "Camie": {
                    "camie_model": DEFAULT_CAMIE_MODEL,
                    "threshold": 0.5,
                    "threshold_profile": "overall",
                    "device": default_onnx,
                    "enabled_categories": list(CAMIE_CATEGORIES),
                    "category_thresholds": dict(CAMIE_DEFAULT_CATEGORY_THRESHOLDS),
                    **(saved.get("Camie") or {}),
                },
            },
            "selected": self.ui_settings.get("interrogation_model"),
            "providers": self._provider_status(),
        }

    def _ensure_clip_models(self) -> None:
        """Load the open_clip model list once, in the background.

        Reading it imports open_clip, which takes over a second.
        """
        if self._clip_models is not None or self._clip_models_loading:
            return
        self._clip_models_loading = True

        def load() -> None:
            try:
                from core.clip_model_loader import get_categorized_models

                models = get_categorized_models()
            except Exception as exc:  # noqa: BLE001
                models = {"sd_1x": [], "sd_20": [], "sdxl": [], "other": list(FALLBACK_CLIP_MODELS)}
                self.bus.publish("toast", {"level": "warn", "message": f"Could not load CLIP models: {exc}"})
            self._clip_models = models
            self._clip_models_loading = False
            self.bus.publish("interrogate.clip_models", models)

        self._spawn("clip-models", load)

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def rpc_interrogate_load(self, config: Dict[str, Any]) -> Dict[str, Any]:
        if self.model_loading:
            raise RpcError("A model is already loading.", code="busy")
        if self.interrogation_busy():
            raise RpcError("Wait for the running batch to finish.", code="busy")
        model_type = config.get("type")
        if model_type not in ("CLIP", "WD", "Camie"):
            raise RpcError(f"Unknown model type: {model_type}")

        self.model_loading = True
        self.bus.publish("interrogate.model", {"state": "loading", "type": model_type})
        self._spawn("model-load", lambda: self._load_model(dict(config)))
        return {"loading": True}

    def _load_model(self, config: Dict[str, Any]) -> None:
        from interrogators import CamieInterrogator, CLIPInterrogator, WDInterrogator

        model_type = config["type"]
        try:
            if self.interrogator:
                self.interrogator.unload_model()
                self.interrogator = None

            if model_type == "CLIP":
                clip_model = config.get("clip_model") or DEFAULT_CLIP_MODEL
                caption_model = config.get("caption_model")
                if caption_model in (None, "", "None"):
                    caption_model = None
                interrogator = CLIPInterrogator(model_name=clip_model)
                load_params = {"mode": config.get("mode", "best"), "device": config.get("device", "cpu")}
                if caption_model:
                    load_params["caption_model"] = caption_model
                interrogator.load_model(**load_params)
                label = clip_model
                detail = f"Caption: {caption_model}" if caption_model else ""
            elif model_type == "WD":
                wd_model = config.get("wd_model") or DEFAULT_WD_MODEL
                interrogator = WDInterrogator(model_name=wd_model)
                interrogator.load_model(
                    threshold=float(config.get("threshold", 0.35)),
                    device=config.get("device", "cpu"),
                    provider_settings=self.provider_settings,
                )
                label = wd_model
                detail = ""
            else:
                camie_model = config.get("camie_model") or DEFAULT_CAMIE_MODEL
                profile = config.get("threshold_profile", "overall")
                interrogator = CamieInterrogator(model_name=camie_model)
                interrogator.load_model(
                    threshold=float(config.get("threshold", 0.5)),
                    device=config.get("device", "cpu"),
                    threshold_profile=profile,
                    category_thresholds=(
                        config.get("category_thresholds") if profile == "category_specific" else None
                    ),
                    enabled_categories=config.get("enabled_categories"),
                    provider_settings=self.provider_settings,
                )
                label = camie_model
                detail = ""

            self.interrogator = interrogator
            self.model_info = {
                "type": model_type,
                "label": label,
                "short": label.split("/")[-1],
                "detail": detail,
                "device": config.get("device", "cpu"),
                "config": config,
                "threshold": interrogator.get_config().get("threshold"),
            }
            saved = dict(self.ui_settings.get("interrogation_configs") or {})
            saved[model_type] = {k: v for k, v in config.items() if k != "type"}
            self.ui_settings.set("interrogation_configs", saved)
            self.ui_settings.set("interrogation_model", model_type)
            self.bus.publish("interrogate.model", {"state": "loaded", **self.model_info})
        except Exception as exc:  # noqa: BLE001
            self.interrogator = None
            self.model_info = None
            self.bus.publish("interrogate.model", {"state": "error", "type": model_type, "error": str(exc)})
        finally:
            self.model_loading = False

    def rpc_interrogate_unload(self) -> Dict[str, Any]:
        if self.interrogation_busy():
            raise RpcError("Wait for the running batch to finish.", code="busy")
        self._unload_interrogator()
        return {"unloaded": True}

    def _unload_interrogator(self, reason: str = "") -> None:
        if self.interrogator:
            try:
                self.interrogator.unload_model()
            finally:
                self.interrogator = None
        self.model_info = None
        self.bus.publish("interrogate.model", {"state": "unloaded", "reason": reason})

    # ------------------------------------------------------------------
    # Batch
    # ------------------------------------------------------------------

    def rpc_interrogate_start(self, txt_mode: str = "merge", paths: Optional[List[str]] = None) -> Dict[str, Any]:
        if self.interrogation_busy():
            raise RpcError("A batch is already running.", code="busy")
        if not self.interrogator or not self.interrogator.is_loaded:
            raise RpcError("Load a model first.", code="no_model")
        if txt_mode not in ("none", "merge", "overwrite"):
            raise RpcError(f"Invalid .txt mode: {txt_mode}")
        with self._lock:
            images = list(self.images)
        if paths:
            wanted = set(paths)
            images = [p for p in images if p in wanted]
        if not images:
            raise RpcError("No images to interrogate.", code="no_images")

        self.ui_settings.set("txt_mode", txt_mode)
        self._discovered = {}
        self._tags_published_at = 0.0
        job = {
            "total": len(images),
            "paths": images,
            "done": 0,
            "cached": 0,
            "failed": 0,
            "errors": [],
            "running": None,
            "started_at": time.monotonic(),
            "paused_at": None,
            "paused_total": 0.0,
            "txt_mode": txt_mode,
            "model": dict(self.model_info or {}),
        }
        self._job = job
        runner = InterrogationBatchRunner(
            [Path(p) for p in images],
            self.interrogator,
            self.database,
            write_files=txt_mode != "none",
            overwrite_files=txt_mode == "overwrite",
            tag_filters=self.tag_filters,
            on_item_started=self._on_item_started,
            on_result=self._on_item_result,
            on_error=self._on_item_error,
        )
        self._runner = runner
        self.bus.publish("interrogate.started", {"total": len(images), "paths": images, "txt_mode": txt_mode})
        self.bus.publish("activity", {"interrogating": True})
        self._spawn("interrogate", lambda: self._run_batch(runner))
        return {"total": len(images)}

    def _run_batch(self, runner: InterrogationBatchRunner) -> None:
        try:
            runner.run()
        finally:
            self._finish_batch(runner)

    def _finish_batch(self, runner: InterrogationBatchRunner) -> None:
        job = self._job or {}
        self._runner = None
        self._publish_tags(force=True)
        auto_unloaded = False
        if self.ui_settings.get("auto_unload") and self.interrogator and self.interrogator.is_loaded:
            try:
                self._unload_interrogator(reason="auto")
                auto_unloaded = True
            except Exception as exc:  # noqa: BLE001
                print(f"Error auto-unloading model: {exc}")
        self.bus.publish(
            "interrogate.finished",
            {
                "cancelled": runner.is_cancelled,
                "done": job.get("done", 0),
                "cached": job.get("cached", 0),
                "failed": job.get("failed", 0),
                "total": job.get("total", 0),
                "errors": job.get("errors", [])[:50],
                "auto_unloaded": auto_unloaded,
                "elapsed": self._active_elapsed(job),
            },
        )
        self.bus.publish("activity", {"interrogating": False})

    def rpc_interrogate_pause(self) -> Dict[str, Any]:
        runner = self._runner
        if runner is None or runner.is_paused:
            return {"paused": bool(runner and runner.is_paused)}
        runner.pause()
        if self._job is not None:
            self._job["paused_at"] = time.monotonic()
        self._publish_progress()
        return {"paused": True}

    def rpc_interrogate_resume(self) -> Dict[str, Any]:
        runner = self._runner
        if runner is None:
            return {"paused": False}
        job = self._job
        if job is not None and job.get("paused_at"):
            job["paused_total"] += time.monotonic() - job["paused_at"]
            job["paused_at"] = None
        runner.resume()
        self._publish_progress()
        return {"paused": False}

    def rpc_interrogate_cancel(self) -> Dict[str, Any]:
        runner = self._runner
        if runner is not None:
            runner.cancel()
            self.bus.publish("interrogate.cancelling", {})
        return {"cancelling": runner is not None}

    def rpc_interrogate_state(self) -> Dict[str, Any]:
        job = self._job
        return {
            "model": self.model_info,
            "loading": self.model_loading,
            "running": self._runner is not None,
            "progress": self._progress_payload() if job else None,
            "tags": self._tags_payload() if job else None,
            "txt_mode": self.ui_settings.get("txt_mode"),
            "auto_unload": self.ui_settings.get("auto_unload"),
        }

    # -- runner callbacks (batch thread) ---------------------------------

    def _on_item_started(self, index: int, image_path: str) -> None:
        if self._job is not None:
            self._job["running"] = image_path
        self.bus.publish("interrogate.item", {"i": index, "path": image_path, "state": "running"})
        self._publish_progress()

    def _on_item_result(self, image_path: str, results: Dict[str, Any], meta: Dict[str, Any]) -> None:
        self._forget_meta(image_path)
        job = self._job
        cached = bool(meta.get("cached"))
        if job is not None:
            job["cached" if cached else "done"] += 1
            job["running"] = None
        tags = results.get("tags") or []
        self.bus.publish(
            "interrogate.item",
            {
                "path": image_path,
                "state": "cached" if cached else "done",
                "source": "sqlite" if cached else "model",
                "tags": len(tags),
                "ms": round(float(meta.get("elapsed_ms", 0.0)), 1),
            },
        )
        # Re-emit of MainWindow's per-image signal, for the gallery.
        self.bus.publish(
            "image_result_ready",
            {"path": image_path, "has_txt": FileManager.has_text_file(Path(image_path))},
        )
        self._accumulate_tags(results)
        self._publish_tags()
        self._publish_progress()

    def _on_item_error(self, image_path: str, message: str) -> None:
        job = self._job
        if job is not None and image_path:
            job["failed"] += 1
            job["running"] = None
            job["errors"].append({"path": image_path, "error": message})
        self.bus.publish(
            "interrogate.item",
            {"path": image_path, "state": "failed", "error": message},
        )
        if not image_path:
            self.bus.publish("toast", {"level": "error", "message": message})
        self._publish_progress()

    # -- aggregation -----------------------------------------------------

    def _accumulate_tags(self, results: Dict[str, Any]) -> None:
        """Same accumulation as the PyQt6 tab: max confidence, count, category."""
        tags = results.get("tags", []) or []
        confidence_scores = results.get("confidence_scores") or {}
        categories: Dict[str, str] = {}
        model_type = (self._job or {}).get("model", {}).get("type")
        raw_output = results.get("raw_output", "")
        if raw_output and model_type == "Camie":
            try:
                raw_data = json.loads(raw_output)
                for category, tag_list in (raw_data.get("categories", {}) or {}).items():
                    for tag_info in tag_list:
                        name = tag_info.get("tag", "")
                        if name:
                            categories[name] = category
            except (ValueError, AttributeError):
                pass

        for tag in tags:
            conf = float(confidence_scores.get(tag, 0.0) or 0.0)
            category = categories.get(tag, "")
            entry = self._discovered.get(tag)
            if entry:
                entry[0] = max(entry[0], conf)
                entry[1] += 1
                entry[2] = category or entry[2]
            else:
                self._discovered[tag] = [conf, 1, category]

    def _tags_payload(self) -> Dict[str, Any]:
        ranked = sorted(self._discovered.items(), key=lambda item: (item[1][1], item[1][0]), reverse=True)
        remove_list = self.tag_filters.remove_list
        replace = self.tag_filters.replace_dict
        rows = []
        for tag, (conf, count, category) in ranked[:TOP_DISCOVERED_TAGS]:
            lower = tag.lower()
            rows.append({
                "tag": tag,
                "conf": round(conf, 4),
                "count": count,
                "category": category,
                "removed": lower in remove_list,
                "replaced": replace.get(lower),
            })
        return {"unique": len(self._discovered), "rows": rows}

    def _publish_tags(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._tags_published_at < TAGS_PUBLISH_INTERVAL:
            return
        self._tags_published_at = now
        self.bus.publish("interrogate.tags", self._tags_payload())

    @staticmethod
    def _active_elapsed(job: Dict[str, Any]) -> float:
        if not job:
            return 0.0
        now = time.monotonic()
        paused = job.get("paused_total", 0.0)
        if job.get("paused_at"):
            paused += now - job["paused_at"]
        return max(0.0, now - job.get("started_at", now) - paused)

    def _progress_payload(self) -> Dict[str, Any]:
        job = self._job or {}
        processed = job.get("done", 0) + job.get("cached", 0) + job.get("failed", 0)
        total = job.get("total", 0)
        elapsed = self._active_elapsed(job)
        rate = processed / elapsed if elapsed > 0 and processed else 0.0
        remaining = max(0, total - processed)
        runner = self._runner
        return {
            "total": total,
            "done": job.get("done", 0),
            "cached": job.get("cached", 0),
            "failed": job.get("failed", 0),
            "processed": processed,
            "running": job.get("running"),
            "rate": round(rate, 3),
            "eta": round(remaining / rate, 1) if rate > 0 else None,
            "elapsed": round(elapsed, 1),
            "paused": bool(runner and runner.is_paused),
            "active": runner is not None,
        }

    def _publish_progress(self) -> None:
        self.bus.publish("interrogate.progress", self._progress_payload())
