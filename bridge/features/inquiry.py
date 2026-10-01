"""llama.cpp multimodal inquiries for the bridge.

Mirrors the PyQt6 Inquiry tab: the same persisted settings file, the same
runtime resolution (managed or custom), the same single-inquiry persistence
(core.inquiry_session) and the same batch loop (core.pipelines).
"""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from bridge.server import RpcError
from core.context_sizing import (
    format_bytes,
    kv_bytes_per_token,
    model_max_context,
    suggest_context_size,
)
from core.file_manager import FileManager
from core.gguf_metadata import read_gguf_metadata
from core.hashing import get_image_metadata, hash_image_content
from core.inquiry_session import record_single_inquiry
from core.llama_cpp_runtime import is_llama_timeout_error
from core.llama_runtime_info import (
    RUNTIME_MODE_CUSTOM,
    RUNTIME_MODE_MANAGED,
    resolve_runtime_binary,
    runtime_mode,
)
from core.model_paths import resolve_model_path
from core.pipelines import MultimodalBatchRunner, ReasoningRelay, StreamRelay, collect_context_sources
from core.reasoning_controls import describe_reasoning_controls, reasoning_controls_from_metadata

TASKS = ["describe", "ocr", "vqa", "custom", "audit"]
# Largest prompt the sizing suggestion assumes; see ui/dialogs.py.
DEFAULT_WORKLOAD_PROMPT_TOKENS = 8192
LLAMA_DEFAULTS: Dict[str, Any] = {
    "llama_runtime_mode": RUNTIME_MODE_MANAGED,
    "llama_binary_path": "",
    "llama_model_path": "",
    "llama_mmproj_path": "",
    "ctx_size": 8192,
    "gpu_layers": -1,
    "temperature": 0.0,
    "max_tokens": 8192,
    "server_port": 8080,
    "disable_reasoning": False,
    "reasoning_budget": -1,
    "no_reasoning_preserve": False,
    # "" leaves the chat template's default effort in place.
    "reasoning_effort": "",
    # DRY on every request; a looping reply is abandoned and retried once.
    "repetition_guard": True,
}
DGX_TIMEOUT_HINT = (
    "Hint: On NVIDIA ARM64 systems (for example DGX Spark), prebuilt llama.cpp "
    "binaries can be unstable for multimodal inference. Reinstall the runtime "
    "with Install Method 'Source build only' to compile a CUDA build matched to this GPU."
)


class InquiryFeature:
    """Model lifecycle, single inquiries and batch inquiries for llama.cpp."""

    def _init_inquiry(self) -> None:
        self.llama = None
        self.llama_info: Optional[Dict[str, Any]] = None
        self.llama_loading = False
        self._single: Optional[Dict[str, Any]] = None
        self._llama_batch: Optional[MultimodalBatchRunner] = None
        self._llama_batch_job: Optional[Dict[str, Any]] = None
        self._batch_sources: Optional[List[Dict[str, Any]]] = None
        self._sources_generation = 0

    def inquiry_busy(self) -> bool:
        return self._single is not None or self._llama_batch is not None

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def _llama_config(self) -> Dict[str, Any]:
        config = dict(LLAMA_DEFAULTS)
        config.update(self.inquiry_settings.get_llama_config())
        config["llama_runtime_mode"] = runtime_mode(config)
        for stale in ("include_prior_tables", "carry_batch_context", "included_model_types"):
            config.pop(stale, None)
        return config

    def rpc_inquiry_state(self) -> Dict[str, Any]:
        options = self.inquiry_settings.get_options()
        options.pop("llama_config", None)
        if not self.inquiry_settings.has_saved_option("batch_use_cache"):
            # Exact-match reuse only makes sense for deterministic sampling.
            options["batch_use_cache"] = float(self._llama_config().get("temperature", 0.0)) == 0.0
        return {
            "config": self._llama_config(),
            "options": options,
            "tasks": TASKS,
            "model": self.llama_info,
            "loading": self.llama_loading,
            "busy": self.inquiry_busy(),
            "single_running": self._single is not None,
            "batch": self._llama_batch_payload() if self._llama_batch_job else None,
            "sources": self._batch_sources,
        }

    def rpc_inquiry_save(self, config: Optional[Dict[str, Any]] = None,
                         options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Persist Inquiry settings exactly where the PyQt6 tab keeps them."""
        payload: Dict[str, Any] = dict(options or {})
        if config is not None:
            merged = self._llama_config()
            for key in LLAMA_DEFAULTS:
                if key in config:
                    merged[key] = config[key]
            if merged.get("llama_mmproj_path") in ("", None):
                merged["llama_mmproj_path"] = None
            payload["llama_config"] = merged
        if payload:
            self.inquiry_settings.update_options(payload)
        return {"saved": True}

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------

    def rpc_inquiry_load(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.llama_loading:
            raise RpcError("The llama model is already loading.", code="busy")
        if self.inquiry_busy():
            raise RpcError("Wait for the running inquiry to finish.", code="busy")
        if config is not None:
            self.rpc_inquiry_save(config=config)
        config = self._llama_config()

        binary_path = resolve_runtime_binary(config)
        if not binary_path or not Path(binary_path).exists():
            if runtime_mode(config) == RUNTIME_MODE_CUSTOM:
                raise RpcError(
                    f"Custom llama-server not found: {binary_path or '(no path set)'}",
                    code="custom_runtime_missing",
                )
            raise RpcError(
                "No llama.cpp runtime is installed.", code="no_runtime",
                data={"accelerator": self._detected_accelerator()},
            )
        if not config.get("llama_model_path"):
            raise RpcError("Multimodal model path is required.", code="no_model_path")

        self.llama_loading = True
        self.bus.publish("inquiry.model", {"state": "loading"})
        self._spawn("llama-load", lambda: self._load_llama(config, binary_path))
        return {"loading": True}

    def _load_llama(self, config: Dict[str, Any], binary_path: str) -> None:
        from interrogators import LlamaCppInterrogator

        interrogator = None
        try:
            model_path, note = resolve_model_path(config["llama_model_path"])
            notes = [f"Model: {note}"] if note else []
            mmproj_path = config.get("llama_mmproj_path")
            if mmproj_path:
                mmproj_path, note = resolve_model_path(mmproj_path)
                if note:
                    notes.append(f"MMProj: {note}")

            if self.llama:
                self.llama.unload_model()
                self.llama = None

            interrogator = LlamaCppInterrogator(model_name="LlamaCpp")
            interrogator.load_model(
                llama_binary_path=binary_path,
                llama_model_path=model_path,
                llama_mmproj_path=mmproj_path,
                ctx_size=config["ctx_size"],
                gpu_layers=config["gpu_layers"],
                temperature=config["temperature"],
                max_tokens=config["max_tokens"],
                server_port=config["server_port"],
                server_host="127.0.0.1",
                disable_reasoning=config.get("disable_reasoning", False),
                no_reasoning_preserve=config.get("no_reasoning_preserve", False),
                reasoning_budget=config.get("reasoning_budget", -1),
                reasoning_effort=config.get("reasoning_effort") or None,
                repetition_guard=config.get("repetition_guard", True) is not False,
            )
            requested_port = int(config.get("server_port", 8080))
            resolved_port = int(interrogator.get_config().get("server_port", requested_port))
            if resolved_port != requested_port:
                saved = self._llama_config()
                saved["server_port"] = resolved_port
                self.inquiry_settings.update_llama_config(saved)

            meta = interrogator.runtime.get_runtime_metadata()
            self.llama = interrogator
            self.llama_info = {
                "label": interrogator.model_name,
                "file": Path(model_path).name,
                "model_path": model_path,
                "mmproj_path": mmproj_path,
                "notes": notes,
                "url": meta.get("base_url"),
                "pid": meta.get("pid"),
                "log_path": meta.get("log_path"),
                "requested_port": requested_port,
                "port": resolved_port,
                "ctx_size": config["ctx_size"],
                # What the chat template lets a request change: the effort
                # radio and the thinking switch are drawn from this.
                "reasoning": interrogator.reasoning_controls,
                "reasoning_effort": interrogator.get_config().get("reasoning_effort"),
            }
            self.bus.publish("inquiry.model", {"state": "loaded", **self.llama_info})
        except Exception as exc:  # noqa: BLE001
            logs = ""
            if interrogator is not None:
                try:
                    logs = interrogator.runtime.get_recent_logs(max_lines=40)
                except Exception:  # noqa: BLE001
                    logs = ""
            self.llama = None
            self.llama_info = None
            self.bus.publish("inquiry.model", {"state": "error", "error": str(exc), "logs": logs})
        finally:
            self.llama_loading = False

    def rpc_inquiry_unload(self) -> Dict[str, Any]:
        if self.inquiry_busy():
            raise RpcError("Wait for the running inquiry to finish.", code="busy")
        if self.llama:
            self.llama.unload_model()
            self.llama = None
        self.llama_info = None
        self.bus.publish("inquiry.model", {"state": "unloaded"})
        return {"unloaded": True}

    def rpc_inquiry_set_reasoning(self, disabled: Optional[bool] = None,
                                  effort: Optional[str] = None) -> Dict[str, Any]:
        """Thinking on/off and effort travel on each request: no reload needed.

        Mid-batch, a change applies from the next image.
        """
        config = self._llama_config()
        if disabled is not None:
            config["disable_reasoning"] = bool(disabled)
        if effort is not None:
            config["reasoning_effort"] = str(effort).strip()
        self.inquiry_settings.update_llama_config(config)

        live = False
        effective = None
        if self.llama is not None:
            if disabled is not None and callable(getattr(self.llama, "set_disable_reasoning", None)):
                self.llama.set_disable_reasoning(bool(disabled))
                live = True
            setter = getattr(self.llama, "set_reasoning_effort", None)
            if callable(setter):
                effective = setter(config.get("reasoning_effort") or None)
                live = True
            if self.llama_info is not None:
                self.llama_info["reasoning_effort"] = effective
        return {
            "disable_reasoning": bool(config.get("disable_reasoning")),
            "reasoning_effort": config.get("reasoning_effort") or "",
            "effective_effort": effective,
            "live": live,
        }

    def rpc_inquiry_set_guard(self, enabled: bool) -> Dict[str, Any]:
        """The repetition guard rides on each request: no reload needed."""
        config = self._llama_config()
        config["repetition_guard"] = bool(enabled)
        self.inquiry_settings.update_llama_config(config)
        setter = getattr(self.llama, "set_repetition_guard", None)
        if callable(setter):
            setter(bool(enabled))
        return {"repetition_guard": bool(enabled), "live": callable(setter)}

    def rpc_inquiry_metadata(self, model_path: str = "", ctx_size: int = 0, max_tokens: int = 0) -> Dict[str, Any]:
        """GGUF sizing facts, as the PyQt6 'Read Metadata' button reports them."""
        config = self._llama_config()
        model_path = model_path or config.get("llama_model_path") or ""
        ctx = int(ctx_size or config.get("ctx_size") or 8192)
        max_out = int(max_tokens or config.get("max_tokens") or ctx)
        if not model_path:
            return {"ok": False, "lines": ["No model path selected."]}
        try:
            metadata = read_gguf_metadata(model_path)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "lines": [f"Could not read GGUF metadata: {exc}"]}

        model_max = model_max_context(metadata)
        per_token = kv_bytes_per_token(metadata)
        if not model_max and per_token is None:
            return {"ok": False, "lines": ["No context length or attention shape found in GGUF metadata."]}

        lines = []
        if model_max:
            lines.append(f"Trained context: {model_max:,} tokens.")
        if per_token:
            lines.append(
                f"Dense-attention KV estimate (f16): {format_bytes(per_token)}/token — "
                f"{format_bytes(per_token * ctx)} at the current {ctx:,}."
            )
        suggestion = suggest_context_size(
            metadata,
            prompt_tokens=DEFAULT_WORKLOAD_PROMPT_TOKENS,
            max_tokens=max_out,
            available_bytes=None,
        )
        bounded = max(256, min(131072, suggestion.suggested_ctx))
        lines.append(
            f"Suggested context: {bounded:,} "
            f"({format_bytes(per_token * bounded if per_token else None)} estimated KV, "
            f"bound by {suggestion.bound_by}), assuming prompts up to "
            f"{DEFAULT_WORKLOAD_PROMPT_TOKENS:,} tokens."
        )
        lines.extend(suggestion.notes)
        lines.append(
            "Estimate excludes weights and compute buffers; hybrid, recurrent, and "
            "sliding-window models may allocate differently. Memory fit is unverified."
        )
        reasoning = reasoning_controls_from_metadata(metadata)
        lines.extend(describe_reasoning_controls(reasoning))
        return {
            "ok": True,
            "lines": lines,
            "trained": model_max,
            "suggested": bounded if bounded != ctx else None,
            "reasoning": reasoning,
        }

    # ------------------------------------------------------------------
    # Single-image inquiry
    # ------------------------------------------------------------------

    def rpc_inquiry_image(self, path: str) -> Dict[str, Any]:
        image_path = self._require_image(path)
        file_hash = hash_image_content(image_path)
        session_key = f"single:{file_hash}"
        interrogations = self.database.get_all_interrogations_for_image(file_hash) or []
        history = []
        if self.llama:
            history = self.database.get_multimodal_history(image_hash=file_hash, model_name=self.llama.model_name)
            self._prime_session(file_hash, session_key)
        latest_raw = ""
        for row in interrogations:
            if (row.get("model_type") or "") == "LlamaCpp" and (row.get("raw_output") or "").strip():
                latest_raw = row.get("raw_output") or ""
                break
        try:
            metadata = get_image_metadata(image_path)
        except ValueError:
            metadata = {}
        return {
            "path": image_path,
            "hash": file_hash,
            "session_key": session_key,
            "meta": metadata,
            "prior": [
                {
                    "index": index,
                    "model_name": row.get("model_name"),
                    "display": self._model_display(row.get("model_name")),
                    "model_type": row.get("model_type"),
                    "tags": len(row.get("tags") or []),
                    "interrogated_at": row.get("interrogated_at"),
                }
                for index, row in enumerate(interrogations)
            ],
            "history": [self._turn_payload(turn, image_path) for turn in history],
            "latest_raw": latest_raw,
        }

    def _prime_session(self, file_hash: str, session_key: str) -> None:
        if not self.llama or not self.llama.is_loaded:
            return
        history = self.database.get_multimodal_history(
            session_key=session_key, mode="single", image_hash=file_hash, model_name=self.llama.model_name,
        )
        self.llama.set_session_history(session_key, history)

    def rpc_inquiry_send(
        self,
        path: str,
        task: str = "describe",
        prompt: str = "",
        prior_indices: Optional[List[int]] = None,
        include_transcripts: bool = False,
    ) -> Dict[str, Any]:
        from interrogators import LlamaCppInterrogator

        if not self.llama or not self.llama.is_loaded:
            raise RpcError("Load a llama model before sending inquiries.", code="no_model")
        if self.inquiry_busy():
            raise RpcError("An inquiry is already running.", code="busy")
        if task not in TASKS:
            raise RpcError(f"Unknown task: {task}")
        image_path = self._require_image(path)
        file_hash = hash_image_content(image_path)
        session_key = f"single:{file_hash}"
        interrogations = self.database.get_all_interrogations_for_image(file_hash) or []

        included_tables = []
        for index in prior_indices or []:
            if isinstance(index, int) and 0 <= index < len(interrogations):
                row = interrogations[index]
                included_tables.append({
                    "model_name": row.get("model_name"),
                    "model_type": row.get("model_type"),
                    "tags": row.get("tags", []),
                    "confidence_scores": row.get("confidence_scores"),
                    "raw_output_summary": (row.get("raw_output") or "")[:1500],
                    "interrogated_at": row.get("interrogated_at"),
                })
        included_transcripts = []
        if include_transcripts:
            history = self.database.get_multimodal_history(image_hash=file_hash, model_name=self.llama.model_name)
            included_transcripts = LlamaCppInterrogator.build_transcript_context(history)
        sidecar_tags = FileManager.read_tags_from_file(Path(image_path)) if task == "audit" else []

        self._prime_session(file_hash, session_key)
        request = {
            "task": task,
            "prompt_text": (prompt or "").strip(),
            "included_tables": included_tables,
            "included_transcripts": included_transcripts,
            "sidecar_tags": sidecar_tags,
            "image_path": image_path,
            "session_key": session_key,
            "image_hash": file_hash,
            "model_name": self.llama.model_name,
            "model_type": self.llama.get_model_type(),
            "model_config": copy.deepcopy(self.llama.get_config()),
        }
        pending_turn = {
            "prompt_type": task,
            "prompt_text": request["prompt_text"],
            "included_tables": included_tables,
            "included_transcripts": included_transcripts,
            "sidecar_tags": sidecar_tags,
            "model_name": self.llama.model_name,
            "image_path": image_path,
        }
        self._single = {"request": request, "started": time.monotonic()}
        self.bus.publish("inquiry.single.started", {"path": image_path, "turn": self._turn_payload(pending_turn, image_path)})
        self.bus.publish("activity", {"inquiring": True})
        self._spawn("inquiry-single", lambda: self._run_single(request))
        return {"started": True}

    def _run_single(self, request: Dict[str, Any]) -> None:
        image_path = request["image_path"]
        interrogator = self.llama
        started = time.monotonic()
        thinking = ReasoningRelay(
            lambda payload: self.bus.publish("inquiry.single.reasoning", {"path": image_path, **payload})
        )
        relay = StreamRelay(lambda text: self.bus.publish("inquiry.single.stream", {"path": image_path, "text": text}))

        def answer_delta(raw_text: str) -> None:
            if raw_text:
                thinking.finish()  # the first answer token ends the thinking
            relay(raw_text)
        try:
            results = interrogator.interrogate(
                image_path,
                task=request["task"],
                prompt=request["prompt_text"],
                session_key=request["session_key"],
                keep_context=True,
                included_tables=request["included_tables"],
                included_transcripts=request["included_transcripts"],
                sidecar_tags=request["sidecar_tags"],
                on_stream_delta=answer_delta,
                on_reasoning_delta=thinking,
                on_restart=thinking.restart,
            )
            relay.flush()
            thinking.finish()
            recorded = record_single_inquiry(self.database, request, results)
            self._forget_meta(image_path)
            turn = self._turn_payload(recorded["turn"], image_path)
            turn["elapsed"] = round(time.monotonic() - started, 1)
            # Shown for this session only; the database keeps the answer.
            turn["thinking"] = thinking.text
            turn["thinking_seconds"] = round(thinking.seconds, 1)
            self.bus.publish(
                "inquiry.single.done",
                {
                    "path": image_path,
                    "turn": turn,
                    "raw": results.get("raw_output", ""),
                    "removed": results.get("audit_removed_tags") or [],
                    "warnings": (results.get("multimodal_response") or {}).get("warnings", []),
                },
            )
            if request["task"] == "audit":
                self.bus.publish(
                    "gallery.tags_saved",
                    {"path": image_path, "has_txt": FileManager.has_text_file(Path(image_path))},
                )
        except Exception as exc:  # noqa: BLE001
            message = self._with_timeout_hint(str(exc).strip() or repr(exc))
            thinking.finish()
            self.bus.publish(
                "inquiry.single.error",
                {
                    "path": image_path,
                    "error": message,
                    "logs": self._recent_llama_logs(),
                    "thinking": thinking.text,
                    "thinking_seconds": round(thinking.seconds, 1),
                },
            )
        finally:
            self._single = None
            self.bus.publish("activity", {"inquiring": False})

    def rpc_inquiry_reset(self, path: str) -> Dict[str, Any]:
        if self.inquiry_busy():
            raise RpcError("Wait for the running inquiry to finish.", code="busy")
        image_path = self._require_image(path)
        file_hash = hash_image_content(image_path)
        session_key = f"single:{file_hash}"
        model_name = None
        if self.llama:
            model_name = self.llama.model_name
            self.llama.reset_session(session_key)
        self.database.clear_multimodal_session(
            session_key=session_key, model_name=model_name, mode="single", image_hash=file_hash,
        )
        return {"reset": True}

    # ------------------------------------------------------------------
    # Batch inquiry
    # ------------------------------------------------------------------

    def rpc_inquiry_sources(self) -> Dict[str, Any]:
        self._refresh_batch_sources()
        return {"refreshing": True}

    def _refresh_batch_sources(self) -> None:
        """Scan the queue for prior-result sources in the background.

        Hashing every queued image reads each file in full, which froze the
        PyQt6 window for ~30 s on large directories when done inline.
        """
        with self._lock:
            images = list(self.images)
            self._sources_generation += 1
            generation = self._sources_generation
        self._batch_sources = None
        self.bus.publish("inquiry.sources", {"sources": None, "scanning": True})

        def scan() -> None:
            sources = collect_context_sources(
                images, self.database, is_cancelled=lambda: generation != self._sources_generation,
            )
            if sources is None or generation != self._sources_generation:
                return
            payload = [
                {
                    "source_key": src["source_key"],
                    "model_name": src.get("model_name"),
                    "display": self._model_display(src.get("model_name")),
                    "model_type": src.get("model_type"),
                    "count": len(src.get("image_hashes") or []),
                }
                for src in sources
            ]
            self._batch_sources = payload
            self.bus.publish("inquiry.sources", {"sources": payload, "scanning": False})

        self._spawn("inquiry-sources", scan)

    def rpc_inquiry_batch_start(
        self,
        task: str = "describe",
        prompt: str = "",
        source_keys: Optional[List[str]] = None,
        include_transcripts: bool = False,
        carry_context: bool = False,
        use_cache: bool = False,
        txt_mode: str = "merge",
    ) -> Dict[str, Any]:
        if not self.llama or not self.llama.is_loaded:
            raise RpcError("Load a llama model before batch inquiry.", code="no_model")
        if self.inquiry_busy():
            raise RpcError("An inquiry is already running.", code="busy")
        if task not in TASKS:
            raise RpcError(f"Unknown task: {task}")
        with self._lock:
            images = list(self.images)
        if not images:
            raise RpcError("No images available from the shared queue.", code="no_images")

        # Audit always merges: it mutates sidecars by deleting rejected tags.
        if task == "audit":
            txt_mode = "merge"
        if txt_mode not in ("none", "merge", "overwrite"):
            raise RpcError(f"Invalid .txt mode: {txt_mode}")

        selected_keys = [key for key in source_keys or [] if isinstance(key, str)]
        known = {src["source_key"]: src for src in self._batch_sources or []}
        included_sources = [
            {"model_name": known[key]["model_name"], "model_type": known[key]["model_type"]}
            for key in selected_keys if key in known
        ]
        options = {
            "batch_task": task,
            "batch_prompt": prompt or "",
            "batch_include_prior_tables": bool(selected_keys),
            "batch_include_prior_transcripts": bool(include_transcripts),
            "batch_carry_context": bool(carry_context),
            "batch_use_cache": bool(use_cache),
            "batch_context_source_keys": selected_keys,
        }
        if task != "audit":
            options["txt_output_mode"] = txt_mode
        self.inquiry_settings.update_options(options)

        job = {
            "task": task,
            "prompt": (prompt or "").strip(),
            "total": len(images),
            "done": 0,
            "cached": 0,
            "failed": 0,
            "rejected": 0,
            "tags": {},
            "started_at": time.monotonic(),
            "running": None,
            "turn_started_at": {},
        }
        self._llama_batch_job = job
        runner = MultimodalBatchRunner(
            image_paths=[Path(p) for p in images],
            interrogator=self.llama,
            database=self.database,
            task=task,
            prompt=job["prompt"],
            tag_filters=self.tag_filters,
            include_prior_tables=bool(selected_keys),
            include_prior_transcripts=bool(include_transcripts),
            included_sources=included_sources,
            carry_context_across_batch=bool(carry_context),
            use_cache=bool(use_cache),
            txt_output_mode=txt_mode,
            on_progress=self._on_llama_batch_progress,
            on_turn_started=self._on_llama_turn_started,
            on_stream_delta=lambda path, text: self.bus.publish("inquiry.batch.stream", {"path": path, "text": text}),
            on_result=self._on_llama_batch_result,
            on_error=self._on_llama_batch_error,
            on_reasoning_delta=lambda path, payload: self.bus.publish(
                "inquiry.batch.reasoning", {"path": path, **payload},
            ),
        )
        self._llama_batch = runner
        self.bus.publish("inquiry.batch.started", {"total": len(images), "task": task})
        self.bus.publish("activity", {"inquiring": True})
        self._spawn("inquiry-batch", lambda: self._run_llama_batch(runner))
        return {"total": len(images)}

    def _run_llama_batch(self, runner: MultimodalBatchRunner) -> None:
        cancelled = False
        try:
            cancelled = runner.run()
        finally:
            self._llama_batch = None
            self.bus.publish(
                "inquiry.batch.finished",
                {**self._llama_batch_payload(), "cancelled": bool(cancelled or runner.was_cancelled)},
            )
            self.bus.publish("activity", {"inquiring": False})
            self._refresh_batch_sources()

    def rpc_inquiry_batch_cancel(self) -> Dict[str, Any]:
        runner = self._llama_batch
        if runner is not None:
            runner.cancel()
            self.bus.publish("inquiry.batch.cancelling", {})
        return {"cancelling": runner is not None}

    def _on_llama_batch_progress(self, current: int, total: int, message: str) -> None:
        self.bus.publish("inquiry.batch.progress", {**self._llama_batch_payload(), "message": message})

    def _on_llama_turn_started(self, image_path: str, pending_turn: Dict[str, Any]) -> None:
        job = self._llama_batch_job
        if job is not None:
            job["running"] = image_path
            job["turn_started_at"][image_path] = time.monotonic()
        self.bus.publish(
            "inquiry.batch.turn",
            {"path": image_path, "turn": self._turn_payload(pending_turn, image_path)},
        )

    def _on_llama_batch_result(self, image_path: str, results: Dict[str, Any], meta: Dict[str, Any]) -> None:
        self._forget_meta(image_path)
        job = self._llama_batch_job
        cached = bool(meta.get("cached"))
        removed = results.get("audit_removed_tags") or []
        if job is not None:
            job["cached" if cached else "done"] += 1
            job["rejected"] += len(removed)
            job["running"] = None
            for tag in results.get("tags", []) or []:
                job["tags"][tag] = job["tags"].get(tag, 0) + 1
        turn = {
            "prompt_type": (job or {}).get("task", "describe"),
            "prompt_text": (job or {}).get("prompt", ""),
            "included_tables": results.get("included_tables", []) or [],
            "included_transcripts": results.get("included_transcripts", []) or [],
            "sidecar_tags": results.get("sidecar_tags", []) or [],
            "response_json": results.get("multimodal_response", {}) or {},
            "tags": results.get("tags", []) or [],
            "model_name": self.llama.model_name if self.llama else "LlamaCpp",
            "image_path": image_path,
        }
        payload = self._turn_payload(turn, image_path)
        payload["cached"] = cached
        payload["elapsed"] = round(float(meta.get("elapsed_ms", 0.0)) / 1000.0, 1)
        payload["removed"] = removed
        payload["thinking"] = meta.get("thinking") or ""
        payload["thinking_seconds"] = round(float(meta.get("thinking_seconds") or 0.0), 1)
        self.bus.publish(
            "inquiry.batch.result",
            {"path": image_path, "turn": payload, "raw": results.get("raw_output", ""), **self._llama_batch_payload()},
        )
        self.bus.publish(
            "image_result_ready",
            {"path": image_path, "has_txt": FileManager.has_text_file(Path(image_path))},
        )

    def _on_llama_batch_error(self, image_path: str, message: str) -> None:
        job = self._llama_batch_job
        if job is not None and image_path:
            job["failed"] += 1
            job["running"] = None
        self.bus.publish(
            "inquiry.batch.error",
            {"path": image_path, "error": self._with_timeout_hint(message), **self._llama_batch_payload()},
        )

    def _llama_batch_payload(self) -> Dict[str, Any]:
        job = self._llama_batch_job or {}
        processed = job.get("done", 0) + job.get("cached", 0) + job.get("failed", 0)
        elapsed = time.monotonic() - job.get("started_at", time.monotonic())
        rate = processed / elapsed if elapsed > 0 and processed else 0.0
        remaining = max(0, job.get("total", 0) - processed)
        ranked = sorted((job.get("tags") or {}).items(), key=lambda item: item[1], reverse=True)[:60]
        return {
            "total": job.get("total", 0),
            "done": job.get("done", 0),
            "cached": job.get("cached", 0),
            "failed": job.get("failed", 0),
            "processed": processed,
            "rejected": job.get("rejected", 0),
            "task": job.get("task"),
            "running": job.get("running"),
            "rate": round(rate, 4),
            "eta": round(remaining / rate, 1) if rate > 0 else None,
            "active": self._llama_batch is not None,
            "tags": [[tag, count] for tag, count in ranked],
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _turn_payload(self, turn: Dict[str, Any], image_path: Optional[str]) -> Dict[str, Any]:
        """A transcript turn in the shape the renderer draws."""
        from interrogators import LlamaCppInterrogator

        response = turn.get("response_json") or {}
        if not isinstance(response, dict):
            response = {}
        warnings = response.get("warnings", []) or []
        parse_mode = response.get("_parse_mode", "")
        unusual = bool(
            turn.get("error")
            or "model_returned_non_json_response" in warnings
            or parse_mode == "non_json_fallback"
        )
        summary = LlamaCppInterrogator.build_prompt_display_summary(
            turn.get("prompt_type") or "describe",
            turn.get("prompt_text") or "",
            turn.get("included_tables") or [],
            included_transcripts=turn.get("included_transcripts") or [],
            sidecar_tags=turn.get("sidecar_tags") or [],
        )
        return {
            "task": turn.get("prompt_type") or "describe",
            "prompt": turn.get("prompt_text") or "",
            "summary": summary,
            "context_tables": len(turn.get("included_tables") or []),
            "context_transcripts": len(turn.get("included_transcripts") or []),
            "comment": (
                response.get("comment") or response.get("answer") or ""
            ),
            "reasoning": response.get("reasoning_summary") or "",
            # The answer came from the retry after a repetition loop.
            "loop_retried": bool(response.get("_loop_retry")),
            "raw": response.get("_debug_raw_response") or "",
            "unusual": unusual,
            "warnings": [w for w in warnings if isinstance(w, str)][:5],
            "tags": list(turn.get("tags") or []),
            "removed": list(response.get("removed_tags") or []),
            "remaining": list(response.get("remaining_tags") or []),
            "delete_tags": list(response.get("delete_tags") or []),
            "model": turn.get("model_name") or "LlamaCpp",
            "model_display": self._model_display(turn.get("model_name") or "LlamaCpp"),
            "image_path": image_path or turn.get("image_path"),
            "created_at": turn.get("created_at"),
        }

    def _model_display(self, model_name: Optional[str]) -> str:
        """A readable name for a stored llama model.

        llama.cpp resolves Hugging Face snapshot symlinks, so a model loaded
        from the HF cache is stored as "LlamaCpp/<blob sha256>". The stored
        name stays as it is (history is keyed on it); the display uses the
        GGUF file name when it is the loaded model.
        """
        name = str(model_name or "")
        if self.llama is not None and self.llama_info and name == self.llama.model_name:
            file_name = str(self.llama_info.get("file") or "")
            if file_name:
                return file_name[:-5] if file_name.lower().endswith(".gguf") else file_name
        if name.startswith("LlamaCpp/"):
            name = name[len("LlamaCpp/"):]
        return name[:-5] if name.lower().endswith(".gguf") else name

    def _recent_llama_logs(self, max_lines: int = 40) -> str:
        if not self.llama:
            return ""
        try:
            return self.llama.runtime.get_recent_logs(max_lines=max_lines)
        except Exception:  # noqa: BLE001
            return ""

    def _with_timeout_hint(self, message: str) -> str:
        import platform

        if not is_llama_timeout_error(message) or DGX_TIMEOUT_HINT in message:
            return message
        if platform.machine().lower() not in ("arm64", "aarch64"):
            return message
        if not self.device_status.get("pytorch_cuda_available"):
            return message
        return f"{message}\n\n{DGX_TIMEOUT_HINT}"

    def _detected_accelerator(self) -> str:
        try:
            from core.llama_provisioner import detect_accelerator

            return detect_accelerator()
        except Exception:  # noqa: BLE001
            return "unknown"
