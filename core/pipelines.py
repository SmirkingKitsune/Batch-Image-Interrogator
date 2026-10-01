"""Qt-free batch pipelines shared by the PyQt6 workers and the Electron bridge.

The PyQt6 front end drives these through the QThread wrappers in ui/workers.py;
the headless bridge (bridge/) runs them on plain threads. Keeping the loops
here means the cache rules, sidecar writes and audit handling cannot drift
between the two front ends.

Each runner reports through optional callbacks and never touches a GUI toolkit,
so importing this module does not import Qt.
"""

import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from core.database import DatabaseBusyError, DatabaseQueuedError, InterrogationDatabase
from core.file_manager import FileManager
from core.hashing import get_image_metadata, hash_image_content
from core.tag_filters import TagFilterSettings

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    tqdm = None

ProgressCallback = Callable[[int, int, str], None]
CancelCheck = Callable[[], bool]


def _noop(*_args, **_kwargs) -> None:
    return None


def context_source_key(model_name: Optional[str], model_type: Optional[str]) -> str:
    """Stable key identifying a prior-result source."""
    return f"{model_type or ''}\u001f{model_name or ''}"


class StreamRelay:
    """Rate-limits streamed model text before it crosses into the UI.

    llama-server emits a delta per token; forwarding every one of them means a
    queued signal and a relayout per token. Emitting on a fixed interval keeps
    the transcript visibly live without drowning the event loop.
    """

    MIN_INTERVAL_SECONDS = 0.06

    def __init__(self, emit: Callable[[str], None]):
        self._emit = emit
        self._last_emit = 0.0
        self._last_text: Optional[str] = None
        self._pending: Optional[str] = None

    def __call__(self, raw_text: str) -> None:
        # Imported here so that importing core never pulls in the interrogators.
        from interrogators.llama_cpp_interrogator import LlamaCppInterrogator

        preview = LlamaCppInterrogator.extract_stream_preview(raw_text)
        if preview == self._last_text:
            self._pending = None
            return
        now = time.monotonic()
        # An empty preview is the "discard what you saw" reset; never drop it.
        if preview and now - self._last_emit < self.MIN_INTERVAL_SECONDS:
            self._pending = preview
            return
        self._send(preview, now)

    def flush(self) -> None:
        """Emit the newest throttled update, if one is still held back."""
        if self._pending is not None:
            self._send(self._pending, time.monotonic())

    def _send(self, preview: str, now: float) -> None:
        self._pending = None
        self._last_text = preview
        self._last_emit = now
        self._emit(preview)


class ReasoningRelay:
    """Rate-limits a model's streamed thinking before it reaches the UI.

    Thinking is plain, append-only text that can run to thousands of tokens,
    so each emit carries only what was added since the last one, with its
    offset -- ``{"offset": int, "text": str}`` -- because re-sending the whole
    text on every update would grow the event stream quadratically over a long
    think. `finish()` ends it with ``{"offset", "text": "", "done": True,
    "seconds"}`` the moment the answer starts. `text` keeps the full trace.
    """

    MIN_INTERVAL_SECONDS = 0.1

    def __init__(self, emit: Callable[[Dict[str, Any]], None]):
        self._emit = emit
        self._last_emit = 0.0
        self._sent = 0
        self._first_at: Optional[float] = None
        self._last_at: Optional[float] = None
        self._finished = False
        self.text = ""

    def __call__(self, fragment: str) -> None:
        if not fragment:
            return
        now = time.monotonic()
        if self._first_at is None:
            self._first_at = now
        self._last_at = now
        self.text += fragment
        if now - self._last_emit >= self.MIN_INTERVAL_SECONDS:
            self._send(now)

    @property
    def seconds(self) -> float:
        """Time from the first thought to the last one."""
        if self._first_at is None or self._last_at is None:
            return 0.0
        return self._last_at - self._first_at

    def flush(self) -> None:
        """Emit whatever is still held back."""
        if self._sent < len(self.text):
            self._send(time.monotonic())

    def finish(self) -> None:
        """Mark the end of thinking: everything held back, then the duration.

        Called when the first answer token arrives -- the only reliable end
        marker, since a structured answer can stream tags for a while before
        any readable text -- and again when the turn ends. Only the first call
        with something thought emits.
        """
        if self._finished or not self.text:
            return
        self.flush()
        self._finished = True
        self._emit({"offset": len(self.text), "text": "", "done": True, "seconds": round(self.seconds, 1)})

    def restart(self, notice: str = "") -> None:
        """The attempt was abandoned (it looped): void what was shown.

        Emits ``{"offset": 0, "text": "", "reset": True, "notice": ...}`` so a
        view clears the old thoughts and can say why the turn started over.
        """
        self.text = ""
        self._sent = 0
        self._last_emit = 0.0
        self._first_at = self._last_at = None
        self._finished = False
        self._emit({"offset": 0, "text": "", "reset": True, "notice": notice})

    def _send(self, now: float) -> None:
        offset = self._sent
        self._sent = len(self.text)
        self._last_emit = now
        self._emit({"offset": offset, "text": self.text[offset:]})


def scan_image_directory(
    directory: str,
    recursive: bool = False,
    is_cancelled: Optional[CancelCheck] = None,
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> Optional[List[str]]:
    """List supported images in a directory, cancellably.

    Returns the sorted, de-duplicated paths, or None when cancelled. Raises
    ValueError for a path that is not a directory.
    """
    is_cancelled = is_cancelled or (lambda: False)
    on_progress = on_progress or _noop

    dir_path = Path(directory)
    if not dir_path.exists() or not dir_path.is_dir():
        raise ValueError(f"Invalid directory: {directory}")

    extensions = FileManager.SUPPORTED_EXTENSIONS
    image_paths: List[str] = []
    count = 0

    if recursive:
        for root, _dirs, files in os.walk(directory):
            if is_cancelled():
                break
            for filename in files:
                if is_cancelled():
                    break
                if os.path.splitext(filename)[1].lower() in extensions:
                    image_paths.append(os.path.join(root, filename))
                    count += 1
                    # Report every 10 files for responsive feedback.
                    if count % 10 == 0:
                        on_progress(count, filename)
    else:
        with os.scandir(directory) as entries:
            for entry in entries:
                if is_cancelled():
                    break
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in extensions:
                    image_paths.append(entry.path)
                    count += 1
                    if count % 10 == 0:
                        on_progress(count, entry.name)

    if is_cancelled():
        return None

    image_paths = sorted(set(image_paths))
    on_progress(len(image_paths), "")
    return image_paths


def collect_context_sources(
    image_paths: Sequence[str],
    database: InterrogationDatabase,
    is_cancelled: Optional[CancelCheck] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Find the prior interrogation sources available across a set of images.

    Each image has to be hashed, which means reading the whole file, and then
    looked up in the database. Returns the sources sorted by coverage, or None
    when cancelled.
    """
    is_cancelled = is_cancelled or (lambda: False)
    on_progress = on_progress or _noop

    total = len(image_paths)
    sources: Dict[str, Dict[str, Any]] = {}
    last_percent = -1

    for index, image_path in enumerate(image_paths):
        if is_cancelled():
            return None

        try:
            file_hash = hash_image_content(image_path)
            rows = database.get_all_interrogations_for_image(file_hash) or []
        except Exception:
            continue

        for interrog in rows:
            source_key = context_source_key(interrog.get("model_name"), interrog.get("model_type"))
            source = sources.setdefault(
                source_key,
                {
                    "source_key": source_key,
                    "model_name": interrog.get("model_name"),
                    "model_type": interrog.get("model_type"),
                    "image_hashes": set(),
                    "latest_at": interrog.get("interrogated_at") or "",
                },
            )
            source["image_hashes"].add(file_hash)
            latest_at = interrog.get("interrogated_at") or ""
            if latest_at > (source.get("latest_at") or ""):
                source["latest_at"] = latest_at

        percent = int(((index + 1) / total) * 100) if total else 100
        if percent != last_percent:
            last_percent = percent
            on_progress(index + 1, total)

    if is_cancelled():
        return None

    return sorted(
        sources.values(),
        key=lambda src: (
            -(len(src.get("image_hashes") or [])),
            str(src.get("model_type") or ""),
            str(src.get("model_name") or ""),
        ),
    )


def filter_tags_for_output(
    results: Dict[str, Any],
    tag_filters: Optional[TagFilterSettings],
    interrogator,
) -> List[str]:
    """Apply the tag filter rules to a result, as they apply to .txt output."""
    tags_to_write = results["tags"]
    if not tag_filters:
        return tags_to_write
    confidence_scores = results.get("confidence_scores")
    if confidence_scores is not None:
        # Confidence-based filtering needs the model's own threshold.
        threshold = interrogator.get_config().get("threshold", 0.35)
        filtered, _ = tag_filters.filter_tags_with_confidence(
            tags_to_write, confidence_scores, threshold
        )
        return filtered
    # No confidence scores (CLIP, llama.cpp): rule-based filtering only.
    return tag_filters.apply_filters(tags_to_write)


class _PauseGate:
    """Blocks a batch between images while paused, without spinning."""

    def __init__(self):
        self._running = threading.Event()
        self._running.set()

    @property
    def paused(self) -> bool:
        return not self._running.is_set()

    def pause(self) -> None:
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    def wait(self, is_cancelled: CancelCheck) -> None:
        while not self._running.wait(timeout=0.2):
            if is_cancelled():
                return


class InterrogationBatchRunner:
    """Batch interrogation loop for the classic taggers (CLIP, WD, Camie).

    Callbacks:
        on_progress(current, total, message)
        on_item_started(index, image_path)
        on_result(image_path, results, meta) where meta carries
            {"cached": bool, "elapsed_ms": float}
        on_error(image_path, message)
    """

    def __init__(
        self,
        image_paths: Sequence[Path],
        interrogator,
        database: InterrogationDatabase,
        write_files: bool = True,
        overwrite_files: bool = False,
        tag_filters: Optional[TagFilterSettings] = None,
        on_progress: Optional[ProgressCallback] = None,
        on_item_started: Optional[Callable[[int, str], None]] = None,
        on_result: Optional[Callable[[str, Dict[str, Any], Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
    ):
        self.image_paths = list(image_paths)
        self.interrogator = interrogator
        self.database = database
        self.write_files = write_files
        self.overwrite_files = overwrite_files
        self.tag_filters = tag_filters
        self.on_progress = on_progress or _noop
        self.on_item_started = on_item_started or _noop
        self.on_result = on_result or _noop
        self.on_error = on_error or _noop
        self.is_cancelled = False
        self._gate = _PauseGate()

    def cancel(self) -> None:
        self.is_cancelled = True
        # A paused batch has to wake up to notice the cancellation.
        self._gate.resume()

    def pause(self) -> None:
        self._gate.pause()

    def resume(self) -> None:
        self._gate.resume()

    @property
    def is_paused(self) -> bool:
        return self._gate.paused

    def run(self) -> None:
        """Execute batch interrogation."""
        total = len(self.image_paths)

        # Register model once
        try:
            model_id = self.database.register_model(
                self.interrogator.model_name,
                self.interrogator.get_model_type(),
                config=self.interrogator.get_config(),
            )
        except Exception as e:
            self.on_error("", f"Failed to register model: {e}")
            return

        for idx, image_path in enumerate(self.image_paths):
            self._gate.wait(lambda: self.is_cancelled)
            if self.is_cancelled:
                break

            image_path = Path(image_path)
            image_path_str = str(image_path)
            started = time.perf_counter()
            try:
                self.on_item_started(idx, image_path_str)
                self.on_progress(idx + 1, total, f"Processing: {image_path.name}")

                file_hash = hash_image_content(image_path_str)

                # Check cache first
                cached = self.database.get_interrogation(file_hash, self.interrogator.model_name)
                from_cache = bool(cached and self.should_use_cached_result(cached))

                if from_cache:
                    results = cached
                    self.on_progress(idx + 1, total, f"Using cached: {image_path.name}")
                else:
                    if cached:
                        self.on_progress(
                            idx + 1, total,
                            f"Reprocessing empty cached result: {image_path.name}",
                        )

                    results = self.interrogator.interrogate(image_path_str)

                    # Register image and save to database
                    metadata = get_image_metadata(image_path_str)
                    try:
                        image_id = self.database.register_image(
                            image_path_str,
                            file_hash,
                            metadata['width'],
                            metadata['height'],
                            metadata['file_size'],
                        )
                    except DatabaseBusyError as e:
                        # User chose to abort - can't save without image_id
                        self.on_error(image_path_str, f"Database busy: {e}")
                        continue

                    try:
                        self.database.save_interrogation(
                            image_id,
                            model_id,
                            results['tags'],
                            results.get('confidence_scores'),
                            results.get('raw_output'),
                        )
                    except DatabaseQueuedError:
                        # Operation was queued for later - continue processing
                        pass
                    except DatabaseBusyError as e:
                        # User chose to abort this operation
                        self.on_error(image_path_str, f"Database busy: {e}")
                        continue

                if self.write_files:
                    FileManager.write_tags_to_file(
                        image_path,
                        filter_tags_for_output(results, self.tag_filters, self.interrogator),
                        overwrite=self.overwrite_files,
                    )

                self.on_result(
                    image_path_str,
                    results,
                    {"cached": from_cache, "elapsed_ms": (time.perf_counter() - started) * 1000.0},
                )

            except Exception as e:
                message = str(e).strip() or repr(e)
                self.on_error(image_path_str, message)

    def should_use_cached_result(self, cached: Dict[str, Any]) -> bool:
        """Return False for stale empty ONNX tagger rows that should be regenerated."""
        model_type = self.interrogator.get_model_type()
        if model_type == 'WD':
            return False

        tags = cached.get('tags')
        if tags:
            return True

        if model_type == 'Camie':
            return False
        return True


class MultimodalBatchRunner:
    """Batch loop for llama.cpp multimodal inquiries.

    Callbacks:
        on_progress(current, total, message)
        on_turn_started(image_path, pending_turn)
        on_stream_delta(image_path, accumulated_text)
        on_reasoning_delta(image_path, payload), optional: streamed thinking
            in ReasoningRelay's payload shape
        on_result(image_path, results, meta) where meta carries
            {"cached": bool, "elapsed_ms": float, "thinking": str,
             "thinking_seconds": float}
        on_error(image_path, message)

    run() returns True when the batch was cancelled.
    """

    CACHE_VERSION = 1
    PROMPT_BUILDER_VERSION = "llama_cpp_interrogator_prompt_v1"
    VALID_TXT_OUTPUT_MODES = {"none", "merge", "overwrite"}

    def __init__(
        self,
        image_paths: Sequence[Path],
        interrogator,
        database: InterrogationDatabase,
        task: str,
        prompt: str,
        write_files: bool = True,
        overwrite_files: bool = False,
        tag_filters: Optional[TagFilterSettings] = None,
        include_prior_tables: bool = False,
        include_prior_transcripts: bool = False,
        included_model_types: Optional[List[str]] = None,
        included_sources: Optional[List[Dict[str, Any]]] = None,
        carry_context_across_batch: bool = False,
        use_cache: bool = False,
        txt_output_mode: Optional[str] = None,
        on_progress: Optional[ProgressCallback] = None,
        on_turn_started: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_stream_delta: Optional[Callable[[str, str], None]] = None,
        on_result: Optional[Callable[[str, Dict[str, Any], Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[str, str], None]] = None,
        on_reasoning_delta: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        self.image_paths = list(image_paths)
        self.interrogator = interrogator
        self.database = database
        self.task = task
        self.prompt = prompt
        self.txt_output_mode = self.resolve_txt_output_mode(
            write_files=write_files,
            overwrite_files=overwrite_files,
            txt_output_mode=txt_output_mode,
        )
        self.write_files = self.txt_output_mode != "none"
        self.overwrite_files = self.txt_output_mode == "overwrite"
        self.tag_filters = tag_filters
        self.include_prior_tables = include_prior_tables
        self.include_prior_transcripts = include_prior_transcripts
        self.included_model_types = set(included_model_types or [])
        self.uses_included_sources = included_sources is not None
        self.included_source_keys = {
            context_source_key(source.get("model_name"), source.get("model_type"))
            for source in included_sources or []
            if isinstance(source, dict)
        }
        self.carry_context_across_batch = carry_context_across_batch
        self.use_cache = bool(use_cache)
        self.on_progress = on_progress or _noop
        self.on_turn_started = on_turn_started or _noop
        self.on_stream_delta = on_stream_delta or _noop
        self.on_reasoning_delta = on_reasoning_delta
        self.on_result = on_result or _noop
        self.on_error = on_error or _noop
        self.is_cancelled = False
        self.was_cancelled = False

    def cancel(self) -> None:
        self.is_cancelled = True
        self.was_cancelled = True

    def run(self) -> bool:
        """Execute multimodal batch interrogation."""
        total = len(self.image_paths)

        if self.is_cancelled:
            self.was_cancelled = True
            return True

        try:
            model_id = self.database.register_model(
                self.interrogator.model_name,
                self.interrogator.get_model_type(),
                config=self.interrogator.get_config(),
            )
        except Exception as e:
            self.on_error("", f"Failed to register model: {e}")
            return self.was_cancelled

        batch_run_id = str(uuid.uuid4())
        shared_session_key = f"batch:{batch_run_id}" if self.carry_context_across_batch else None

        progress_ctx = (
            tqdm(
                total=total,
                desc=f"llama-cpp {self.task}",
                unit="job",
                dynamic_ncols=True,
                leave=False,
            )
            if tqdm and total > 0
            else nullcontext()
        )
        with progress_ctx as tqdm_bar:
            for idx, image_path in enumerate(self.image_paths):
                if self.is_cancelled:
                    self.was_cancelled = True
                    break

                image_path = Path(image_path)
                image_path_str = str(image_path)
                started = time.perf_counter()
                try:
                    self.on_progress(idx + 1, total, f"Processing: {image_path.name}")
                    file_hash = hash_image_content(image_path_str)

                    metadata = get_image_metadata(image_path_str)
                    image_id = self.database.register_image(
                        image_path_str,
                        file_hash,
                        metadata["width"],
                        metadata["height"],
                        metadata["file_size"],
                    )

                    included_tables = self.build_included_tables(file_hash)
                    included_transcripts = self.build_included_transcripts(file_hash)
                    sidecar_tags = (
                        FileManager.read_tags_from_file(image_path)
                        if self.task == "audit"
                        else []
                    )
                    cache_key = None
                    cache_metadata: Dict[str, Any] = {}
                    can_use_cache = self.use_cache and not self.carry_context_across_batch
                    if can_use_cache:
                        cache_key, cache_metadata = self.build_cache_identity(
                            included_tables,
                            included_transcripts,
                            sidecar_tags,
                        )
                        cached = self.database.get_interrogation_cache_entry(
                            file_hash,
                            self.interrogator.model_name,
                            cache_key,
                        )
                    else:
                        cached = None

                    if shared_session_key:
                        session_key = shared_session_key
                    else:
                        session_key = f"batch:{batch_run_id}:{file_hash}"

                    thinking = None
                    if cached:
                        results = cached
                        self.on_progress(idx + 1, total, f"Using exact cache: {image_path.name}")
                    else:
                        # The request half of the turn is fully known now, so
                        # the transcript can show it while the model works.
                        self.on_turn_started(
                            image_path_str,
                            {
                                "prompt_type": self.task,
                                "prompt_text": self.prompt,
                                "included_tables": included_tables,
                                "included_transcripts": included_transcripts,
                                "sidecar_tags": sidecar_tags,
                                "model_name": self.interrogator.model_name,
                                "image_path": image_path_str,
                            },
                        )
                        extra: Dict[str, Any] = {}
                        if self.on_reasoning_delta is not None:
                            thinking = ReasoningRelay(
                                lambda payload, path=image_path_str: self.on_reasoning_delta(path, payload)
                            )
                            extra["on_reasoning_delta"] = thinking
                            extra["on_restart"] = thinking.restart
                        relay = StreamRelay(
                            lambda text, path=image_path_str: self.on_stream_delta(path, text)
                        )

                        def answer_delta(raw_text: str, held=thinking, preview=relay) -> None:
                            if held is not None and raw_text:
                                held.finish()  # the first answer token ends the thinking
                            preview(raw_text)

                        results = self.interrogator.interrogate(
                            image_path_str,
                            task=self.task,
                            prompt=self.prompt,
                            session_key=session_key,
                            keep_context=bool(self.carry_context_across_batch),
                            included_tables=included_tables,
                            included_transcripts=included_transcripts,
                            sidecar_tags=sidecar_tags,
                            on_stream_delta=answer_delta,
                            **extra,
                        )
                        relay.flush()
                        if thinking is not None:
                            thinking.finish()
                        if cache_key:
                            self.database.save_interrogation_cache_entry(
                                image_id=image_id,
                                model_id=model_id,
                                cache_key=cache_key,
                                cache_metadata=cache_metadata,
                                results=results,
                            )
                    results["included_tables"] = included_tables
                    results["included_transcripts"] = included_transcripts
                    results["sidecar_tags"] = sidecar_tags

                    if self.task == "audit" and self.write_files:
                        removed_tags, remaining_tags = FileManager.delete_tags_from_file(
                            image_path,
                            (results.get("multimodal_response") or {}).get("delete_tags", []),
                        )
                        response_json = results.get("multimodal_response", {}) or {}
                        response_json["removed_tags"] = removed_tags
                        response_json["remaining_tags"] = remaining_tags
                        results["multimodal_response"] = response_json
                        results["audit_removed_tags"] = removed_tags
                        results["audit_remaining_tags"] = remaining_tags
                        results["tags"] = remaining_tags

                    # Keep latest multimodal result in main interrogations table.
                    self.database.save_interrogation(
                        image_id,
                        model_id,
                        results["tags"],
                        results.get("confidence_scores"),
                        results.get("raw_output"),
                    )

                    response_json = results.get("multimodal_response", {})
                    session_id = self.database.create_or_get_multimodal_session(
                        image_id=image_id,
                        model_id=model_id,
                        mode="batch",
                        session_key=session_key,
                    )
                    self.database.append_multimodal_turn(
                        session_id=session_id,
                        prompt_type=self.task,
                        prompt_text=self.prompt,
                        included_tables=included_tables,
                        included_transcripts=included_transcripts,
                        sidecar_tags=sidecar_tags,
                        response_json=response_json,
                        tags=results["tags"],
                        reasoning_summary=response_json.get("reasoning_summary", ""),
                    )

                    if self.write_files and self.task != "audit":
                        FileManager.write_tags_to_file(
                            image_path,
                            filter_tags_for_output(results, self.tag_filters, self.interrogator),
                            overwrite=self.txt_output_mode == "overwrite",
                        )

                    self.on_result(
                        image_path_str,
                        results,
                        {
                            "cached": bool(cached),
                            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                            # Display only: never part of `results`, which is
                            # what the database and the exact-match cache keep.
                            "thinking": thinking.text if thinking is not None else "",
                            "thinking_seconds": thinking.seconds if thinking is not None else 0.0,
                        },
                    )

                except DatabaseQueuedError:
                    # Continue even if DB operation queued.
                    continue
                except DatabaseBusyError as e:
                    self.on_error(image_path_str, f"Database busy: {e}")
                except Exception as e:
                    message = str(e).strip() or repr(e)
                    self.on_error(image_path_str, message)
                finally:
                    if tqdm_bar is not None:
                        tqdm_bar.set_postfix_str(image_path.name, refresh=False)
                        tqdm_bar.update(1)

        return self.was_cancelled

    @classmethod
    def resolve_txt_output_mode(
        cls,
        write_files: bool,
        overwrite_files: bool,
        txt_output_mode: Optional[str],
    ) -> str:
        """Normalize text-output settings to the three UI modes."""
        if txt_output_mode is not None:
            if txt_output_mode not in cls.VALID_TXT_OUTPUT_MODES:
                raise ValueError(f"Invalid txt_output_mode: {txt_output_mode}")
            return txt_output_mode

        if not write_files:
            return "none"
        if overwrite_files:
            return "overwrite"
        return "merge"

    def build_included_tables(self, file_hash: str) -> List[Dict[str, Any]]:
        """Build prior interrogation context tables for a single image."""
        if not self.include_prior_tables:
            return []
        if self.uses_included_sources and not self.included_source_keys:
            return []

        tables = self.database.get_all_interrogations_for_image(file_hash)
        filtered: List[Dict[str, Any]] = []
        for row in tables:
            source_key = context_source_key(row.get("model_name"), row.get("model_type"))
            if self.uses_included_sources:
                if source_key not in self.included_source_keys:
                    continue
            elif self.included_model_types and row.get("model_type") not in self.included_model_types:
                continue

            filtered.append(
                {
                    "model_name": row.get("model_name"),
                    "model_type": row.get("model_type"),
                    "tags": row.get("tags", []),
                    "confidence_scores": row.get("confidence_scores"),
                    "raw_output_summary": (row.get("raw_output") or "")[:1500],
                    "interrogated_at": row.get("interrogated_at"),
                }
            )
        return filtered

    def build_included_transcripts(self, file_hash: str) -> List[Dict[str, Any]]:
        """Build prior inquiry transcript context for a single image."""
        if not self.include_prior_transcripts:
            return []
        history = self.database.get_multimodal_history(
            image_hash=file_hash,
            model_name=self.interrogator.model_name,
        )
        builder = getattr(self.interrogator, "build_transcript_context", None)
        if callable(builder):
            return builder(history)
        return history

    def build_cache_identity(
        self,
        included_tables: List[Dict[str, Any]],
        included_transcripts: Optional[List[Dict[str, Any]]] = None,
        sidecar_tags: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Build a deterministic exact-match cache key and its metadata."""
        normalized_config = self.normalize_cache_config(self.interrogator.get_config())
        context_digest = self.stable_digest(
            {
                "tables": included_tables,
                "transcripts": included_transcripts or [],
                "sidecar_tags": sidecar_tags or [],
            }
        )
        metadata: Dict[str, Any] = {
            "cache_version": self.CACHE_VERSION,
            "prompt_builder_version": self.PROMPT_BUILDER_VERSION,
            "model_name": self.interrogator.model_name,
            "model_type": self.interrogator.get_model_type(),
            "llama_config": normalized_config,
            "temperature": normalized_config.get("temperature"),
            "task": self.task,
            "prompt": self.prompt,
            "include_prior_tables": self.include_prior_tables,
            "included_source_keys": sorted(self.included_source_keys),
            "included_model_types": sorted(self.included_model_types),
            "context_tables": included_tables,
            "context_transcripts": included_transcripts or [],
            "sidecar_tags": sidecar_tags or [],
            "context_digest": context_digest,
        }
        return self.stable_digest(metadata), metadata

    @staticmethod
    def stable_digest(payload: Any) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def normalize_cache_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only llama settings that can affect model output."""
        if not isinstance(config, dict):
            return {}

        relevant_keys = (
            "llama_binary_path",
            "llama_model_path",
            "llama_mmproj_path",
            "ctx_size",
            "gpu_layers",
            "temperature",
            "max_tokens",
            "server_host",
            "server_port",
        )
        normalized: Dict[str, Any] = {}
        for key in relevant_keys:
            if key not in config:
                continue
            value = config.get(key)
            if isinstance(value, Path):
                value = str(value)
            normalized[key] = value
        # Reasoning settings change the answer too. Each is keyed only when it
        # departs from the default, so entries cached before these were part
        # of the key keep matching.
        if config.get("disable_reasoning"):
            normalized["disable_reasoning"] = True
        try:
            budget = int(config.get("reasoning_budget", -1))
        except (TypeError, ValueError):
            budget = -1
        if budget >= 0:
            normalized["reasoning_budget"] = budget
        if config.get("reasoning_effort"):
            normalized["reasoning_effort"] = str(config["reasoning_effort"])
        # Unguarded runs can end in a loop's junk; keep them apart from the
        # guarded (default) results so turning the guard back on never serves it.
        if config.get("repetition_guard") is False:
            normalized["repetition_guard"] = False
        return normalized
