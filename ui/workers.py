"""Worker threads for background processing in PyQt6.

The batch loops themselves live in core.pipelines so the Electron bridge runs
the same code; the workers here only move them onto a QThread and turn their
callbacks into signals.
"""

from PyQt6.QtCore import QThread, Qt, QSize, pyqtSignal
from PyQt6.QtGui import QImage, QImageReader
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from core import (
    InterrogationDatabase, FileManager, TagFilterSettings, ProvisionConfig
)
from core.base_interrogator import BaseInterrogator
from core.pipelines import (
    InterrogationBatchRunner,
    MultimodalBatchRunner,
    StreamRelay,
    collect_context_sources,
    context_source_key,
    scan_image_directory,
)
from ui.thumbnail_cache import ThumbnailCache

# StreamRelay and context_source_key are imported for callers that still
# reach them through this module.

# Shared by the gallery widget and the thumbnail worker.
thumbnail_cache = ThumbnailCache()


def decode_thumbnail(image_path: str, target_size: QSize,
                     use_cache: bool = True) -> Optional[QImage]:
    """Decode an image to thumbnail size, using the on-disk cache.

    Returns None when the file cannot be read. QImage is used rather than
    QPixmap so this is safe to call from a worker thread; only the GUI thread
    may build the QPixmap.
    """
    if use_cache:
        cached = thumbnail_cache.get(image_path, target_size)
        if cached is not None:
            return cached

    image = _decode_thumbnail_uncached(image_path, target_size)
    if image is not None and use_cache:
        thumbnail_cache.store(image_path, target_size, image)
    return image


def _decode_thumbnail_uncached(image_path: str, target_size: QSize) -> Optional[QImage]:
    """Decode an image straight to thumbnail size, bypassing the cache."""
    reader = QImageReader(image_path)
    scaled_size = reader.size()
    decoded_scaled = scaled_size.isValid()
    if decoded_scaled:
        scaled_size.scale(target_size, Qt.AspectRatioMode.KeepAspectRatio)
        reader.setScaledSize(scaled_size)

    image = reader.read()
    if image.isNull():
        return None

    if not decoded_scaled:
        # Formats that cannot report their size up front still need scaling.
        image = image.scaled(
            target_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
    return image


class ThumbnailLoadWorker(QThread):
    """Worker thread for decoding gallery thumbnails without blocking the UI."""

    # Signals
    thumbnails_ready = pyqtSignal(list)  # [(image_path, QImage or None), ...]
    progress = pyqtSignal(int, int)      # decoded, total

    # Emitting one signal per image floods the event loop on large galleries.
    BATCH_SIZE = 24

    def __init__(self, image_paths: List[str], target_size: QSize):
        super().__init__()
        self.image_paths = list(image_paths)
        self.target_size = QSize(target_size)
        self.is_cancelled = False

    def cancel(self):
        """Cancel the operation."""
        self.is_cancelled = True

    def run(self):
        """Decode thumbnails, emitting them in batches.

        Unreadable files are reported with a None image so the gallery can drop
        them, matching the inline path which never adds them at all.
        """
        total = len(self.image_paths)
        batch: List[Tuple[str, Optional[QImage]]] = []
        decoded = 0
        last_percent = -1

        for image_path in self.image_paths:
            if self.is_cancelled:
                return

            batch.append((image_path, decode_thumbnail(image_path, self.target_size)))
            decoded += 1

            if len(batch) >= self.BATCH_SIZE:
                self.thumbnails_ready.emit(batch)
                batch = []
                # Report only when the whole percentage moves.
                percent = int((decoded / total) * 100) if total else 100
                if percent != last_percent:
                    last_percent = percent
                    self.progress.emit(decoded, total)

        if batch:
            self.thumbnails_ready.emit(batch)
        self.progress.emit(decoded, total)


class ClipModelListWorker(QThread):
    """Worker thread for loading the list of available CLIP models.

    Only model names are needed, but reaching them imports open_clip, which
    pulls in torch and builds a tokenizer -- over a second of work that would
    otherwise run on the GUI thread at startup.
    """

    # Signals
    completed = pyqtSignal(dict)  # categorized model lists
    failed = pyqtSignal(str)      # error message

    def run(self):
        """Import open_clip and categorize the available models."""
        try:
            from core.clip_model_loader import get_categorized_models
            self.completed.emit(get_categorized_models())
        except Exception as e:
            self.failed.emit(str(e))


class BatchContextScanWorker(QThread):
    """Worker thread for finding prior interrogation sources across a queue.

    Each image has to be hashed, which means reading the whole file, and then
    looked up in the database. Running that inline blocks the GUI for tens of
    seconds on a large directory.
    """

    # Signals
    completed = pyqtSignal(list)  # sorted source dicts
    progress = pyqtSignal(int, int)  # scanned, total

    def __init__(self, image_paths: List[str], database: InterrogationDatabase):
        super().__init__()
        self.image_paths = list(image_paths)
        self.database = database
        self.is_cancelled = False

    def cancel(self):
        """Cancel the operation."""
        self.is_cancelled = True

    def run(self):
        """Collect prior-result sources for every queued image."""
        sources = collect_context_sources(
            self.image_paths,
            self.database,
            is_cancelled=lambda: self.is_cancelled,
            on_progress=self.progress.emit,
        )
        if sources is None:
            return
        self.completed.emit(sources)


class InterrogationWorker(QThread):
    """Worker thread for batch image interrogation."""
    
    # Signals
    progress = pyqtSignal(int, int, str)  # current, total, message
    result = pyqtSignal(str, dict)  # image_path, results
    error = pyqtSignal(str, str)  # image_path, error_message
    finished = pyqtSignal()
    
    def __init__(self, image_paths: List[Path], interrogator: BaseInterrogator,
                 database: InterrogationDatabase, write_files: bool = True,
                 overwrite_files: bool = False, tag_filters: Optional[TagFilterSettings] = None):
        super().__init__()
        self.image_paths = image_paths
        self.interrogator = interrogator
        self.database = database
        self.write_files = write_files
        self.overwrite_files = overwrite_files
        self.tag_filters = tag_filters
        self._runner = InterrogationBatchRunner(
            image_paths,
            interrogator,
            database,
            write_files=write_files,
            overwrite_files=overwrite_files,
            tag_filters=tag_filters,
            on_progress=self.progress.emit,
            on_result=lambda path, results, _meta: self.result.emit(path, results),
            on_error=self.error.emit,
        )

    @property
    def is_cancelled(self) -> bool:
        return self._runner.is_cancelled

    def cancel(self):
        """Cancel the operation."""
        self._runner.cancel()

    def run(self):
        """Execute batch interrogation."""
        self._runner.run()
        self.finished.emit()

    def _should_use_cached_result(self, cached: Dict[str, Any]) -> bool:
        """Return False for stale empty ONNX tagger rows that should be regenerated."""
        return self._runner.should_use_cached_result(cached)


class SingleInquiryWorker(QThread):
    """Runs one multimodal inquiry off the GUI thread, streaming as it goes."""

    stream_delta = pyqtSignal(str)  # accumulated readable response text
    completed = pyqtSignal(dict)  # results
    failed = pyqtSignal(str)  # error message

    def __init__(
        self,
        interrogator: BaseInterrogator,
        image_path: str,
        task: str,
        prompt: str,
        session_key: Optional[str],
        included_tables: Optional[List[Dict[str, Any]]] = None,
        included_transcripts: Optional[List[Dict[str, Any]]] = None,
        sidecar_tags: Optional[List[str]] = None,
    ):
        super().__init__()
        self.interrogator = interrogator
        self.image_path = image_path
        self.task = task
        self.prompt = prompt
        self.session_key = session_key
        self.included_tables = included_tables or []
        self.included_transcripts = included_transcripts or []
        self.sidecar_tags = sidecar_tags or []

    def run(self):
        relay = StreamRelay(self.stream_delta.emit)
        try:
            results = self.interrogator.interrogate(
                self.image_path,
                task=self.task,
                prompt=self.prompt,
                session_key=self.session_key,
                keep_context=True,
                included_tables=self.included_tables,
                included_transcripts=self.included_transcripts,
                sidecar_tags=self.sidecar_tags,
                on_stream_delta=relay,
            )
        except Exception as exc:
            self.failed.emit(str(exc).strip() or repr(exc))
            return
        relay.flush()
        self.completed.emit(results)


class MultimodalInterrogationWorker(QThread):
    """Worker thread for llama.cpp multimodal batch interrogation."""

    CACHE_VERSION = MultimodalBatchRunner.CACHE_VERSION
    PROMPT_BUILDER_VERSION = MultimodalBatchRunner.PROMPT_BUILDER_VERSION
    VALID_TXT_OUTPUT_MODES = MultimodalBatchRunner.VALID_TXT_OUTPUT_MODES

    # Signals
    progress = pyqtSignal(int, int, str)  # current, total, message
    turn_started = pyqtSignal(str, dict)  # image_path, pending turn
    stream_delta = pyqtSignal(str, str)  # image_path, accumulated response text
    result = pyqtSignal(str, dict)  # image_path, results
    error = pyqtSignal(str, str)  # image_path, error_message
    finished = pyqtSignal(bool)  # was_cancelled

    def __init__(
        self,
        image_paths: List[Path],
        interrogator: BaseInterrogator,
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
    ):
        super().__init__()
        self._runner = MultimodalBatchRunner(
            image_paths=image_paths,
            interrogator=interrogator,
            database=database,
            task=task,
            prompt=prompt,
            write_files=write_files,
            overwrite_files=overwrite_files,
            tag_filters=tag_filters,
            include_prior_tables=include_prior_tables,
            include_prior_transcripts=include_prior_transcripts,
            included_model_types=included_model_types,
            included_sources=included_sources,
            carry_context_across_batch=carry_context_across_batch,
            use_cache=use_cache,
            txt_output_mode=txt_output_mode,
            on_progress=self.progress.emit,
            on_turn_started=self.turn_started.emit,
            on_stream_delta=self.stream_delta.emit,
            on_result=lambda path, results, _meta: self.result.emit(path, results),
            on_error=self.error.emit,
        )
        runner = self._runner
        self.image_paths = runner.image_paths
        self.interrogator = runner.interrogator
        self.database = runner.database
        self.task = runner.task
        self.prompt = runner.prompt
        self.txt_output_mode = runner.txt_output_mode
        self.write_files = runner.write_files
        self.overwrite_files = runner.overwrite_files
        self.tag_filters = runner.tag_filters
        self.include_prior_tables = runner.include_prior_tables
        self.include_prior_transcripts = runner.include_prior_transcripts
        self.included_model_types = runner.included_model_types
        self.uses_included_sources = runner.uses_included_sources
        self.included_source_keys = runner.included_source_keys
        self.carry_context_across_batch = runner.carry_context_across_batch
        self.use_cache = runner.use_cache

    @property
    def is_cancelled(self) -> bool:
        return self._runner.is_cancelled

    @property
    def was_cancelled(self) -> bool:
        return self._runner.was_cancelled

    def cancel(self):
        """Cancel the operation."""
        self._runner.cancel()

    def run(self):
        """Execute multimodal batch interrogation."""
        self.finished.emit(self._runner.run())

    @classmethod
    def _resolve_txt_output_mode(
        cls,
        write_files: bool,
        overwrite_files: bool,
        txt_output_mode: Optional[str],
    ) -> str:
        """Normalize text-output settings to the three UI modes."""
        return MultimodalBatchRunner.resolve_txt_output_mode(
            write_files, overwrite_files, txt_output_mode
        )

    def _build_included_tables(self, file_hash: str) -> List[Dict[str, Any]]:
        """Build prior interrogation context tables for a single image."""
        return self._runner.build_included_tables(file_hash)

    def _build_included_transcripts(self, file_hash: str) -> List[Dict[str, Any]]:
        """Build prior inquiry transcript context for a single image."""
        return self._runner.build_included_transcripts(file_hash)

    @staticmethod
    def _source_key(model_name: Optional[str], model_type: Optional[str]) -> str:
        """Stable key for matching selected batch context sources."""
        return context_source_key(model_name, model_type)

    def _build_cache_identity(
        self,
        included_tables: List[Dict[str, Any]],
        included_transcripts: Optional[List[Dict[str, Any]]] = None,
        sidecar_tags: Optional[List[str]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Build a deterministic exact-match cache key and its metadata."""
        return self._runner.build_cache_identity(included_tables, included_transcripts, sidecar_tags)

    @classmethod
    def _stable_digest(cls, payload: Any) -> str:
        return MultimodalBatchRunner.stable_digest(payload)

    @staticmethod
    def _normalize_cache_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only llama settings that can affect model output."""
        return MultimodalBatchRunner.normalize_cache_config(config)


class OrganizationWorker(QThread):
    """Worker thread for organizing images by tags."""
    
    # Signals
    progress = pyqtSignal(int, int, str)  # current, total, message
    moved = pyqtSignal(str, str)  # source_path, destination_path
    error = pyqtSignal(str, str)  # image_path, error_message
    finished = pyqtSignal(int)  # total_moved
    
    def __init__(self, image_paths: List[Path], tag_criteria: List[str],
                 target_subdir: str, match_mode: str = 'any', move_text: bool = True,
                 target_root_dir: Optional[Path] = None):
        super().__init__()
        self.image_paths = image_paths
        self.tag_criteria = tag_criteria
        self.target_subdir = target_subdir
        self.match_mode = match_mode
        self.move_text = move_text
        self.target_root_dir = target_root_dir
        self.is_cancelled = False
    
    def cancel(self):
        """Cancel the operation."""
        self.is_cancelled = True
    
    def run(self):
        """Execute batch organization."""
        total = len(self.image_paths)
        moved_count = 0
        
        for idx, image_path in enumerate(self.image_paths):
            if self.is_cancelled:
                break
            
            try:
                self.progress.emit(idx + 1, total, f"Checking: {image_path.name}")
                
                # Try to organize
                was_moved = FileManager.organize_by_tags(
                    image_path,
                    self.tag_criteria,
                    self.target_subdir,
                    self.move_text,
                    self.match_mode,
                    self.target_root_dir
                )
                
                if was_moved:
                    moved_count += 1
                    destination_root = self.target_root_dir if self.target_root_dir else image_path.parent
                    dest_path = str(destination_root / self.target_subdir / image_path.name)
                    self.moved.emit(str(image_path), dest_path)
                
            except Exception as e:
                self.error.emit(str(image_path), str(e))
        
        self.finished.emit(moved_count)


class DirectoryLoadWorker(QThread):
    """Worker thread for scanning directories for images without blocking UI."""

    # Signals
    progress = pyqtSignal(int, str)   # count_so_far, current_file
    finished = pyqtSignal(list)        # complete list of image paths (strings)
    error = pyqtSignal(str)            # error_message

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}

    def __init__(self, directory: str, recursive: bool = False):
        super().__init__()
        self.directory = directory
        self.recursive = recursive
        self.is_cancelled = False

    def cancel(self):
        """Cancel the directory scan."""
        self.is_cancelled = True

    def run(self):
        """Execute directory scan using os.scandir for cancellable iteration."""
        try:
            image_paths = scan_image_directory(
                self.directory,
                recursive=self.recursive,
                is_cancelled=lambda: self.is_cancelled,
                on_progress=self.progress.emit,
            )
            self.finished.emit(image_paths or [])
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit([])


class DatabaseQueueWorker(QThread):
    """Worker thread for processing queued database operations."""

    # Signals
    progress = pyqtSignal(int, int, str)  # current, total, message
    operation_completed = pyqtSignal(str)  # operation_id
    operation_failed = pyqtSignal(str, str)  # operation_id, error_message
    finished = pyqtSignal(int, int)  # success_count, failed_count

    def __init__(self, database: InterrogationDatabase):
        super().__init__()
        self.database = database
        self.is_cancelled = False

    def cancel(self):
        """Cancel the operation."""
        self.is_cancelled = True

    def run(self):
        """Execute queued operations processing."""
        status = self.database.get_queue_status()
        operations = status.get('operations', [])
        total = len(operations)

        if total == 0:
            self.finished.emit(0, 0)
            return

        success_count = 0
        failed_count = 0

        for idx, op in enumerate(operations):
            if self.is_cancelled:
                break

            op_id = op.get('id', '')
            op_name = op.get('operation', 'unknown')
            summary = op.get('summary', op_name)

            self.progress.emit(idx + 1, total, f"Processing: {summary}")

            try:
                # Get the pending operations and find this one
                pending = self.database._operation_queue.get_pending_operations()
                matching_op = next((p for p in pending if p.id == op_id), None)

                if matching_op:
                    self.database._execute_queued_operation(
                        matching_op.id,
                        matching_op.operation,
                        matching_op.params
                    )
                    self.database._operation_queue.mark_completed(op_id)
                    success_count += 1
                    self.operation_completed.emit(op_id)
                else:
                    # Operation no longer in queue
                    failed_count += 1

            except Exception as e:
                self.database._operation_queue.mark_failed(op_id, str(e))
                failed_count += 1
                self.operation_failed.emit(op_id, str(e))

        self.finished.emit(success_count, failed_count)


class CacheScanWorker(QThread):
    """Worker thread for scanning model caches in background."""

    # Signals
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(dict)  # {type: [ModelCacheInfo]}

    def __init__(self, cache_manager):
        """Initialize the cache scan worker.

        Args:
            cache_manager: ModelCacheManager instance
        """
        super().__init__()
        self.cache_manager = cache_manager
        self.is_cancelled = False

    def cancel(self):
        """Cancel the scan operation."""
        self.is_cancelled = True

    def run(self):
        """Execute cache scanning."""
        try:
            # Get all models and scan their cache status
            all_models = self.cache_manager.get_all_models()

            # Count total models for progress
            total = sum(len(models) for models in all_models.values())
            current = 0

            # Emit progress for each model (cache manager already scanned them)
            for model_type, models in all_models.items():
                for model in models:
                    if self.is_cancelled:
                        self.finished.emit({})
                        return

                    current += 1
                    status = "Cached" if model.is_cached else "Not cached"
                    self.progress.emit(current, total, f"{model.display_name}: {status}")

            self.finished.emit(all_models)

        except Exception as e:
            print(f"Error scanning caches: {e}")
            self.finished.emit({})


class CacheDeleteWorker(QThread):
    """Worker thread for deleting model caches in background."""

    # Signals
    progress = pyqtSignal(int, int, str)  # current, total, message
    finished = pyqtSignal(int, int)  # success_count, error_count

    def __init__(self, cache_manager, model_ids: List[str], delete_tensorrt: bool = False):
        """Initialize the cache delete worker.

        Args:
            cache_manager: ModelCacheManager instance
            model_ids: List of model IDs to delete
            delete_tensorrt: Whether to also delete TensorRT engines
        """
        super().__init__()
        self.cache_manager = cache_manager
        self.model_ids = model_ids
        self.delete_tensorrt = delete_tensorrt
        self.is_cancelled = False

    def cancel(self):
        """Cancel the delete operation."""
        self.is_cancelled = True

    def run(self):
        """Execute cache deletion."""
        total = len(self.model_ids)
        success_count = 0
        error_count = 0

        for idx, model_id in enumerate(self.model_ids):
            if self.is_cancelled:
                break

            self.progress.emit(idx + 1, total, f"Deleting: {model_id}")

            try:
                # Delete HuggingFace cache
                if self.cache_manager.delete_hf_model_cache(model_id):
                    success_count += 1
                else:
                    # Model might not have been cached
                    pass

                # Delete TensorRT engine if requested
                if self.delete_tensorrt:
                    self.cache_manager.delete_tensorrt_engine(model_id)

            except Exception as e:
                print(f"Error deleting cache for {model_id}: {e}")
                error_count += 1

        self.finished.emit(success_count, error_count)


class TensorRTConversionWorker(QThread):
    """Worker thread for converting ONNX models to TensorRT engines."""

    # Signals
    progress = pyqtSignal(int, int, str)  # current, total, message
    conversion_complete = pyqtSignal(str)  # model_id
    conversion_failed = pyqtSignal(str, str)  # model_id, error_message
    finished = pyqtSignal(int, int)  # success_count, error_count

    def __init__(self, cache_manager, model_ids: List[str], provider_settings=None):
        """Initialize the TensorRT conversion worker.

        Args:
            cache_manager: ModelCacheManager instance
            model_ids: List of model IDs to convert
            provider_settings: Optional ONNXProviderSettings for provider options
        """
        super().__init__()
        self.cache_manager = cache_manager
        self.model_ids = model_ids
        self.provider_settings = provider_settings
        self.is_cancelled = False

    def cancel(self):
        """Cancel the conversion operation."""
        self.is_cancelled = True

    def run(self):
        """Execute TensorRT conversion."""
        try:
            import onnxruntime as ort
        except ImportError:
            self.finished.emit(0, len(self.model_ids))
            return

        # Check if TensorRT provider is available
        available_providers = ort.get_available_providers()
        if 'TensorrtExecutionProvider' not in available_providers:
            for model_id in self.model_ids:
                self.conversion_failed.emit(model_id, "TensorRT provider not available")
            self.finished.emit(0, len(self.model_ids))
            return

        total = len(self.model_ids)
        success_count = 0
        error_count = 0

        # Get TensorRT cache directory
        from core.model_cache import TRT_CACHE_DIR
        TRT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        for idx, model_id in enumerate(self.model_ids):
            if self.is_cancelled:
                break

            self.progress.emit(idx + 1, total, f"Converting: {model_id}")

            try:
                # Get ONNX model path
                onnx_path = self.cache_manager.get_onnx_model_path(model_id)

                if onnx_path is None or not onnx_path.exists():
                    self.conversion_failed.emit(model_id, "ONNX model not found (not cached)")
                    error_count += 1
                    continue

                # Create TensorRT session to trigger engine compilation
                providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider']
                provider_options = [
                    {
                        'trt_engine_cache_enable': True,
                        'trt_engine_cache_path': str(TRT_CACHE_DIR),
                        'trt_fp16_enable': True,
                        'trt_max_workspace_size': 2147483648,  # 2GB
                    },
                    {}  # CUDA provider options (empty)
                ]

                # This will create the TensorRT engine if it doesn't exist
                session = ort.InferenceSession(
                    str(onnx_path),
                    providers=providers,
                    provider_options=provider_options
                )

                # Run a dummy inference to ensure engine is compiled
                # Get input shape from the model
                input_info = session.get_inputs()[0]
                input_name = input_info.name
                input_shape = input_info.shape

                # Create dummy input (typically images are [1, 3, H, W] or [1, H, W, 3])
                import numpy as np
                # Handle dynamic dimensions
                resolved_shape = []
                for dim in input_shape:
                    if isinstance(dim, int):
                        resolved_shape.append(dim)
                    else:
                        # Dynamic dimension, use common sizes
                        resolved_shape.append(448)  # Common WD tagger size

                dummy_input = np.zeros(resolved_shape, dtype=np.float32)
                session.run(None, {input_name: dummy_input})

                self.conversion_complete.emit(model_id)
                success_count += 1

            except Exception as e:
                error_msg = str(e)
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."
                self.conversion_failed.emit(model_id, error_msg)
                error_count += 1

        self.finished.emit(success_count, error_count)


class LlamaProvisionWorker(QThread):
    """Worker thread for installing or building llama-server.

    A source build runs for minutes, so the plan executes here and reports each
    step back to the UI. Output is streamed rather than buffered: the CUDA and
    CMake errors worth reading arrive long before the process exits.
    """

    # Signals
    progress = pyqtSignal(int, int, str, float)  # step, total_steps, label, fraction
    log = pyqtSignal(str, bool)  # line, is_stderr
    finished = pyqtSignal(str, str)  # binary_path ("" on failure), error_message

    def __init__(self, config: ProvisionConfig):
        """Initialize the provisioning worker.

        Args:
            config: ProvisionConfig describing the desired runtime.
        """
        super().__init__()
        self.config = config
        self.is_cancelled = False
        self.build_log_path: Optional[str] = None
        # Non-empty when the ladder settled for a weaker backend than requested.
        self.target_mismatch = ""

    def cancel(self):
        """Request cancellation; the current step stops at its next checkpoint."""
        self.is_cancelled = True

    def run(self):
        """Execute the install plan."""
        from core.llama_provisioner import LlamaProvisioner, ProvisionCancelled

        provisioner = None
        try:
            provisioner = LlamaProvisioner(
                config=self.config,
                log_sink=lambda line, is_stderr: self.log.emit(line, is_stderr),
                progress_sink=lambda p: self.progress.emit(p.step, p.total, p.label, p.fraction),
                cancel_check=lambda: self.is_cancelled,
            )
            binary = provisioner.ensure_runtime()
            self.build_log_path = str(provisioner.build_log_path or "")
            self.target_mismatch = provisioner.target_mismatch
            self.finished.emit(str(binary), "")
        except ProvisionCancelled:
            self.build_log_path = str(provisioner.build_log_path or "") if provisioner else ""
            self.finished.emit("", "Provisioning cancelled.")
        except Exception as exc:
            self.build_log_path = str(provisioner.build_log_path or "") if provisioner else ""
            self.finished.emit("", str(exc))


class LlamaUpdateCheckWorker(QThread):
    """Worker thread for checking whether a newer llama.cpp runtime exists.

    The check hits the GitHub API, so it belongs off the UI thread even though
    it is far quicker than provisioning itself.
    """

    # Signals
    completed = pyqtSignal(object)  # UpdateStatus

    def __init__(self, config: ProvisionConfig, installed_version: str):
        """Initialize the update check worker.

        Args:
            config: ProvisionConfig describing the runtime to check.
            installed_version: Version string recorded for the current install.
        """
        super().__init__()
        self.config = config
        self.installed_version = installed_version

    def run(self):
        """Query upstream and report what an update would involve."""
        from core.llama_provisioner import UpdateStatus, check_for_update

        try:
            self.completed.emit(check_for_update(self.config, self.installed_version))
        except Exception as exc:
            self.completed.emit(
                UpdateStatus(current_version=self.installed_version, error=str(exc))
            )


class LlamaHealthCheckWorker(QThread):
    """Worker thread for starting a managed runtime validation probe."""

    completed = pyqtSignal(str)  # Empty string means healthy.

    def __init__(self, binary_path: str):
        super().__init__()
        self.binary_path = binary_path

    def run(self):
        from core.llama_provisioner import ProvisionError, validate_server

        try:
            validate_server(Path(self.binary_path))
        except (ProvisionError, OSError) as exc:
            self.completed.emit(str(exc))
            return
        except Exception as exc:
            self.completed.emit(f"Runtime health check failed: {exc}")
            return
        self.completed.emit("")
