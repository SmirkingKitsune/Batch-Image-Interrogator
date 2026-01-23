"""Worker threads for background processing in PyQt6."""

import os
from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from core import (
    InterrogationDatabase, hash_image_content, get_image_metadata,
    FileManager, TagFilterSettings, DatabaseBusyError, DatabaseQueuedError
)
from core.base_interrogator import BaseInterrogator


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
        self.is_cancelled = False
    
    def cancel(self):
        """Cancel the operation."""
        self.is_cancelled = True
    
    def run(self):
        """Execute batch interrogation."""
        total = len(self.image_paths)
        model_id = None
        
        # Register model once
        try:
            model_id = self.database.register_model(
                self.interrogator.model_name,
                self.interrogator.get_model_type(),
                config=self.interrogator.get_config()
            )
        except Exception as e:
            self.error.emit("", f"Failed to register model: {e}")
            self.finished.emit()
            return
        
        for idx, image_path in enumerate(self.image_paths):
            if self.is_cancelled:
                break
            
            try:
                image_path_str = str(image_path)
                self.progress.emit(idx + 1, total, f"Processing: {image_path.name}")
                
                # Hash the image
                file_hash = hash_image_content(image_path_str)
                
                # Check cache first
                cached = self.database.get_interrogation(
                    file_hash, 
                    self.interrogator.model_name
                )
                
                if cached:
                    # Use cached results
                    results = cached
                    self.progress.emit(
                        idx + 1, total, 
                        f"Using cached: {image_path.name}"
                    )
                else:
                    # Interrogate image
                    results = self.interrogator.interrogate(image_path_str)

                    # Register image and save to database
                    metadata = get_image_metadata(image_path_str)
                    try:
                        image_id = self.database.register_image(
                            image_path_str,
                            file_hash,
                            metadata['width'],
                            metadata['height'],
                            metadata['file_size']
                        )
                    except DatabaseBusyError as e:
                        # User chose to abort - can't save without image_id
                        self.error.emit(image_path_str, f"Database busy: {e}")
                        continue

                    try:
                        self.database.save_interrogation(
                            image_id,
                            model_id,
                            results['tags'],
                            results.get('confidence_scores'),
                            results.get('raw_output')
                        )
                    except DatabaseQueuedError:
                        # Operation was queued for later - continue processing
                        pass
                    except DatabaseBusyError as e:
                        # User chose to abort this operation
                        self.error.emit(image_path_str, f"Database busy: {e}")
                        continue

                # Write to text file if requested
                if self.write_files:
                    # Apply tag filters if configured
                    tags_to_write = results['tags']
                    if self.tag_filters:
                        confidence_scores = results.get('confidence_scores')

                        # Use confidence-based filtering if scores are available
                        if confidence_scores is not None:
                            # Get threshold from interrogator config
                            threshold = self.interrogator.get_config().get('threshold', 0.35)
                            tags_to_write, _ = self.tag_filters.filter_tags_with_confidence(
                                tags_to_write,
                                confidence_scores,
                                threshold
                            )
                        else:
                            # No confidence scores (CLIP), use simple filtering
                            tags_to_write = self.tag_filters.apply_filters(tags_to_write)

                    # Check if file exists and handle overwrite
                    txt_path = FileManager.get_text_file_path(image_path)
                    if not txt_path.exists() or self.overwrite_files:
                        FileManager.write_tags_to_file(
                            image_path,
                            tags_to_write,
                            overwrite=self.overwrite_files
                        )
                
                # Emit result
                self.result.emit(image_path_str, results)
                
            except Exception as e:
                self.error.emit(str(image_path), str(e))
        
        self.finished.emit()


class OrganizationWorker(QThread):
    """Worker thread for organizing images by tags."""
    
    # Signals
    progress = pyqtSignal(int, int, str)  # current, total, message
    moved = pyqtSignal(str, str)  # source_path, destination_path
    error = pyqtSignal(str, str)  # image_path, error_message
    finished = pyqtSignal(int)  # total_moved
    
    def __init__(self, image_paths: List[Path], tag_criteria: List[str],
                 target_subdir: str, match_mode: str = 'any', move_text: bool = True):
        super().__init__()
        self.image_paths = image_paths
        self.tag_criteria = tag_criteria
        self.target_subdir = target_subdir
        self.match_mode = match_mode
        self.move_text = move_text
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
                    self.match_mode
                )
                
                if was_moved:
                    moved_count += 1
                    dest_path = str(image_path.parent / self.target_subdir / image_path.name)
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
            dir_path = Path(self.directory)
            if not dir_path.exists() or not dir_path.is_dir():
                self.error.emit(f"Invalid directory: {self.directory}")
                self.finished.emit([])
                return

            image_paths = []
            count = 0

            if self.recursive:
                # Use os.walk for recursive scanning (cancellable)
                for root, dirs, files in os.walk(self.directory):
                    if self.is_cancelled:
                        break

                    for filename in files:
                        if self.is_cancelled:
                            break

                        ext = os.path.splitext(filename)[1].lower()
                        if ext in self.SUPPORTED_EXTENSIONS:
                            full_path = os.path.join(root, filename)
                            image_paths.append(full_path)
                            count += 1

                            # Emit progress every 10 files for responsive feedback
                            if count % 10 == 0:
                                self.progress.emit(count, filename)
            else:
                # Use os.scandir for non-recursive scanning (faster than glob)
                with os.scandir(self.directory) as entries:
                    for entry in entries:
                        if self.is_cancelled:
                            break

                        if entry.is_file():
                            ext = os.path.splitext(entry.name)[1].lower()
                            if ext in self.SUPPORTED_EXTENSIONS:
                                image_paths.append(entry.path)
                                count += 1

                                # Emit progress every 10 files for responsive feedback
                                if count % 10 == 0:
                                    self.progress.emit(count, entry.name)

            if self.is_cancelled:
                self.finished.emit([])
                return

            # Sort and deduplicate
            image_paths = sorted(set(image_paths))

            # Emit final progress
            self.progress.emit(len(image_paths), "")
            self.finished.emit(image_paths)

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
