"""Model cache management for HuggingFace and TensorRT cached models."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Cache directory paths
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
TRT_CACHE_DIR = Path.home() / ".cache" / "tensorrt_engines"
TORCH_CACHE_DIR = Path.home() / ".cache" / "torch" / "hub"


@dataclass
class ModelCacheInfo:
    """Data class for model cache information."""
    model_id: str           # e.g., "SmilingWolf/wd-v1-4-moat-tagger-v2"
    model_type: str         # "WD", "Camie", "CLIP"
    display_name: str       # Human-readable name
    is_cached: bool
    cache_path: Optional[Path]
    cache_size_bytes: int
    has_tensorrt_engine: bool
    tensorrt_engine_size: int
    tensorrt_engine_path: Optional[Path] = None


class ModelCacheManager:
    """Manages model cache detection, deletion, and TensorRT conversion."""

    # Known WD Tagger models (from ui/dialogs.py)
    WD_MODELS = [
        'SmilingWolf/wd-v1-4-moat-tagger-v2',
        'SmilingWolf/wd-v1-4-vit-tagger-v2',
        'SmilingWolf/wd-v1-4-vit-tagger',
        'SmilingWolf/wd-v1-4-convnext-tagger-v2',
        'SmilingWolf/wd-v1-4-convnext-tagger',
        'SmilingWolf/wd-v1-4-convnextv2-tagger-v2',
        'SmilingWolf/wd-v1-4-swinv2-tagger-v2',
        'SmilingWolf/wd-vit-tagger-v3',
        'SmilingWolf/wd-vit-large-tagger-v3',
        'SmilingWolf/wd-convnext-tagger-v3',
        'SmilingWolf/wd-swinv2-tagger-v3',
        'SmilingWolf/wd-eva02-large-tagger-v3',
    ]

    # Known Camie models (from interrogators/camie_interrogator.py)
    CAMIE_MODELS = [
        'Camais03/camie-tagger',
        'Camais03/camie-tagger-v2',
    ]

    def __init__(self):
        """Initialize the model cache manager."""
        self._ensure_cache_dirs()

    def _ensure_cache_dirs(self):
        """Ensure cache directories exist."""
        TRT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _hf_repo_to_cache_dir(self, repo_id: str) -> Path:
        """Convert HuggingFace repo ID to cache directory path.

        Example: "SmilingWolf/wd-v1-4-moat-tagger-v2" ->
                 "models--SmilingWolf--wd-v1-4-moat-tagger-v2"
        """
        cache_name = f"models--{repo_id.replace('/', '--')}"
        return HF_CACHE_DIR / cache_name

    def _get_directory_size(self, path: Path) -> int:
        """Get total size of a directory in bytes."""
        if not path.exists():
            return 0

        total_size = 0
        try:
            for item in path.rglob('*'):
                if item.is_file():
                    try:
                        total_size += item.stat().st_size
                    except (OSError, PermissionError):
                        pass
        except (OSError, PermissionError):
            pass

        return total_size

    def _model_id_to_trt_name(self, model_id: str) -> str:
        """Convert model ID to TensorRT engine filename base.

        Example: "SmilingWolf/wd-v1-4-moat-tagger-v2" ->
                 "SmilingWolf_wd-v1-4-moat-tagger-v2"
        """
        return model_id.replace('/', '_')

    def is_hf_model_cached(self, repo_id: str) -> Tuple[bool, Optional[Path], int]:
        """Check if a HuggingFace model is cached.

        Returns:
            Tuple of (is_cached, cache_path, cache_size_bytes)
        """
        cache_dir = self._hf_repo_to_cache_dir(repo_id)

        if cache_dir.exists() and cache_dir.is_dir():
            # Check if there's actual content (not just metadata)
            snapshots_dir = cache_dir / "snapshots"
            if snapshots_dir.exists():
                # Check if any snapshots have content
                for snapshot in snapshots_dir.iterdir():
                    if snapshot.is_dir() and any(snapshot.iterdir()):
                        size = self._get_directory_size(cache_dir)
                        return True, cache_dir, size

        return False, None, 0

    def get_tensorrt_engine_info(self, model_id: str) -> Tuple[bool, Optional[Path], int]:
        """Check if a TensorRT engine exists for the given model.

        Returns:
            Tuple of (has_engine, engine_path, engine_size_bytes)
        """
        trt_name = self._model_id_to_trt_name(model_id)

        # Look for engine files matching the model name
        # TensorRT engines are typically named like: model_name_fp16.engine
        if TRT_CACHE_DIR.exists():
            for engine_file in TRT_CACHE_DIR.glob(f"{trt_name}*"):
                if engine_file.is_file():
                    size = engine_file.stat().st_size
                    return True, engine_file, size
                elif engine_file.is_dir():
                    # Some implementations use directories
                    size = self._get_directory_size(engine_file)
                    return True, engine_file, size

        return False, None, 0

    def get_model_info(self, model_id: str, model_type: str) -> ModelCacheInfo:
        """Get complete cache information for a model."""
        # Get HuggingFace cache info
        is_cached, cache_path, cache_size = self.is_hf_model_cached(model_id)

        # Get TensorRT engine info
        has_trt, trt_path, trt_size = self.get_tensorrt_engine_info(model_id)

        # Generate display name from model_id
        display_name = model_id.split('/')[-1] if '/' in model_id else model_id

        return ModelCacheInfo(
            model_id=model_id,
            model_type=model_type,
            display_name=display_name,
            is_cached=is_cached,
            cache_path=cache_path,
            cache_size_bytes=cache_size,
            has_tensorrt_engine=has_trt,
            tensorrt_engine_size=trt_size,
            tensorrt_engine_path=trt_path
        )

    def get_all_models(self) -> Dict[str, List[ModelCacheInfo]]:
        """Get information about all known models, grouped by type.

        Returns:
            Dictionary with keys "WD", "Camie" and lists of ModelCacheInfo
        """
        result = {
            'WD': [],
            'Camie': [],
        }

        # Get WD model info
        for model_id in self.WD_MODELS:
            info = self.get_model_info(model_id, 'WD')
            result['WD'].append(info)

        # Get Camie model info
        for model_id in self.CAMIE_MODELS:
            info = self.get_model_info(model_id, 'Camie')
            result['Camie'].append(info)

        return result

    def delete_hf_model_cache(self, repo_id: str) -> bool:
        """Delete the HuggingFace cache for a model.

        Returns:
            True if deleted successfully, False otherwise
        """
        cache_dir = self._hf_repo_to_cache_dir(repo_id)

        if cache_dir.exists() and cache_dir.is_dir():
            try:
                shutil.rmtree(cache_dir)
                return True
            except (OSError, PermissionError) as e:
                print(f"Error deleting cache for {repo_id}: {e}")
                return False

        return False

    def delete_tensorrt_engine(self, model_id: str) -> bool:
        """Delete the TensorRT engine for a model.

        Returns:
            True if deleted successfully, False otherwise
        """
        trt_name = self._model_id_to_trt_name(model_id)
        deleted = False

        if TRT_CACHE_DIR.exists():
            for engine_file in TRT_CACHE_DIR.glob(f"{trt_name}*"):
                try:
                    if engine_file.is_file():
                        engine_file.unlink()
                        deleted = True
                    elif engine_file.is_dir():
                        shutil.rmtree(engine_file)
                        deleted = True
                except (OSError, PermissionError) as e:
                    print(f"Error deleting TensorRT engine for {model_id}: {e}")

        return deleted

    def get_total_cache_size(self) -> Dict[str, int]:
        """Get total cache sizes for each cache type.

        Returns:
            Dictionary with keys 'huggingface', 'tensorrt', 'total'
        """
        hf_size = 0
        trt_size = 0

        # Calculate HuggingFace cache size for known models
        all_models = self.get_all_models()
        for model_type, models in all_models.items():
            for model in models:
                hf_size += model.cache_size_bytes
                trt_size += model.tensorrt_engine_size

        return {
            'huggingface': hf_size,
            'tensorrt': trt_size,
            'total': hf_size + trt_size
        }

    def get_onnx_model_path(self, model_id: str) -> Optional[Path]:
        """Get the path to the ONNX model file for a cached model.

        Returns:
            Path to the .onnx file if found, None otherwise
        """
        cache_dir = self._hf_repo_to_cache_dir(model_id)

        if not cache_dir.exists():
            return None

        # Look in snapshots for .onnx files
        snapshots_dir = cache_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    for onnx_file in snapshot.glob("*.onnx"):
                        return onnx_file

        return None

    @staticmethod
    def format_size(size_bytes: int) -> str:
        """Format byte size to human-readable string.

        Examples:
            format_size(1024) -> "1.0 KB"
            format_size(1048576) -> "1.0 MB"
            format_size(1073741824) -> "1.0 GB"
        """
        if size_bytes == 0:
            return "0 B"

        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        size = float(size_bytes)

        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1

        if unit_index == 0:
            return f"{int(size)} B"
        else:
            return f"{size:.1f} {units[unit_index]}"
