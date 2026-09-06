"""Image hashing and metadata extraction utilities."""

import hashlib
import os
import threading
from collections import OrderedDict
from pathlib import Path
from PIL import Image
from typing import Dict, Optional, Tuple

# Hashing an image means reading the whole file, and the UI hashes the same
# images repeatedly (selection, transcript lookups, batch context scans). Cache
# on identity rather than content so an edited file is re-read.
_HASH_CACHE_MAX = 20000
_hash_cache: "OrderedDict[Tuple[str, int, int], str]" = OrderedDict()
_hash_cache_lock = threading.Lock()


def _identity(image_path: str) -> Optional[Tuple[str, int, int]]:
    """Cache key for a file: path, modification time and size."""
    try:
        stat = os.stat(image_path)
    except OSError:
        return None
    return (os.path.abspath(image_path), stat.st_mtime_ns, stat.st_size)


def hash_image_content(image_path: str) -> str:
    """Generate SHA256 hash of image file content."""
    key = _identity(image_path)

    if key is not None:
        with _hash_cache_lock:
            cached = _hash_cache.get(key)
            if cached is not None:
                _hash_cache.move_to_end(key)
                return cached

    sha256_hash = hashlib.sha256()

    with open(image_path, "rb") as f:
        # 1 MiB blocks: 4 KiB reads spend most of their time in syscalls.
        for byte_block in iter(lambda: f.read(1024 * 1024), b""):
            sha256_hash.update(byte_block)

    digest = sha256_hash.hexdigest()

    if key is not None:
        with _hash_cache_lock:
            _hash_cache[key] = digest
            _hash_cache.move_to_end(key)
            while len(_hash_cache) > _HASH_CACHE_MAX:
                _hash_cache.popitem(last=False)

    return digest


def clear_hash_cache() -> None:
    """Drop memoized hashes."""
    with _hash_cache_lock:
        _hash_cache.clear()


def get_image_metadata(image_path: str) -> Dict:
    """Extract image metadata (dimensions, file size)."""
    path = Path(image_path)

    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        raise ValueError(f"Failed to read image metadata: {e}")

    file_size = path.stat().st_size

    return {
        'width': width,
        'height': height,
        'file_size': file_size
    }
