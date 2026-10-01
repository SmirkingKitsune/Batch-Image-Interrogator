"""Thumbnails for the Electron gallery, decoded with Pillow.

The PyQt6 gallery decodes through QImageReader; the bridge must not import Qt,
so it keeps its own JPEG cache beside the PyQt6 one. Entries are keyed by path,
modification time, size and target size, so an edited image gets a new entry
rather than a stale thumbnail.
"""

from __future__ import annotations

import hashlib
import io
import os
import threading
from pathlib import Path
from typing import Tuple

from PIL import Image, ImageOps

from core.file_manager import FileManager

THUMBNAIL_CACHE_DIR = Path.home() / ".cache" / "image_interrogator_thumbnails" / "web"
JPEG_QUALITY = 86
# Transparent images are flattened onto the gallery's tile colour.
FLATTEN_BACKGROUND = (23, 26, 31)


def resolve_image_path(path: str) -> Path:
    """Validate that a request names an existing, supported image file."""
    if not path:
        raise ValueError("No image path given")
    target = Path(path)
    if target.suffix.lower() not in FileManager.SUPPORTED_EXTENSIONS:
        raise ValueError("Not a supported image type")
    if not target.is_file():
        raise FileNotFoundError(path)
    return target


class Thumbnailer:
    """Produces and caches JPEG thumbnails."""

    def __init__(self, cache_dir: Path = THUMBNAIL_CACHE_DIR, max_parallel: int = 4):
        self.cache_dir = Path(cache_dir)
        # Decoding a large PNG is CPU- and memory-heavy; bound the concurrency
        # so a scrolling gallery cannot starve the rest of the backend.
        self._slots = threading.BoundedSemaphore(max_parallel)

    def __call__(self, path: str, size: int) -> Tuple[bytes, str]:
        return self.thumbnail(path, size), "image/jpeg"

    def cache_key(self, path: str, size: int) -> str:
        """Identity of one thumbnail; it changes when the image is edited."""
        return self._identify(path, size)[1]

    @staticmethod
    def _identify(path: str, size: int) -> Tuple[Path, str]:
        target = resolve_image_path(path)
        stat = target.stat()
        key = hashlib.sha1(
            f"{target.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{size}".encode("utf-8")
        ).hexdigest()
        return target, key

    def thumbnail(self, path: str, size: int) -> bytes:
        target, key = self._identify(path, size)
        cached = self.cache_dir / key[:2] / f"{key}.jpg"
        try:
            return cached.read_bytes()
        except OSError:
            pass

        with self._slots:
            data = self._render(target, size)
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            temp = cached.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            temp.write_bytes(data)
            os.replace(temp, cached)
        except OSError:
            pass  # A read-only cache only costs speed.
        return data

    @staticmethod
    def _render(target: Path, size: int) -> bytes:
        with Image.open(target) as image:
            # JPEG can decode at a reduced scale, which is most of the win.
            image.draft("RGB", (size, size))
            image = ImageOps.exif_transpose(image)
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
                rgba = image.convert("RGBA")
                flat = Image.new("RGB", rgba.size, FLATTEN_BACKGROUND)
                flat.paste(rgba, mask=rgba.split()[-1])
                image = flat
            elif image.mode != "RGB":
                image = image.convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=JPEG_QUALITY, optimize=True)
            return buffer.getvalue()
