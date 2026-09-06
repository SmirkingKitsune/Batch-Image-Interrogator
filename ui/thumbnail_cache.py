"""On-disk cache for gallery thumbnails.

Decoding a source image to a thumbnail costs roughly 180 ms for a large PNG,
and PNG cannot be scaled during decode the way JPEG can, so a full gallery
re-decodes everything from scratch on every visit. Caching the decoded
thumbnail turns that into a ~0.2 ms read.

Entries are keyed by path, modification time and size, so editing or replacing
an image naturally produces a new key rather than serving a stale thumbnail.
"""

import hashlib
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QSize
from PyQt6.QtGui import QImage

# Matches the convention used for the other model/engine caches.
THUMBNAIL_CACHE_DIR = Path.home() / ".cache" / "image_interrogator_thumbnails"


class ThumbnailCache:
    """Stores decoded thumbnails on disk, keyed by source file identity."""

    # Both encodings are lossless, so a cached thumbnail is pixel-identical to
    # a freshly decoded one. WebP at quality 100 is about 35% smaller than PNG
    # but drops the alpha channel, so transparent thumbnails use PNG instead.
    WEBP_QUALITY = 100
    MAX_BYTES = 512 * 1024 * 1024
    # Pruning walks the whole cache, so amortize it across many writes.
    PRUNE_INTERVAL = 500

    def __init__(self, cache_dir: Optional[Path] = None, max_bytes: Optional[int] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else THUMBNAIL_CACHE_DIR
        self.max_bytes = self.MAX_BYTES if max_bytes is None else max_bytes
        self._lock = threading.Lock()
        self._writes_since_prune = 0

    def get(self, image_path: str, target_size: QSize) -> Optional[QImage]:
        """Return the cached thumbnail, or None on a miss."""
        key = self._key(image_path, target_size)
        if key is None:
            return None

        for path in self._candidate_paths(key):
            if not path.exists():
                continue
            image = QImage(str(path))
            if image.isNull():
                # Truncated or corrupt entry; drop it and fall back to decoding.
                self._discard(path)
                return None
            self._touch(path)
            return image
        return None

    def store(self, image_path: str, target_size: QSize, image: QImage) -> bool:
        """Write a thumbnail to the cache. Returns True when stored."""
        key = self._key(image_path, target_size)
        if key is None or image is None or image.isNull():
            return False

        if self._has_transparency(image):
            attempts = [("png", "PNG", -1)]
        else:
            # Fall back to PNG if this Qt build has no WebP writer.
            attempts = [("webp", "WEBP", self.WEBP_QUALITY), ("png", "PNG", -1)]

        for extension, fmt, quality in attempts:
            destination = self._path_for(key, extension)
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                # Write to a temporary file first so a crash cannot leave a
                # partial entry that later reads would treat as valid.
                # Unique per thread: the gallery widget and the worker can
                # decode the same image at the same time.
                temporary = destination.with_name(
                    f"{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp"
                )
                if not image.save(str(temporary), fmt, quality):
                    temporary.unlink(missing_ok=True)
                    continue
                os.replace(temporary, destination)
                break
            except OSError:
                continue
        else:
            return False

        with self._lock:
            self._writes_since_prune += 1
            due = self._writes_since_prune >= self.PRUNE_INTERVAL
            if due:
                self._writes_since_prune = 0
        if due:
            self.prune()
        return True

    def prune(self) -> int:
        """Drop least-recently-used entries until under the size cap.

        Returns the number of bytes reclaimed.
        """
        entries = []
        total = 0
        for path in self.cache_dir.rglob("*"):
            if not path.is_file() or path.name.endswith(".tmp"):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            entries.append((stat.st_mtime, stat.st_size, path))
            total += stat.st_size

        if total <= self.max_bytes:
            return 0

        reclaimed = 0
        entries.sort()  # Oldest access first.
        for _mtime, size, path in entries:
            if total - reclaimed <= self.max_bytes:
                break
            if self._discard(path):
                reclaimed += size
        return reclaimed

    def clear(self) -> None:
        """Remove every cached thumbnail."""
        shutil.rmtree(self.cache_dir, ignore_errors=True)

    def total_bytes(self) -> int:
        """Current size of the cache on disk."""
        total = 0
        for path in self.cache_dir.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                continue
        return total

    def _key(self, image_path: str, target_size: QSize) -> Optional[str]:
        """Identity of a source file at a given thumbnail size."""
        try:
            stat = os.stat(image_path)
        except OSError:
            return None
        raw = (
            f"{os.path.abspath(image_path)}|{stat.st_mtime_ns}|{stat.st_size}"
            f"|{target_size.width()}x{target_size.height()}"
        )
        return hashlib.sha1(raw.encode("utf-8", "surrogateescape")).hexdigest()

    def _path_for(self, key: str, extension: str) -> Path:
        # Shard on the first byte so no single directory holds every entry.
        return self.cache_dir / key[:2] / f"{key}.{extension}"

    def _candidate_paths(self, key: str):
        # WebP first: transparent thumbnails are the rare case.
        return (self._path_for(key, "webp"), self._path_for(key, "png"))

    @staticmethod
    def _has_transparency(image: QImage) -> bool:
        """True when any pixel is not fully opaque."""
        if not image.hasAlphaChannel():
            return False
        converted = image if image.format() == QImage.Format.Format_ARGB32 else \
            image.convertToFormat(QImage.Format.Format_ARGB32)
        raw = converted.constBits()
        if raw is None:
            return True
        raw.setsize(converted.sizeInBytes())
        return min(bytes(raw)[3::4], default=255) < 255

    @staticmethod
    def _touch(path: Path) -> None:
        """Refresh access time for LRU, without writing metadata constantly."""
        try:
            now = time.time()
            if now - path.stat().st_mtime > 86400:
                os.utime(path, (now, now))
        except OSError:
            pass

    @staticmethod
    def _discard(path: Path) -> bool:
        try:
            path.unlink()
            return True
        except OSError:
            return False
