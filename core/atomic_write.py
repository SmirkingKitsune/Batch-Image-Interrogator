"""Crash-safe file replacement for settings and small state files.

`open(path, "w")` truncates before it writes, so a power loss inside that window
leaves the file empty or half-written. Settings here are rewritten on every UI
change and several of them -- the tuned inquiry prompt, the tag filters -- are
gitignored, so a truncated write destroys the only copy that exists.

The sequence below is the standard durable replace: serialise first, write to a
temp file beside the target, flush it to disk, then rename over the original.
`os.replace` is atomic on POSIX and on Windows, so a reader either sees the
whole old file or the whole new one, never a partial write.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Union

PathLike = Union[str, Path]


def write_text_atomic(path: PathLike, text: str, encoding: str = "utf-8") -> None:
    """Replace `path` with `text`, leaving the original intact on failure."""
    target = Path(path)
    # Beside the target, so the rename stays within one filesystem. A temp file
    # in /tmp would make os.replace a cross-device error.
    tmp = target.with_name(target.name + ".tmp")

    try:
        with open(tmp, "w", encoding=encoding) as handle:
            handle.write(text)
            # flush() only reaches the OS buffer; fsync is what survives a power
            # cut. Without it the rename can land on disk before the contents,
            # which is the failure this module exists to prevent.
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise

    _sync_parent_directory(target)


def write_json_atomic(path: PathLike, data: Any, **json_kwargs: Any) -> None:
    """Write `data` as JSON, atomically.

    Serialising up front matters as much as the atomic rename: `json.dump`
    writing straight into an opened file means a non-serialisable value raises
    partway through and leaves the file truncated. Here a serialisation error
    happens before anything is opened.
    """
    json_kwargs.setdefault("indent", 2)
    write_text_atomic(path, json.dumps(data, **json_kwargs))


def _sync_parent_directory(target: Path) -> None:
    """Persist the rename itself, best effort.

    The rename is only durable once the directory entry is on disk. Not every
    platform allows opening a directory -- Windows does not -- so failure here
    is ignored: the contents are already synced and the replace is still atomic.
    """
    try:
        fd = os.open(str(target.parent), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)
