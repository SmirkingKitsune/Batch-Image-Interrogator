"""Resolving model file paths to names llama.cpp can work with.

The HuggingFace hub cache stores every downloaded file as a content-addressed
blob under ``blobs/<sha256>`` and exposes it through a symlink in
``snapshots/<revision>/<original-name>``. Both paths reach the same bytes, so
for most purposes either will do.

Split GGUFs are the exception. llama.cpp locates the remaining parts of a
multi-part model by pattern-matching the filename it was given
(``…-00001-of-00002.gguf``), so a blob path makes the other parts unfindable and
the load fails with "invalid split file name" — an error that names neither the
real cause nor the fix.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

# llama.cpp's split naming convention.
SPLIT_NAME_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.IGNORECASE)


def is_split_part(path: Path) -> bool:
    """Whether a filename identifies one part of a multi-part GGUF."""
    return SPLIT_NAME_RE.search(Path(path).name) is not None


def is_hf_blob(path: Path) -> bool:
    """Whether a path points into a HuggingFace hub cache blobs directory."""
    return Path(path).parent.name == "blobs"


def hf_snapshot_alias(path: Path) -> Optional[Path]:
    """The snapshots/ symlink naming this blob, or None.

    Returns None when the path is not a blob, when the cache has no snapshot
    referencing it, or on a platform where the hub duplicates files instead of
    symlinking them — in which case the caller should keep the original path.
    """
    blob = Path(path)
    if not is_hf_blob(blob) or not blob.exists():
        return None

    snapshots = blob.parent.parent / "snapshots"
    if not snapshots.is_dir():
        return None

    try:
        target = blob.resolve()
    except OSError:
        return None

    matches = []
    for candidate in snapshots.rglob("*"):
        try:
            if candidate.is_file() and candidate.resolve() == target:
                matches.append(candidate)
        except OSError:
            continue

    if not matches:
        return None
    # Deterministic when several revisions reference the same blob; they are by
    # definition the same bytes, so any is correct.
    return sorted(matches)[0]


def resolve_model_path(path: str) -> Tuple[str, str]:
    """Return (path_to_use, explanation).

    The explanation is empty when nothing was substituted, and otherwise
    describes the change for the caller to surface. Substitution is only ever
    between two names for one file, so it cannot load something other than what
    was asked for.
    """
    if not path:
        return path, ""

    original = Path(path)
    alias = hf_snapshot_alias(original)
    if alias is None:
        return path, ""

    reason = (
        "it is part of a split GGUF, which llama.cpp locates by filename"
        if is_split_part(alias)
        else "HuggingFace blob names are opaque"
    )
    return str(alias), f"Using {alias.name} instead of the cache blob: {reason}."
