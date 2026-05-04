"""Minimal GGUF metadata reader for llama.cpp model settings."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional


GGUF_MAGIC = b"GGUF"

GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


_SCALAR_FORMATS = {
    GGUF_TYPE_UINT8: "<B",
    GGUF_TYPE_INT8: "<b",
    GGUF_TYPE_UINT16: "<H",
    GGUF_TYPE_INT16: "<h",
    GGUF_TYPE_UINT32: "<I",
    GGUF_TYPE_INT32: "<i",
    GGUF_TYPE_FLOAT32: "<f",
    GGUF_TYPE_BOOL: "<?",
    GGUF_TYPE_UINT64: "<Q",
    GGUF_TYPE_INT64: "<q",
    GGUF_TYPE_FLOAT64: "<d",
}


class GGUFMetadataError(ValueError):
    """Raised when GGUF metadata cannot be read."""


def read_gguf_metadata(model_path: str) -> Dict[str, Any]:
    """Read GGUF key-value metadata without loading tensor data."""
    path = Path(model_path).expanduser()
    metadata: Dict[str, Any] = {}

    with path.open("rb") as handle:
        magic = handle.read(4)
        if magic != GGUF_MAGIC:
            raise GGUFMetadataError(f"Not a GGUF file: {path}")

        version = _read_struct(handle, "<I")
        if int(version) < 2:
            raise GGUFMetadataError(f"Unsupported GGUF version: {version}")

        _tensor_count = _read_struct(handle, "<Q")
        metadata_count = _read_struct(handle, "<Q")

        for _ in range(int(metadata_count)):
            key = _read_string(handle)
            value_type = _read_struct(handle, "<I")
            metadata[key] = _read_value(handle, int(value_type))

    return metadata


def get_gguf_context_length(model_path: str) -> Optional[int]:
    """Return the model context length from GGUF metadata when present."""
    metadata = read_gguf_metadata(model_path)
    architecture = metadata.get("general.architecture")
    candidate_keys = []
    if isinstance(architecture, str) and architecture:
        candidate_keys.append(f"{architecture}.context_length")
    candidate_keys.extend(
        [
            "llama.context_length",
            "qwen2.context_length",
            "qwen3.context_length",
            "gemma.context_length",
            "gemma2.context_length",
            "gemma3.context_length",
        ]
    )

    for key in candidate_keys:
        value = metadata.get(key)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _read_struct(handle: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = handle.read(size)
    if len(data) != size:
        raise GGUFMetadataError("Unexpected end of GGUF metadata")
    return struct.unpack(fmt, data)[0]


def _read_string(handle: BinaryIO) -> str:
    length = _read_struct(handle, "<Q")
    data = handle.read(int(length))
    if len(data) != int(length):
        raise GGUFMetadataError("Unexpected end of GGUF string")
    return data.decode("utf-8", errors="replace")


def _read_value(handle: BinaryIO, value_type: int) -> Any:
    if value_type == GGUF_TYPE_STRING:
        return _read_string(handle)

    if value_type == GGUF_TYPE_ARRAY:
        element_type = _read_struct(handle, "<I")
        length = _read_struct(handle, "<Q")
        return [_read_value(handle, int(element_type)) for _ in range(int(length))]

    fmt = _SCALAR_FORMATS.get(value_type)
    if not fmt:
        raise GGUFMetadataError(f"Unsupported GGUF metadata value type: {value_type}")
    return _read_struct(handle, fmt)
