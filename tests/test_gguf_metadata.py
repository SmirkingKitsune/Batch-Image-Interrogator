"""Unit tests for minimal GGUF metadata parsing."""

import struct
import tempfile
import unittest
from pathlib import Path

from core.gguf_metadata import GGUFMetadataError, get_gguf_context_length, read_gguf_metadata


class GGUFMetadataTests(unittest.TestCase):
    def _write_fake_gguf(self, path: Path, entries):
        with path.open("wb") as handle:
            handle.write(b"GGUF")
            handle.write(struct.pack("<I", 3))
            handle.write(struct.pack("<Q", 0))
            handle.write(struct.pack("<Q", len(entries)))
            for key, value_type, value in entries:
                encoded_key = key.encode("utf-8")
                handle.write(struct.pack("<Q", len(encoded_key)))
                handle.write(encoded_key)
                handle.write(struct.pack("<I", value_type))
                if value_type == 8:
                    encoded_value = value.encode("utf-8")
                    handle.write(struct.pack("<Q", len(encoded_value)))
                    handle.write(encoded_value)
                elif value_type == 4:
                    handle.write(struct.pack("<I", int(value)))
                elif value_type == 10:
                    handle.write(struct.pack("<Q", int(value)))
                else:
                    raise AssertionError(f"Unsupported test value type: {value_type}")

    def test_reads_architecture_context_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.gguf"
            self._write_fake_gguf(
                model_path,
                [
                    ("general.architecture", 8, "qwen3"),
                    ("qwen3.context_length", 4, 32768),
                ],
            )

            metadata = read_gguf_metadata(str(model_path))

            self.assertEqual(metadata["general.architecture"], "qwen3")
            self.assertEqual(get_gguf_context_length(str(model_path)), 32768)

    def test_rejects_non_gguf_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "not.gguf"
            model_path.write_bytes(b"nope")

            with self.assertRaises(GGUFMetadataError):
                read_gguf_metadata(str(model_path))


if __name__ == "__main__":
    unittest.main()
