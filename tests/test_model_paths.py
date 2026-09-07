"""Unit tests for HuggingFace cache path resolution.

llama.cpp finds the remaining parts of a split GGUF by pattern-matching the
filename it was handed, so a content-addressed blob path makes them unfindable.
These cover recognising that case and substituting the equivalent snapshot name.
"""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.model_paths import (  # noqa: E402
    hf_snapshot_alias,
    is_hf_blob,
    is_split_part,
    resolve_model_path,
)


class HubCache:
    """A miniature HuggingFace hub cache."""

    def __init__(self, root: Path, revision: str = "abc123"):
        self.root = root / "models--org--Example-GGUF"
        self.blobs = self.root / "blobs"
        self.snapshot = self.root / "snapshots" / revision
        self.blobs.mkdir(parents=True)
        self.snapshot.mkdir(parents=True)

    def add(self, digest: str, name: str, subdir: str = "") -> Path:
        """Create a blob and the snapshot symlink that names it."""
        blob = self.blobs / digest
        blob.write_bytes(b"GGUF")
        link_dir = self.snapshot / subdir if subdir else self.snapshot
        link_dir.mkdir(parents=True, exist_ok=True)
        link = link_dir / name
        link.symlink_to(blob)
        return blob


class TestSplitDetection(unittest.TestCase):
    def test_split_parts_are_recognised(self):
        self.assertTrue(is_split_part(Path("model-00001-of-00002.gguf")))
        self.assertTrue(is_split_part(Path("Qwen3.8-27B-BF16-00002-of-00002.gguf")))

    def test_single_file_models_are_not_split(self):
        self.assertFalse(is_split_part(Path("model.gguf")))
        self.assertFalse(is_split_part(Path("mmproj-F16.gguf")))

    def test_near_miss_names_are_not_split(self):
        # Needs the five-digit -NNNNN-of-NNNNN form.
        self.assertFalse(is_split_part(Path("model-1-of-2.gguf")))
        self.assertFalse(is_split_part(Path("model-00001-of-00002.bin")))


class TestBlobDetection(unittest.TestCase):
    def test_blob_directory_is_recognised(self):
        self.assertTrue(is_hf_blob(Path("/cache/models--o--n/blobs/deadbeef")))

    def test_ordinary_paths_are_not_blobs(self):
        self.assertFalse(is_hf_blob(Path("/models/model.gguf")))
        self.assertFalse(is_hf_blob(Path("/cache/models--o--n/snapshots/rev/model.gguf")))


class TestResolution(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = HubCache(Path(self._tmp.name))

    def tearDown(self):
        self._tmp.cleanup()

    def test_split_blob_resolves_to_its_snapshot_name(self):
        blob = self.cache.add("aa" * 32, "Model-BF16-00001-of-00002.gguf", subdir="BF16")
        resolved, note = resolve_model_path(str(blob))
        self.assertTrue(resolved.endswith("Model-BF16-00001-of-00002.gguf"))
        self.assertIn("split GGUF", note)

    def test_resolved_path_is_the_same_file(self):
        blob = self.cache.add("bb" * 32, "Model-00001-of-00002.gguf")
        resolved, _ = resolve_model_path(str(blob))
        self.assertEqual(Path(resolved).resolve(), blob.resolve())

    def test_single_file_blob_also_resolves_with_a_milder_note(self):
        blob = self.cache.add("cc" * 32, "mmproj-F16.gguf")
        resolved, note = resolve_model_path(str(blob))
        self.assertTrue(resolved.endswith("mmproj-F16.gguf"))
        self.assertIn("opaque", note)
        self.assertNotIn("split", note)

    def test_ordinary_path_is_returned_untouched(self):
        path = "/models/some-model.gguf"
        self.assertEqual(resolve_model_path(path), (path, ""))

    def test_snapshot_path_is_left_alone(self):
        self.cache.add("dd" * 32, "Model-00001-of-00002.gguf")
        link = self.cache.snapshot / "Model-00001-of-00002.gguf"
        resolved, note = resolve_model_path(str(link))
        self.assertEqual(resolved, str(link))
        self.assertEqual(note, "")

    def test_empty_path_is_returned_untouched(self):
        self.assertEqual(resolve_model_path(""), ("", ""))

    def test_blob_with_no_snapshot_reference_is_kept(self):
        # An orphaned blob has no better name to offer.
        orphan = self.cache.blobs / ("ee" * 32)
        orphan.write_bytes(b"GGUF")
        resolved, note = resolve_model_path(str(orphan))
        self.assertEqual(resolved, str(orphan))
        self.assertEqual(note, "")

    def test_missing_blob_is_kept(self):
        missing = str(self.cache.blobs / ("ff" * 32))
        self.assertEqual(resolve_model_path(missing), (missing, ""))

    def test_resolution_is_deterministic_across_revisions(self):
        # The same blob referenced by two revisions must not alternate.
        blob = self.cache.add("ab" * 32, "Model-00001-of-00002.gguf")
        other = self.cache.root / "snapshots" / "zzz999"
        other.mkdir(parents=True)
        (other / "Model-00001-of-00002.gguf").symlink_to(blob)

        first = hf_snapshot_alias(blob)
        self.assertEqual(first, hf_snapshot_alias(blob))
        self.assertIsNotNone(first)


if __name__ == "__main__":
    unittest.main()
