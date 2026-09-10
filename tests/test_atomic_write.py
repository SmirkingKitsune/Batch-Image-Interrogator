"""Tests for crash-safe settings writes.

The property that matters is not "the write works" but "a failed write leaves
the previous contents intact". These settings files are gitignored and rewritten
on every UI change, so a truncating write is an unrecoverable loss.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.atomic_write import write_json_atomic, write_text_atomic  # noqa: E402
from core.inquiry_settings import InquirySettings  # noqa: E402
from core.tag_filters import TagFilterSettings  # noqa: E402


class AtomicWriteTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "settings.json"

    def test_writes_and_round_trips(self):
        write_json_atomic(self.path, {"a": 1, "b": "x"})
        self.assertEqual(json.loads(self.path.read_text()), {"a": 1, "b": "x"})

    def test_replaces_existing_contents(self):
        write_json_atomic(self.path, {"v": 1})
        write_json_atomic(self.path, {"v": 2})
        self.assertEqual(json.loads(self.path.read_text()), {"v": 2})

    def test_non_ascii_survives(self):
        write_json_atomic(self.path, {"p": "artist’s gallery — ünïcode"},
                          ensure_ascii=False)
        self.assertEqual(json.loads(self.path.read_text())["p"],
                         "artist’s gallery — ünïcode")

    def test_original_survives_a_failed_write(self):
        write_json_atomic(self.path, {"prompt": "the tuned original"})

        with patch("core.atomic_write.os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                write_json_atomic(self.path, {"prompt": "replacement"})

        # The whole point: the previous contents are still there.
        self.assertEqual(json.loads(self.path.read_text()),
                         {"prompt": "the tuned original"})

    def test_original_survives_unserialisable_data(self):
        # json.dump writing into an already-truncated file was the old failure
        # mode; serialising first means the file is never opened at all.
        write_json_atomic(self.path, {"prompt": "the tuned original"})
        with self.assertRaises(TypeError):
            write_json_atomic(self.path, {"bad": object()})
        self.assertEqual(json.loads(self.path.read_text()),
                         {"prompt": "the tuned original"})

    def test_no_temp_file_is_left_behind(self):
        write_json_atomic(self.path, {"v": 1})
        self.assertEqual([p.name for p in self.dir.iterdir()], ["settings.json"])

    def test_temp_file_is_cleaned_up_after_failure(self):
        with patch("core.atomic_write.os.replace", side_effect=OSError("boom")):
            with self.assertRaises(OSError):
                write_json_atomic(self.path, {"v": 1})
        self.assertEqual(list(self.dir.iterdir()), [])

    def test_temp_file_is_a_sibling_not_in_tmp(self):
        # A temp file on another filesystem would make os.replace fail with
        # EXDEV, so it has to live beside the target.
        seen = {}
        real_replace = os.replace

        def spy(src, dst):
            seen["src"] = Path(src)
            return real_replace(src, dst)

        with patch("core.atomic_write.os.replace", side_effect=spy):
            write_json_atomic(self.path, {"v": 1})
        self.assertEqual(seen["src"].parent, self.path.parent)

    def test_fsync_is_called_on_the_data(self):
        with patch("core.atomic_write.os.fsync") as fsync:
            write_text_atomic(self.path, "hello")
        self.assertTrue(fsync.called, "data must be flushed to disk before rename")

    def test_survives_a_directory_that_cannot_be_synced(self):
        # Windows cannot open a directory; that must not break the write.
        with patch("core.atomic_write.os.open", side_effect=OSError("no dirs")):
            write_json_atomic(self.path, {"v": 1})
        self.assertEqual(json.loads(self.path.read_text()), {"v": 1})


class SettingsUseAtomicWritesTests(unittest.TestCase):
    """The real files this exists to protect."""

    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def test_inquiry_settings_keep_prompt_after_failed_save(self):
        path = self.dir / "inquiry_settings.json"
        settings = InquirySettings(settings_file=str(path))
        settings.update_options({"batch_prompt": "expensively tuned prompt"})

        with patch("core.atomic_write.os.replace", side_effect=OSError("power cut")):
            settings.update_options({"batch_prompt": "clobbered"})

        reloaded = InquirySettings(settings_file=str(path))
        self.assertEqual(reloaded.get_options()["batch_prompt"],
                         "expensively tuned prompt")

    def test_tag_filters_survive_a_failed_save(self):
        path = self.dir / "tag_filters.json"
        filters = TagFilterSettings(str(path))
        filters.remove_list = {"unwanted"}
        filters.save_settings()

        with patch("core.atomic_write.os.replace", side_effect=OSError("power cut")):
            filters.remove_list = {"clobbered"}
            filters.save_settings()

        reloaded = TagFilterSettings(str(path))
        self.assertEqual(set(reloaded.remove_list), {"unwanted"})


if __name__ == "__main__":
    unittest.main()
