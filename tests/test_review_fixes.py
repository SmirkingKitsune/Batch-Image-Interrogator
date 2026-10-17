"""Regression tests for the code-review fixes (tag filtering + hash caching)."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from core.tag_filters import TagFilterSettings
from core.hashing import clear_hash_cache, hash_image_content
from core.file_manager import FileManager


class TagFilterDedupTests(unittest.TestCase):
    def _make(self, tmpdir):
        return TagFilterSettings(str(Path(tmpdir) / "tag_filters.json"))

    def test_underscore_collapse_deduplicates_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tf = self._make(tmpdir)
            tf.set_replace_underscores(True)
            # Both collapse to "long hair"; output must contain it once.
            result = tf.apply_filters(["long_hair", "long hair", "smile"])
            self.assertEqual(result.count("long hair"), 1)
            self.assertIn("smile", result)

    def test_confidence_dedup_keeps_highest_score(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tf = self._make(tmpdir)
            tf.set_replace_underscores(True)
            tags = ["long_hair", "long hair"]
            scores = {"long_hair": 0.9, "long hair": 0.5}
            out_tags, out_scores = tf.filter_tags_with_confidence(tags, scores, 0.0)
            self.assertEqual(out_tags, ["long hair"])
            self.assertAlmostEqual(out_scores["long hair"], 0.9)

    def test_case_dedup_keeps_highest_score_and_first_spelling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tf = self._make(tmpdir)
            tags, scores = tf.filter_tags_with_confidence(
                ["Cat", "cat"], {"Cat": 0.4, "cat": 0.9}, 0.0
            )
            self.assertEqual(tags, ["Cat"])
            self.assertEqual(scores, {"Cat": 0.9})

    def test_replacement_into_blacklisted_tag_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tf = self._make(tmpdir)
            tf.add_remove_tag("nsfw")
            tf.add_replace_rule("explicit", "nsfw")
            self.assertEqual(tf.apply_filters(["explicit", "smile"]), ["smile"])

    def test_skip_list_cache_updates_on_mutation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tf = self._make(tmpdir)
            tf.set_replace_underscores(True)
            # Default skip list preserves emoji-style tags.
            self.assertEqual(tf.replace_underscore_in_tag("o_o"), "o_o")
            # Newly added skip tag is honored immediately (cache rebuilt).
            tf.add_underscore_skip_tag("foo_bar")
            self.assertEqual(tf.replace_underscore_in_tag("foo_bar"), "foo_bar")
            # Reset drops it again, so it collapses normally.
            tf.reset_underscore_skip_list()
            self.assertEqual(tf.replace_underscore_in_tag("foo_bar"), "foo bar")


class HashCacheTests(unittest.TestCase):
    def setUp(self):
        clear_hash_cache()

    def test_cached_hash_matches_fresh_hash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "img.bin"
            path.write_bytes(b"hello world")
            cached = hash_image_content(str(path))
            clear_hash_cache()
            self.assertEqual(hash_image_content(str(path)), cached)

    def test_cache_invalidates_when_content_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "img.bin"
            path.write_bytes(b"first")
            first = hash_image_content(str(path))
            # Different size => identity changes => cache refreshes.
            path.write_bytes(b"second-and-longer")
            second = hash_image_content(str(path))
            self.assertNotEqual(first, second)
            clear_hash_cache()
            self.assertEqual(hash_image_content(str(path)), second)


class FileManagerSafetyTests(unittest.TestCase):
    def test_overwrite_writes_expected_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Path(tmpdir) / "a.png"
            FileManager.write_tags_to_file(img, ["x", "y"], overwrite=True)
            self.assertEqual((Path(tmpdir) / "a.txt").read_text(encoding="utf-8"), "x, y")

    def test_merge_dedupe_is_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Path(tmpdir) / "a.png"
            (Path(tmpdir) / "a.txt").write_text("Cat, dog", encoding="utf-8")
            FileManager.write_tags_to_file(img, ["cat", "bird"], overwrite=False)
            tags = [t.strip() for t in (Path(tmpdir) / "a.txt").read_text(encoding="utf-8").split(",")]
            # 'cat' collides with existing 'Cat' (case-insensitive) and is dropped,
            # keeping the first-seen spelling.
            self.assertEqual(tags, ["Cat", "dog", "bird"])

    def test_merge_does_not_truncate_on_read_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            img = Path(tmpdir) / "a.png"
            txt = Path(tmpdir) / "a.txt"
            txt.write_text("keep1, keep2", encoding="utf-8")
            original = FileManager._read_tags_strict

            def boom(_path):
                raise OSError("simulated read failure")

            FileManager._read_tags_strict = staticmethod(boom)
            try:
                with self.assertRaises(OSError):
                    FileManager.write_tags_to_file(img, ["new"], overwrite=False)
            finally:
                FileManager._read_tags_strict = staticmethod(original)
            # Existing content must be intact (not truncated to just "new").
            self.assertEqual(txt.read_text(encoding="utf-8"), "keep1, keep2")


class WDCategoryTests(unittest.TestCase):
    def test_ratings_use_argmax_and_general_uses_threshold(self):
        import pandas as pd
        from interrogators.wd_interrogator import WDInterrogator

        class FakeModel:
            def get_inputs(self):
                return [SimpleNamespace(name="input")]

            def run(self, _outputs, _inputs):
                # [general(rating), sensitive(rating), 1girl(general), smile(general)]
                return [np.array([[0.9, 0.7, 0.8, 0.1]], dtype=np.float32)]

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.png"
            Image.new("RGB", (8, 8), (120, 80, 40)).save(image_path)

            interrogator = WDInterrogator()
            interrogator.is_loaded = True
            interrogator.target_size = 8
            interrogator.threshold = 0.35
            interrogator.model = FakeModel()
            interrogator.tags = pd.DataFrame(
                {
                    "name": ["general", "sensitive", "1girl", "smile"],
                    "category": [9, 9, 0, 0],
                }
            )

            tags = interrogator.interrogate(str(image_path))["tags"]
            # Only the top rating ("general") is kept; "sensitive" rating dropped.
            self.assertIn("general", tags)
            self.assertNotIn("sensitive", tags)
            # General tag above threshold kept; below-threshold dropped.
            self.assertIn("1girl", tags)
            self.assertNotIn("smile", tags)


if __name__ == "__main__":
    unittest.main()


class WDRatingCacheTests(unittest.TestCase):
    def test_fresh_and_cached_ratings_match_at_same_and_higher_threshold(self):
        import pandas as pd
        from interrogators.wd_interrogator import WDInterrogator
        w = WDInterrogator()
        w.is_loaded = True
        w.threshold = 0.8
        w.tags = pd.DataFrame({"name": ["general", "explicit", "smile"], "category": [9, 9, 0]})
        w.input_name = "input"
        w.preprocess_image = lambda _: np.zeros((1, 1, 1, 3))
        w.model = SimpleNamespace(run=lambda *_: [np.array([[0.6, 0.4, 0.95]])])
        fresh = w.interrogate("unused")
        cached = {**fresh, "config": w.result_signature(), "result_version": w.RESULT_VERSION}
        self.assertEqual(fresh["tags"], ["smile", "general"])
        self.assertEqual(w.reuse_cached_result(cached), fresh)
        w.threshold = 0.99
        self.assertEqual(w.reuse_cached_result(cached)["tags"], ["general"])

    def test_old_rating_policy_is_not_reused(self):
        from interrogators.wd_interrogator import WDInterrogator
        w = WDInterrogator()
        old = {"tags": ["general", "explicit"], "confidence_scores": {"general": 0.6, "explicit": 0.4},
               "config": {"preprocess": w.PREPROCESS_VERSION, "threshold": w.threshold}, "result_version": 1}
        self.assertIsNone(w.reuse_cached_result(old))
