"""Tests for the database-vs-sidecar tag analysis shared by both front ends."""

import tempfile
import unittest
from pathlib import Path

from core.tag_filters import TagFilterSettings
from core.tag_review import (
    apply_common_tag_edits,
    build_tag_comparison,
    collect_editor_tags,
    compute_common_tags,
    extract_wd_ratings,
    plan_apply_to_file,
)


class TagReviewTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.filters = TagFilterSettings(str(Path(self._tmp.name) / "filters.json"))

    def tearDown(self):
        self._tmp.cleanup()


class BuildTagComparisonTests(TagReviewTestCase):
    def test_statuses_cover_every_case(self):
        self.filters.add_remove_tag("watermark")
        self.filters.add_replace_rule("1girl", "solo female character")
        db_tags = ["solo", "watermark", "golden hour", "1girl", "faint"]
        confidence = {"solo": 0.97, "watermark": 0.52, "golden hour": 0.5, "1girl": 0.99, "faint": 0.1}
        file_tags = ["solo", "faint", "manual_tag"]

        rows = {row["tag"]: row for row in build_tag_comparison(db_tags, confidence, file_tags, self.filters)}

        self.assertEqual(rows["solo"]["status"], "in_both")
        self.assertEqual(rows["watermark"]["status"], "removed_by_filter")
        self.assertEqual(rows["golden hour"]["status"], "db_only")
        # Below the review threshold, yet present in the file: kept by hand.
        self.assertEqual(rows["faint"]["status"], "manually_added")
        self.assertEqual(rows["solo female character"]["status"], "replaced")
        self.assertEqual(rows["solo female character"]["original_tag"], "1girl")
        self.assertEqual(rows["manual_tag"]["status"], "file_only")
        self.assertIsNone(rows["manual_tag"]["confidence"])

    def test_underscore_equivalence_when_enabled(self):
        self.filters.set_replace_underscores(True)
        rows = build_tag_comparison(["long_hair"], {"long_hair": 0.9}, ["long hair"], self.filters)
        self.assertEqual([row["status"] for row in rows], ["in_both"])

    def test_without_filters_every_db_tag_would_be_written(self):
        rows = build_tag_comparison(["a", "b"], {}, ["a"], None)
        self.assertEqual({row["tag"]: row["status"] for row in rows}, {"a": "in_both", "b": "db_only"})


class PlanApplyToFileTests(TagReviewTestCase):
    def test_plan_adds_rewrites_and_keeps_manual_tags(self):
        self.filters.add_remove_tag("watermark")
        self.filters.add_replace_rule("1girl", "solo female character")
        plan = plan_apply_to_file(
            ["1girl", "solo", "watermark", "golden hour"],
            {"1girl": 0.99, "solo": 0.97, "watermark": 0.52, "golden hour": 0.5},
            ["1girl", "solo", "manual_tag"],
            self.filters,
        )
        self.assertEqual(plan["tags"], ["solo female character", "solo", "manual_tag", "golden hour"])
        self.assertEqual(plan["added"], ["golden hour"])
        self.assertEqual(plan["rewritten"], [("1girl", "solo female character")])
        self.assertEqual(plan["kept_manual"], ["manual_tag"])

    def test_prefix_tags_are_added_once(self):
        self.filters.add_prefix_tag("trigger_word")
        plan = plan_apply_to_file(["cat"], {"cat": 0.9}, ["trigger_word", "cat"], self.filters)
        self.assertEqual(plan["tags"], ["trigger_word", "cat"])
        self.assertEqual(plan["added"], [])

    def test_matching_file_produces_no_changes(self):
        plan = plan_apply_to_file(["cat", "dog"], {"cat": 0.9, "dog": 0.8}, ["cat", "dog"], self.filters)
        self.assertEqual(plan["tags"], ["cat", "dog"])
        self.assertEqual(plan["added"], [])
        self.assertEqual(plan["rewritten"], [])


class RatingsAndEditorTests(TagReviewTestCase):
    def test_wd_ratings_accept_either_spelling(self):
        ratings = extract_wd_ratings(
            ["general", "rating:explicit"], {"general": 0.94, "rating:explicit": 0.01}
        )
        self.assertEqual(ratings, {"general": 0.94, "sensitive": 0.0, "questionable": 0.0, "explicit": 0.01})

    def test_editor_offers_db_spelling_and_marks_file_tags(self):
        self.filters.set_replace_underscores(True)
        all_tags, selected = collect_editor_tags(
            [{"tags": ["long_hair", "smile"]}], ["long hair", "hand_added"], self.filters
        )
        self.assertEqual(all_tags, ["hand_added", "long_hair", "smile"])
        self.assertEqual(selected, ["hand_added", "long_hair"])


class CommonTagTests(TagReviewTestCase):
    def test_common_tags_intersect_database_and_file_sources(self):
        common = compute_common_tags(
            [
                ([["cat", "outdoors"]], ["sky"]),
                ([], ["cat", "sky", "tree"]),
            ],
            self.filters,
        )
        self.assertEqual(common, {"cat", "sky"})

    def test_edits_preserve_unique_tags_and_normalized_matches(self):
        self.filters.set_replace_underscores(True)
        edited = apply_common_tag_edits(
            ["long hair", "unique", "cat"], tags_to_remove={"long_hair"}, tags_to_add={"cat", "new"},
            tag_filters=self.filters,
        )
        self.assertEqual(edited, ["unique", "cat", "new"])


if __name__ == "__main__":
    unittest.main()
