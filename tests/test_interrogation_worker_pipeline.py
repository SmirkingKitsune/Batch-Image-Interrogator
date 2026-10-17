"""Regression tests for classic interrogation worker cache and txt output behavior."""

import json
import math
import sqlite3
import tempfile
import unittest
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image

from core.database import InterrogationDatabase
from core.hashing import hash_image_content
from core.base_interrogator import BaseInterrogator
from core.pipelines import InterrogationBatchRunner
from interrogators.camie_interrogator import CamieInterrogator
from interrogators.clip_interrogator import CLIPInterrogator
from interrogators.wd_interrogator import WDInterrogator
from ui.workers import InterrogationWorker


class FixedScoreSession:
    """Stands in for the ONNX session: the same scores for every image."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="input")]

    def run(self, _outputs, _inputs):
        self.calls += 1
        return [np.array([self.scores], dtype=np.float32)]


def make_wd(threshold, scores):
    """A loaded WDInterrogator whose model scores tags as given in `scores`."""
    wd = WDInterrogator("FakeWD")
    wd.tags = pd.DataFrame({"name": list(scores)})
    wd.model = FixedScoreSession(list(scores.values()))
    wd.threshold = threshold
    wd.config = {"threshold": threshold}
    wd.is_loaded = True
    return wd


def make_camie(probs, threshold=0.5, category_thresholds=None, enabled_categories=None):
    """A loaded CamieInterrogator whose model gives each (tag, category) in
    `probs` its probability, set up the way load_model would."""
    camie = CamieInterrogator("FakeCamie")
    camie.tags_data = {"tags": [{"name": tag, "category": category} for tag, category in probs]}
    camie.model = FixedScoreSession([math.log(p / (1 - p)) for p in probs.values()])
    camie.threshold = threshold
    camie.threshold_profile = "category_specific" if category_thresholds else "overall"
    camie.category_thresholds = category_thresholds or {
        category: threshold for category in CamieInterrogator.CATEGORIES
    }
    camie.enabled_categories = enabled_categories or list(CamieInterrogator.CATEGORIES)
    camie.config = {"threshold": threshold}
    camie.is_loaded = True
    return camie


CAMIE_PROBS = {
    ("cat_ears", "general"): 0.9,
    ("smile", "general"): 0.6,
    ("outdoors", "general"): 0.4,
    ("hatsune_miku", "character"): 0.8,
    ("rating_general", "rating"): 0.95,
}


class FakeClipLibrary:
    """Stands in for clip_interrogator.Interrogator, one prompt per mode."""

    def __init__(self):
        self.calls = 0

    def _prompt(self, mode):
        self.calls += 1
        return f"a cat, {mode}"

    def interrogate(self, _image):
        return self._prompt("best")

    def interrogate_fast(self, _image):
        return self._prompt("fast")

    def interrogate_classic(self, _image):
        return self._prompt("classic")

    def interrogate_negative(self, _image):
        return self._prompt("negative")


def make_clip(mode="best", caption_model="blip-large", library=None):
    clip = CLIPInterrogator("FakeCLIP/test")
    clip.ci = library or FakeClipLibrary()
    clip.mode = mode
    clip.caption_model = caption_model
    clip.is_loaded = True
    return clip


class FakeTagger(BaseInterrogator):
    def __init__(self, model_name="FakeWD", model_type="WD", tags=None):
        super().__init__(model_name)
        self.model_type = model_type
        # Stand in for the real interrogator's cache version.
        if model_type == "WD":
            self.RESULT_VERSION = WDInterrogator.RESULT_VERSION
        self.tags = tags or ["new_tag"]
        self.calls = 0
        self.is_loaded = True

    def load_model(self, **kwargs):
        self.is_loaded = True

    def unload_model(self):
        self.is_loaded = False

    def interrogate(self, image_path: str, **kwargs):
        self.calls += 1
        return {
            "tags": list(self.tags),
            "confidence_scores": {tag: 0.9 for tag in self.tags},
            "raw_output": ", ".join(self.tags),
        }

    def get_model_type(self) -> str:
        return self.model_type

    def get_config(self):
        return {"threshold": 0.35}


class InterrogationWorkerPipelineTests(unittest.TestCase):
    def _make_image(self, directory: Path) -> Path:
        image_path = directory / "image.png"
        Image.new("RGB", (8, 8), (20, 40, 60)).save(image_path)
        return image_path

    def test_empty_wd_cache_is_reprocessed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))

            image_hash = hash_image_content(str(image_path))
            image_id = db.register_image(str(image_path), image_hash, 8, 8, image_path.stat().st_size)
            model_id = db.register_model("FakeWD", "WD", config={"threshold": 0.35})
            db.save_interrogation(image_id, model_id, [], {}, "")

            interrogator = make_wd(0.35, {"fresh_tag": 0.9})
            worker = InterrogationWorker([image_path], interrogator, db, write_files=False)
            worker.run()

            self.assertEqual(interrogator.model.calls, 1)
            cached = db.get_interrogation(image_hash, "FakeWD")
            self.assertEqual(cached["tags"], ["fresh_tag"])

    def test_non_empty_wd_cache_is_reprocessed_after_preprocessing_fix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))

            image_hash = hash_image_content(str(image_path))
            image_id = db.register_image(str(image_path), image_hash, 8, 8, image_path.stat().st_size)
            model_id = db.register_model("FakeWD", "WD", config={"threshold": 0.35})
            # Rows saved before the fix carry no signature.
            db.save_interrogation(image_id, model_id, ["stale_tag"], {"stale_tag": 0.4}, "stale_tag")

            interrogator = make_wd(0.35, {"fresh_tag": 0.9})
            worker = InterrogationWorker([image_path], interrogator, db, write_files=False)
            worker.run()

            self.assertEqual(interrogator.model.calls, 1)
            cached = db.get_interrogation(image_hash, "FakeWD")
            self.assertEqual(cached["tags"], ["fresh_tag"])
            self.assertEqual(cached["config"], {"preprocess": 2, "threshold": 0.35, "rating_policy": 1})

    def _run(self, image_path, db, interrogator):
        """Run a one-image batch; return (results, served_from_cache)."""
        seen = []
        InterrogationBatchRunner(
            [image_path], interrogator, db, write_files=False,
            on_result=lambda _path, results, meta: seen.append((results, meta["cached"])),
            on_error=lambda _path, message: self.fail(message),
        ).run()
        self.assertEqual(len(seen), 1)
        return seen[0]

    def _run_tags(self, image_path, db, interrogator):
        """Run a one-image batch; return (tags, served_from_cache)."""
        results, cached = self._run(image_path, db, interrogator)
        return results["tags"], cached

    def test_wd_rerun_is_served_from_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))
            wd = make_wd(0.35, {"cat": 0.9, "dog": 0.5, "tree": 0.1})

            self.assertEqual(self._run_tags(image_path, db, wd), (["cat", "dog"], False))
            self.assertEqual(self._run_tags(image_path, db, wd), (["cat", "dog"], True))
            self.assertEqual(wd.model.calls, 1)

    def test_wd_cache_serves_a_higher_threshold_by_dropping_tags_below_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))
            scores = {"cat": 0.9, "dog": 0.5, "tree": 0.4}
            self._run_tags(image_path, db, make_wd(0.35, scores))

            stricter = make_wd(0.45, scores)
            self.assertEqual(self._run_tags(image_path, db, stricter), (["cat", "dog"], True))
            self.assertEqual(stricter.model.calls, 0)
            # The saved row keeps the wider result for later runs.
            cached = db.get_interrogation(hash_image_content(str(image_path)), "FakeWD")
            self.assertEqual(cached["tags"], ["cat", "dog", "tree"])

    def test_wd_cache_is_rerun_for_a_lower_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))
            scores = {"cat": 0.9, "dog": 0.5, "tree": 0.4}
            self._run_tags(image_path, db, make_wd(0.45, scores))

            looser = make_wd(0.35, scores)
            self.assertEqual(self._run_tags(image_path, db, looser), (["cat", "dog", "tree"], False))
            self.assertEqual(looser.model.calls, 1)
            cached = db.get_interrogation(hash_image_content(str(image_path)), "FakeWD")
            self.assertEqual(cached["config"], {"preprocess": 2, "threshold": 0.35, "rating_policy": 1})

    def test_wd_cache_is_rerun_after_a_preprocessing_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))
            self._run_tags(image_path, db, make_wd(0.35, {"cat": 0.9}))

            changed = make_wd(0.35, {"cat": 0.9})
            changed.PREPROCESS_VERSION = WDInterrogator.PREPROCESS_VERSION + 1
            self.assertEqual(self._run_tags(image_path, db, changed), (["cat"], False))

    def test_unstamped_camie_cache_is_reprocessed_then_reused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))

            image_hash = hash_image_content(str(image_path))
            image_id = db.register_image(str(image_path), image_hash, 8, 8, image_path.stat().st_size)
            model_id = db.register_model("FakeCamie", "Camie", config={"threshold": 0.5})
            # Empty rows like this were left by the v2 metadata bug.
            db.save_interrogation(image_id, model_id, [], {}, "")

            camie = make_camie({("fresh_tag", "general"): 0.9})
            self.assertEqual(self._run_tags(image_path, db, camie), (["fresh_tag"], False))
            self.assertEqual(self._run_tags(image_path, db, camie), (["fresh_tag"], True))
            self.assertEqual(camie.model.calls, 1)

    def test_camie_cache_serves_stricter_settings_like_a_fresh_run(self):
        stricter_settings = [
            {"threshold": 0.7},
            {"category_thresholds": {**CamieInterrogator.THRESHOLD_PROFILES["category_specific"]["defaults"],
                                     "general": 0.5}},
            {"enabled_categories": ["general", "rating"]},
        ]
        for settings in stricter_settings:
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                image_path = self._make_image(tmp)
                db = InterrogationDatabase(str(tmp / "interrogations.db"))
                self._run(image_path, db, make_camie(CAMIE_PROBS, threshold=0.35))

                stricter = make_camie(CAMIE_PROBS, **settings)
                served, cached = self._run(image_path, db, stricter)
                fresh, _ = self._run(image_path, InterrogationDatabase(str(tmp / "fresh.db")),
                                     make_camie(CAMIE_PROBS, **settings))

                self.assertTrue(cached)
                self.assertEqual(stricter.model.calls, 0)
                self.assertEqual(served["tags"], fresh["tags"])
                self.assertEqual(served["confidence_scores"], fresh["confidence_scores"])
                self.assertEqual(json.loads(served["raw_output"]), json.loads(fresh["raw_output"]))

    def test_camie_cache_is_rerun_for_looser_settings(self):
        saved_settings = {"threshold": 0.35, "enabled_categories": ["general"]}
        looser_settings = [
            {"threshold": 0.3},
            {"enabled_categories": ["general", "character"]},
        ]
        for settings in looser_settings:
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as tmpdir:
                tmp = Path(tmpdir)
                image_path = self._make_image(tmp)
                db = InterrogationDatabase(str(tmp / "interrogations.db"))
                self._run(image_path, db, make_camie(CAMIE_PROBS, **saved_settings))

                looser = make_camie(CAMIE_PROBS, **{**saved_settings, **settings})
                _, cached = self._run(image_path, db, looser)
                self.assertFalse(cached)
                self.assertEqual(looser.model.calls, 1)

    def test_clip_cache_is_kept_per_mode_and_caption_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))
            library = FakeClipLibrary()

            self.assertEqual(self._run_tags(image_path, db, make_clip("best", "blip-large", library)),
                             (["a cat", "best"], False))
            self.assertEqual(self._run_tags(image_path, db, make_clip("best", "blip-large", library)),
                             (["a cat", "best"], True))
            self.assertEqual(self._run_tags(image_path, db, make_clip("fast", "blip-large", library)),
                             (["a cat", "fast"], False))
            self.assertEqual(self._run_tags(image_path, db, make_clip("fast", "blip2-flan-t5-xl", library)),
                             (["a cat", "fast"], False))
            self.assertEqual(library.calls, 3)

    def test_unstamped_clip_cache_is_reprocessed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))

            image_hash = hash_image_content(str(image_path))
            image_id = db.register_image(str(image_path), image_hash, 8, 8, image_path.stat().st_size)
            model_id = db.register_model("FakeCLIP/test", "CLIP")
            db.save_interrogation(image_id, model_id, ["old prompt"], None, "old prompt")

            self.assertEqual(self._run_tags(image_path, db, make_clip()), (["a cat", "best"], False))

    def test_database_without_config_column_is_migrated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            db_path = tmp / "interrogations.db"
            db = InterrogationDatabase(str(db_path))
            image_id = db.register_image("a.png", "hash-a", 8, 8, 10)
            model_id = db.register_model("FakeWD", "WD")
            db.save_interrogation(image_id, model_id, ["cat"], {"cat": 0.9}, "cat")
            conn = sqlite3.connect(db_path)
            conn.execute("ALTER TABLE interrogations DROP COLUMN config")
            conn.close()

            db = InterrogationDatabase(str(db_path))
            legacy = db.get_interrogation("hash-a", "FakeWD")
            self.assertEqual((legacy["tags"], legacy["config"]), (["cat"], None))

            db.save_interrogation(image_id, model_id, ["dog"], {"dog": 0.8}, "dog",
                                  config={"preprocess": 2, "threshold": 0.35})
            saved = db.get_interrogation("hash-a", "FakeWD")
            self.assertEqual((saved["tags"], saved["config"]), (["dog"], {"preprocess": 2, "threshold": 0.35}))

    def test_wd_result_is_reused_on_the_next_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))

            interrogator = FakeTagger("FakeWD", "WD", ["fresh_tag"])
            InterrogationWorker([image_path], interrogator, db, write_files=False).run()

            cached_flags = []
            worker = InterrogationWorker([image_path], interrogator, db, write_files=False)
            worker._runner.on_result = lambda _path, _results, meta: cached_flags.append(meta["cached"])
            worker.run()

            self.assertEqual(interrogator.calls, 1)
            self.assertEqual(cached_flags, [True])
            cached = db.get_interrogation(hash_image_content(str(image_path)), "FakeWD")
            self.assertEqual(cached["result_version"], WDInterrogator.RESULT_VERSION)

    def test_merge_mode_updates_existing_txt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            image_path.with_suffix(".txt").write_text("existing_tag", encoding="utf-8")

            db = InterrogationDatabase(str(tmp / "interrogations.db"))
            interrogator = FakeTagger("FakeCamie", "Camie", ["new_tag"])
            worker = InterrogationWorker(
                [image_path],
                interrogator,
                db,
                write_files=True,
                overwrite_files=False,
            )
            worker.run()

            self.assertEqual(
                image_path.with_suffix(".txt").read_text(encoding="utf-8"),
                "existing_tag, new_tag",
            )

    def test_wd_config_excludes_provider_settings_object(self):
        class FakeProviderSettings:
            def create_inference_session(self, model_path, device):
                return object()

        fake_ort = ModuleType("onnxruntime")
        fake_hf = ModuleType("huggingface_hub")
        fake_hf.hf_hub_download = lambda _model, filename: filename
        fake_pandas = ModuleType("pandas")
        fake_pandas.read_csv = lambda _path: []

        original_modules = {
            name: sys.modules.get(name)
            for name in ("onnxruntime", "huggingface_hub", "pandas")
        }
        sys.modules.update(
            {
                "onnxruntime": fake_ort,
                "huggingface_hub": fake_hf,
                "pandas": fake_pandas,
            }
        )
        try:
            interrogator = WDInterrogator("FakeWD")
            interrogator.load_model(
                threshold=0.35,
                device="cpu",
                provider_settings=FakeProviderSettings(),
            )
        finally:
            for name, module in original_modules.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        with tempfile.TemporaryDirectory() as tmpdir:
            db = InterrogationDatabase(str(Path(tmpdir) / "interrogations.db"))
            model_id = db.register_model(
                interrogator.model_name,
                interrogator.get_model_type(),
                config=interrogator.get_config(),
            )

        self.assertIsInstance(model_id, int)
        self.assertNotIn("provider_settings", interrogator.get_config())


class InterrogationResultVersionTests(unittest.TestCase):
    def test_existing_database_gains_result_version_column(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "interrogations.db"
            conn = sqlite3.connect(db_path)
            conn.executescript(
                """
                CREATE TABLE images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    width INTEGER, height INTEGER, file_size INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL UNIQUE,
                    model_type TEXT NOT NULL,
                    version TEXT, config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE interrogations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    model_id INTEGER NOT NULL,
                    tags TEXT NOT NULL,
                    confidence_scores TEXT,
                    raw_output TEXT,
                    interrogated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(image_id, model_id)
                );
                INSERT INTO images (file_path, file_hash) VALUES ('a.png', 'hash-a');
                INSERT INTO models (model_name, model_type) VALUES ('FakeWD', 'WD');
                INSERT INTO interrogations (image_id, model_id, tags) VALUES (1, 1, '["old_tag"]');
                """
            )
            conn.commit()
            conn.close()

            db = InterrogationDatabase(str(db_path))

            cached = db.get_interrogation("hash-a", "FakeWD")
            self.assertEqual(cached["tags"], ["old_tag"])
            self.assertEqual(cached["result_version"], 0)

    def test_queued_save_keeps_result_version(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db = InterrogationDatabase(str(Path(tmpdir) / "interrogations.db"))
            image_id = db.register_image("a.png", "hash-a", 8, 8, 1)
            model_id = db.register_model("FakeWD", "WD")

            params = {
                "image_id": image_id,
                "model_id": model_id,
                "tags": ["tag"],
                "confidence_scores": {"tag": 0.9},
                "raw_output": "tag",
                "result_version": 1,
            }
            db._execute_queued_operation("op-1", "save_interrogation", params)
            self.assertEqual(db.get_interrogation("hash-a", "FakeWD")["result_version"], 1)

            # Operations queued before the column existed carry no version.
            params.pop("result_version")
            db._execute_queued_operation("op-2", "save_interrogation", params)
            self.assertEqual(db.get_interrogation("hash-a", "FakeWD")["result_version"], 0)


if __name__ == "__main__":
    unittest.main()
