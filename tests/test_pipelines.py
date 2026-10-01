"""Tests for the Qt-free batch pipelines behind both front ends."""

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from PIL import Image

from core.base_interrogator import BaseInterrogator
from core.database import InterrogationDatabase
from core.pipelines import (
    InterrogationBatchRunner,
    collect_context_sources,
    scan_image_directory,
)


class FakeTagger(BaseInterrogator):
    def __init__(self, model_type="Camie", tags=None, delay=0.0):
        super().__init__(f"Fake{model_type}")
        self.model_type = model_type
        self.tags = tags or ["cat"]
        self.delay = delay
        self.calls = 0
        self.is_loaded = True

    def load_model(self, **kwargs):
        self.is_loaded = True

    def interrogate(self, image_path, **kwargs):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        return {"tags": list(self.tags), "confidence_scores": {t: 0.9 for t in self.tags}, "raw_output": ""}

    def get_model_type(self):
        return self.model_type

    def get_config(self):
        return {"threshold": 0.35}


def make_images(directory: Path, count: int):
    paths = []
    for index in range(count):
        path = directory / f"image_{index}.png"
        Image.new("RGB", (8, 8), (index * 20 % 255, 40, 60)).save(path)
        paths.append(path)
    return paths


class InterrogationBatchRunnerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.db = InterrogationDatabase(str(self.tmp / "interrogations.db"))

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def test_result_meta_distinguishes_model_runs_from_cache_hits(self):
        images = make_images(self.tmp, 2)
        tagger = FakeTagger("Camie")
        metas = []

        InterrogationBatchRunner(images, tagger, self.db, write_files=False,
                                 on_result=lambda _p, _r, meta: metas.append(meta)).run()
        InterrogationBatchRunner(images, tagger, self.db, write_files=False,
                                 on_result=lambda _p, _r, meta: metas.append(meta)).run()

        self.assertEqual([m["cached"] for m in metas], [False, False, True, True])
        self.assertTrue(all(m["elapsed_ms"] >= 0 for m in metas))
        self.assertEqual(tagger.calls, 2, "the second run is served from the database")

    def test_item_started_reports_each_image_in_order(self):
        images = make_images(self.tmp, 3)
        started = []
        InterrogationBatchRunner(images, FakeTagger(), self.db, write_files=False,
                                 on_item_started=lambda i, p: started.append((i, Path(p).name))).run()
        self.assertEqual(started, [(0, "image_0.png"), (1, "image_1.png"), (2, "image_2.png")])

    def test_pause_holds_the_batch_between_images_until_resumed(self):
        images = make_images(self.tmp, 3)
        results = []
        runner = InterrogationBatchRunner(images, FakeTagger(delay=0.05), self.db, write_files=False)

        def pause_after_first(path, _results, _meta):
            results.append(path)
            if len(results) == 1:
                runner.pause()

        runner.on_result = pause_after_first
        thread = threading.Thread(target=runner.run)
        thread.start()
        time.sleep(0.6)
        self.assertEqual(len(results), 1, "paused after the first image")
        self.assertTrue(runner.is_paused)
        runner.resume()
        thread.join(timeout=5)
        self.assertEqual(len(results), 3)

    def test_cancel_wakes_a_paused_batch(self):
        images = make_images(self.tmp, 3)
        runner = InterrogationBatchRunner(images, FakeTagger(), self.db, write_files=False)
        runner.pause()
        thread = threading.Thread(target=runner.run)
        thread.start()
        time.sleep(0.3)
        runner.cancel()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())


class ScanTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_scan_is_sorted_filtered_and_optionally_recursive(self):
        (self.tmp / "sub").mkdir()
        make_images(self.tmp, 2)
        Image.new("RGB", (4, 4)).save(self.tmp / "sub" / "nested.JPG")
        (self.tmp / "notes.txt").write_text("not an image")

        flat = scan_image_directory(str(self.tmp))
        deep = scan_image_directory(str(self.tmp), recursive=True)

        self.assertEqual([Path(p).name for p in flat], ["image_0.png", "image_1.png"])
        self.assertEqual(len(deep), 3)
        self.assertEqual(deep, sorted(deep))

    def test_scan_rejects_a_missing_directory_and_honours_cancel(self):
        with self.assertRaises(ValueError):
            scan_image_directory(str(self.tmp / "missing"))
        make_images(self.tmp, 1)
        self.assertIsNone(scan_image_directory(str(self.tmp), is_cancelled=lambda: True))

    def test_context_sources_count_images_per_model(self):
        images = make_images(self.tmp, 2)
        db = InterrogationDatabase(str(self.tmp / "db.sqlite"))
        try:
            InterrogationBatchRunner(images, FakeTagger("Camie"), db, write_files=False).run()
            InterrogationBatchRunner(images[:1], FakeTagger("WD"), db, write_files=False).run()
            sources = collect_context_sources([str(p) for p in images], db)
        finally:
            db.close()
        counts = {src["model_type"]: len(src["image_hashes"]) for src in sources}
        self.assertEqual(counts, {"Camie": 2, "WD": 1})
        self.assertEqual(sources[0]["model_type"], "Camie", "widest coverage first")


class NoQtImportTests(unittest.TestCase):
    def test_core_pipelines_do_not_import_qt(self):
        import subprocess

        code = (
            "import sys; import core.pipelines, core.tag_review, core.llama_runtime_info, "
            "core.inquiry_session, core.model_catalog; "
            "sys.exit(1 if any(m.startswith('PyQt6') for m in sys.modules) else 0)"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
