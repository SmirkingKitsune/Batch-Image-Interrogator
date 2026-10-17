"""Tests for the Qt-free batch pipelines behind both front ends."""

import json
import sqlite3
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from PIL import Image

from core.base_interrogator import BaseInterrogator
from core.database import LOCAL_DB_NAME, InterrogationDatabase
from core.hashing import hash_image_content
from core.pipelines import (
    InterrogationBatchRunner,
    MultimodalBatchRunner,
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


class MultimodalBatchRunnerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.db = InterrogationDatabase(str(self.tmp / "interrogations.db"))

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def test_cancel_lets_the_image_in_progress_finish(self):
        images = make_images(self.tmp, 3)
        results, errors, seen_cancel = [], [], []
        runner = None

        class CancellingInterrogator(FakeTagger):
            def interrogate(self, image_path, should_cancel=None, **kwargs):
                runner.cancel()  # first click, while this image is running
                seen_cancel.append(should_cancel())
                return super().interrogate(image_path)  # counts the call

        llama = CancellingInterrogator("LlamaCpp")
        runner = MultimodalBatchRunner(
            images, llama, self.db, task="describe", prompt="", write_files=False,
            on_result=lambda path, _r, _m: results.append(path),
            on_error=lambda _path, message: errors.append(message),
        )

        self.assertTrue(runner.run())
        self.assertEqual(llama.calls, 1, "no further image starts")
        self.assertEqual(seen_cancel, [False], "a plain cancel never interrupts the image")
        self.assertEqual(results, [str(images[0])], "the finished image is kept")
        self.assertEqual(errors, [])

    def test_force_cancel_abandons_the_image_in_progress(self):
        from interrogators.llama_cpp_interrogator import LlamaCppInterrogationCancelled

        images = make_images(self.tmp, 3)
        results, errors = [], []
        runner = None

        class StoppableInterrogator(FakeTagger):
            aborted = 0

            def abort_active_request(self):
                self.aborted += 1

            def interrogate(self, image_path, should_cancel=None, **kwargs):
                self.calls += 1
                runner.cancel()
                runner.force_cancel()  # second click
                if should_cancel():
                    raise LlamaCppInterrogationCancelled("Inquiry stopped")
                return super().interrogate(image_path)

        llama = StoppableInterrogator("LlamaCpp")
        runner = MultimodalBatchRunner(
            images, llama, self.db, task="describe", prompt="", write_files=False,
            on_result=lambda path, _r, _m: results.append(path),
            on_error=lambda _path, message: errors.append(message),
        )

        self.assertTrue(runner.run())
        self.assertEqual(llama.calls, 1)
        self.assertEqual(llama.aborted, 1, "the running request is cut off")
        self.assertEqual(results, [])
        self.assertEqual(errors, [], "a force stop is not reported as an error")

    def test_no_repack_joins_the_cache_key_only_when_enabled(self):
        base = {"llama_model_path": "model.gguf", "ctx_size": 4096}
        self.assertNotIn(
            "no_repack", MultimodalBatchRunner.normalize_cache_config({**base, "no_repack": False})
        )
        self.assertTrue(
            MultimodalBatchRunner.normalize_cache_config({**base, "no_repack": True})["no_repack"]
        )


def make_local_db(folder: Path, image: Path, model_name: str, tags):
    """A local-mode database in `folder` holding one result for `image`."""
    local = InterrogationDatabase(str(folder / LOCAL_DB_NAME))
    image_id = local.register_image(str(image), hash_image_content(str(image)), 8, 8, image.stat().st_size)
    model_id = local.register_model(model_name, "Camie")
    local.save_interrogation(image_id, model_id, list(tags), {t: 0.5 for t in tags}, "local raw")
    local.close()
    return folder / LOCAL_DB_NAME


class LocalDatabaseFallbackTests(unittest.TestCase):
    """Global mode consults local databases left in the selected directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.photos = self.tmp / "photos"
        self.photos.mkdir()
        self.db = InterrogationDatabase(str(self.tmp / "global.db"))

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def _run(self, images, tagger):
        metas = []
        InterrogationBatchRunner(images, tagger, self.db, write_files=False,
                                 on_result=lambda _p, _r, meta: metas.append(meta)).run()
        return metas

    def test_result_only_in_the_local_database_is_copied_and_reused(self):
        [image] = make_images(self.photos, 1)
        tagger = FakeTagger("Camie", tags=["model-tag"])
        local_db = make_local_db(self.photos, image, tagger.model_name, ["local-tag"])
        local_bytes = local_db.read_bytes()
        folder_before = sorted(p.name for p in self.photos.iterdir())

        self.db.switch_to_directory(str(self.photos))
        metas = self._run([image], tagger)

        self.assertEqual(tagger.calls, 0, "the local result is reused, not recomputed")
        self.assertTrue(metas[0]["cached"])
        copied = self.db.get_interrogation(hash_image_content(str(image)), tagger.model_name)
        self.assertEqual(copied["tags"], ["local-tag"])
        self.assertEqual(copied["raw_output"], "local raw")
        # The local database is only ever read.
        self.assertEqual(local_db.read_bytes(), local_bytes)
        self.assertEqual(sorted(p.name for p in self.photos.iterdir()), folder_before)

    def test_global_entry_is_kept_over_the_local_one(self):
        [image] = make_images(self.photos, 1)
        tagger = FakeTagger("Camie", tags=["global-tag"])
        make_local_db(self.photos, image, tagger.model_name, ["local-tag"])
        self._run([image], tagger)  # global entry exists before the directory is selected
        self.assertEqual(tagger.calls, 1)

        self.db.switch_to_directory(str(self.photos))
        self.assertFalse(self.db.import_missing_interrogation(
            str(image), hash_image_content(str(image)), tagger.model_name))
        self._run([image], tagger)

        self.assertEqual(tagger.calls, 1)
        kept = self.db.get_interrogation(hash_image_content(str(image)), tagger.model_name)
        self.assertEqual(kept["tags"], ["global-tag"])

    def test_subfolder_database_is_found_from_the_selected_directory(self):
        sub = self.photos / "cats"
        sub.mkdir()
        [image] = make_images(sub, 1)
        tagger = FakeTagger("Camie")
        make_local_db(sub, image, tagger.model_name, ["from-subfolder"])

        self.db.switch_to_directory(str(self.photos))
        self._run([image], tagger)

        self.assertEqual(tagger.calls, 0)
        copied = self.db.get_interrogation(hash_image_content(str(image)), tagger.model_name)
        self.assertEqual(copied["tags"], ["from-subfolder"])

    def test_outdated_local_result_is_copied_then_reprocessed(self):
        class NewerTagger(FakeTagger):
            RESULT_VERSION = 2

        [image] = make_images(self.photos, 1)
        tagger = NewerTagger("Camie", tags=["fresh"])
        make_local_db(self.photos, image, tagger.model_name, ["stale"])

        self.db.switch_to_directory(str(self.photos))
        metas = self._run([image], tagger)

        self.assertEqual(tagger.calls, 1, "the usual cache rules still apply to copied results")
        self.assertFalse(metas[0]["cached"])
        saved = self.db.get_interrogation(hash_image_content(str(image)), tagger.model_name)
        self.assertEqual(saved["tags"], ["fresh"])

    def test_local_database_open_elsewhere_is_read_with_its_newest_writes(self):
        [image] = make_images(self.photos, 1)
        file_hash = hash_image_content(str(image))
        local_db = make_local_db(self.photos, image, "FakeCamie", ["older"])

        # Another instance in local mode has the database open; its latest write
        # is still only in the -wal file.
        writer = sqlite3.connect(local_db)
        try:
            writer.execute("UPDATE interrogations SET tags = '[\"newer\"]'")
            writer.commit()
            self.assertTrue(local_db.with_name(LOCAL_DB_NAME + "-wal").exists())

            self.db.switch_to_directory(str(self.photos))
            self.assertTrue(self.db.import_missing_interrogation(str(image), file_hash, "FakeCamie"))
        finally:
            writer.close()
        self.assertEqual(self.db.get_interrogation(file_hash, "FakeCamie")["tags"], ["newer"])

    def test_database_from_before_result_versions_is_still_read(self):
        [image] = make_images(self.photos, 1)
        file_hash = hash_image_content(str(image))
        conn = sqlite3.connect(self.photos / LOCAL_DB_NAME)
        conn.executescript("""
            CREATE TABLE images (id INTEGER PRIMARY KEY, file_path TEXT, file_hash TEXT UNIQUE,
                                 width INTEGER, height INTEGER, file_size INTEGER);
            CREATE TABLE models (id INTEGER PRIMARY KEY, model_name TEXT UNIQUE, model_type TEXT,
                                 version TEXT, config TEXT);
            CREATE TABLE interrogations (id INTEGER PRIMARY KEY, image_id INTEGER, model_id INTEGER,
                                         tags TEXT, confidence_scores TEXT, raw_output TEXT,
                                         interrogated_at TIMESTAMP, UNIQUE(image_id, model_id));
        """)
        conn.execute("INSERT INTO images VALUES (1, ?, ?, 8, 8, 1)", (str(image), file_hash))
        conn.execute("INSERT INTO models VALUES (1, 'FakeCamie', 'Camie', NULL, NULL)")
        conn.execute("""INSERT INTO interrogations VALUES
                        (1, 1, 1, '["old-schema"]', NULL, 'raw', '2025-01-01 00:00:00')""")
        conn.commit()
        conn.close()

        self.db.switch_to_directory(str(self.photos))
        self.assertTrue(self.db.import_missing_interrogation(str(image), file_hash, "FakeCamie"))
        copied = self.db.get_interrogation(file_hash, "FakeCamie")
        self.assertEqual(copied["tags"], ["old-schema"])
        self.assertEqual(copied["interrogated_at"], "2025-01-01 00:00:00")
        self.assertEqual(copied["result_version"], 0)

    def test_nothing_is_imported_from_outside_the_selected_directory(self):
        [image] = make_images(self.photos, 1)
        file_hash = hash_image_content(str(image))
        make_local_db(self.photos, image, "FakeCamie", ["local-tag"])

        elsewhere = self.tmp / "elsewhere"
        elsewhere.mkdir()
        self.db.switch_to_directory(str(elsewhere))
        self.assertFalse(self.db.import_missing_interrogation(str(image), file_hash, "FakeCamie"))


class LocalModeGlobalFallbackTests(unittest.TestCase):
    """Local mode consults the global database and writes only locally."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.photos = self.tmp / "photos"
        self.photos.mkdir()
        self.global_path = self.tmp / "global.db"
        self.db = InterrogationDatabase(str(self.global_path))

    def tearDown(self):
        self.db.close()
        self._tmp.cleanup()

    def _run(self, images, tagger):
        metas = []
        InterrogationBatchRunner(images, tagger, self.db, write_files=False,
                                 on_result=lambda _p, _r, meta: metas.append(meta)).run()
        return metas

    def _use_local_mode(self):
        self.db.set_local_mode(True)
        self.db.switch_to_directory(str(self.photos))

    @staticmethod
    def _tags_in(db_path: Path, image: Path, model_name: str):
        """Tags stored for an image in a database file, or None."""
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("""
                SELECT i.tags FROM interrogations i
                JOIN images img ON i.image_id = img.id JOIN models m ON i.model_id = m.id
                WHERE img.file_hash = ? AND m.model_name = ?
            """, (hash_image_content(str(image)), model_name)).fetchone()
        finally:
            conn.close()
        return json.loads(row[0]) if row else None

    def test_result_only_in_the_global_database_is_copied_locally(self):
        [image] = make_images(self.photos, 1)
        tagger = FakeTagger("Camie", tags=["global-tag"])
        self._run([image], tagger)  # global mode, so this lands in the global database
        self.assertEqual(tagger.calls, 1)

        self._use_local_mode()
        metas = self._run([image], tagger)

        self.assertEqual(tagger.calls, 1, "the global result is reused, not recomputed")
        self.assertTrue(metas[0]["cached"])
        local_path = self.photos / LOCAL_DB_NAME
        self.assertEqual(self._tags_in(local_path, image, tagger.model_name), ["global-tag"])

    def test_a_miss_in_both_writes_only_to_the_local_database(self):
        [image] = make_images(self.photos, 1)
        tagger = FakeTagger("Camie", tags=["new-tag"])

        self._use_local_mode()
        self._run([image], tagger)

        self.assertEqual(tagger.calls, 1)
        self.assertEqual(self._tags_in(self.photos / LOCAL_DB_NAME, image, tagger.model_name), ["new-tag"])
        self.assertIsNone(self._tags_in(self.global_path, image, tagger.model_name))

    def test_local_entry_is_kept_over_the_global_one(self):
        [image] = make_images(self.photos, 1)
        make_local_db(self.photos, image, "FakeCamie", ["local-tag"])
        tagger = FakeTagger("Camie", tags=["global-tag"])
        self._run([image], tagger)  # global mode: the global database gets its own entry

        self._use_local_mode()
        self._run([image], tagger)

        self.assertEqual(tagger.calls, 1)
        self.assertEqual(self._tags_in(self.photos / LOCAL_DB_NAME, image, "FakeCamie"), ["local-tag"])
        self.assertEqual(self._tags_in(self.global_path, image, "FakeCamie"), ["global-tag"])

    def test_global_database_is_not_a_source_before_a_directory_is_selected(self):
        [image] = make_images(self.photos, 1)
        self.db.set_local_mode(True)  # no directory yet, so the global database is active
        self.assertFalse(self.db.import_missing_interrogation(
            str(image), hash_image_content(str(image)), "FakeCamie"))


class DatabaseModePersistenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "global.db"

    def tearDown(self):
        self._tmp.cleanup()

    def _reopen_mode(self) -> bool:
        db = InterrogationDatabase(str(self.path))
        try:
            return db.use_local_db
        finally:
            db.close()

    def test_last_mode_is_restored_on_the_next_start(self):
        self.assertFalse(self._reopen_mode(), "global until a mode is chosen")

        db = InterrogationDatabase(str(self.path))
        db.set_local_mode(True)
        db.close()
        self.assertTrue(self._reopen_mode())

        db = InterrogationDatabase(str(self.path))
        db.set_local_mode(False)
        db.close()
        self.assertFalse(self._reopen_mode())

    def test_settings_live_beside_the_global_database(self):
        db = InterrogationDatabase(str(self.path))
        db.set_local_mode(True)
        db.close()
        saved = json.loads((self.path.parent / "database_settings.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, {"use_local_db": True})

    def test_unreadable_settings_fall_back_to_global(self):
        (self.path.parent / "database_settings.json").write_text("{not json", encoding="utf-8")
        self.assertFalse(self._reopen_mode())


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
