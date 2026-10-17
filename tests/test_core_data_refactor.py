"""Regression tests for core data lifecycle, queue, file, and hash hardening."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time
from unittest import mock

import pytest

from core.database import InterrogationDatabase
from core.db_queue import (
    DatabaseOperationQueue,
    QueuePersistenceError,
)
from core.file_manager import FileManager
import core.hashing as hashing
from ui.workers import OrganizationWorker


def _queue_params():
    return {
        "image_id": 1,
        "model_id": 2,
        "tags": ["cat"],
        "confidence_scores": None,
        "raw_output": None,
    }


class TestDatabaseLifecycle:
    def test_switching_back_restores_custom_global_path(self, tmp_path):
        global_path = tmp_path / "custom-global.db"
        local_dir = tmp_path / "images"
        local_dir.mkdir()
        database = InterrogationDatabase(str(global_path))

        database.set_local_mode(True)
        database.switch_to_directory(str(local_dir))
        assert database.db_path == local_dir / ".interrogations.db"

        database.set_local_mode(False)
        database.switch_to_directory(str(local_dir))
        assert database.db_path == global_path
        assert database.get_queue_status()["database_path"] == str(global_path)

    def test_additive_schema_upgrade_preserves_existing_turn(self, tmp_path):
        db_path = tmp_path / "legacy.db"
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE multimodal_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    turn_index INTEGER NOT NULL,
                    prompt_type TEXT NOT NULL,
                    prompt_text TEXT,
                    included_tables_json TEXT,
                    response_json TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    reasoning_summary TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(session_id, turn_index)
                )
                """
            )
            conn.execute(
                """
                INSERT INTO multimodal_turns (
                    session_id, turn_index, prompt_type, response_json,
                    tags_json, reasoning_summary
                ) VALUES (1, 0, 'describe', '{}', '[]', 'kept')
                """
            )

        InterrogationDatabase(str(db_path))
        with sqlite3.connect(db_path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(multimodal_turns)")}
            assert {"included_transcripts_json", "sidecar_tags_json"} <= columns
            assert conn.execute(
                "SELECT reasoning_summary FROM multimodal_turns"
            ).fetchone()[0] == "kept"
            assert conn.execute("PRAGMA user_version").fetchone()[0] == 1

    def test_write_transaction_rolls_back_and_foreign_keys_are_enabled(self, tmp_path):
        database = InterrogationDatabase(str(tmp_path / "transaction.db"))
        with pytest.raises(RuntimeError):
            with database._get_connection(write=True) as conn:
                conn.execute(
                    """
                    INSERT INTO images (file_path, file_hash, width, height, file_size)
                    VALUES ('image.png', 'rolled-back', 1, 1, 1)
                    """
                )
                raise RuntimeError("abort transaction")

        with database._get_connection() as conn:
            assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert conn.execute(
                "SELECT COUNT(*) FROM images WHERE file_hash = 'rolled-back'"
            ).fetchone()[0] == 0

    def test_concurrent_instances_append_unique_turn_indexes(self, tmp_path):
        db_path = tmp_path / "shared.db"
        first = InterrogationDatabase(str(db_path))
        second = InterrogationDatabase(str(db_path))
        image_id = first.register_image("image.png", "hash", 1, 1, 1)
        model_id = first.register_model("model", "test")
        session_id = first.create_or_get_multimodal_session(
            image_id, model_id, "single", "session"
        )

        def append(index):
            database = first if index % 2 == 0 else second
            return database.append_multimodal_turn(
                session_id=session_id,
                prompt_type="describe",
                prompt_text=str(index),
                included_tables=[],
                response_json={"index": index},
                tags=[],
                reasoning_summary="",
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            turn_ids = list(executor.map(append, range(24)))

        assert len(set(turn_ids)) == 24
        with sqlite3.connect(db_path) as conn:
            indexes = [
                row[0]
                for row in conn.execute(
                    "SELECT turn_index FROM multimodal_turns ORDER BY turn_index"
                )
            ]
        assert indexes == list(range(24))


class TestDurableQueue:
    def test_failed_mutations_roll_back_in_memory(self, tmp_path):
        queue = DatabaseOperationQueue(tmp_path / "queue.json", tmp_path / "db.sqlite")
        operation_id = queue.add_operation("save_interrogation", _queue_params())

        with mock.patch.object(queue, "_save_queue", return_value=False):
            with pytest.raises(QueuePersistenceError):
                queue.add_operation("save_interrogation", _queue_params())
            assert queue.get_pending_count() == 1

            with pytest.raises(QueuePersistenceError):
                queue.remove_operation(operation_id)
            assert queue.get_pending_count() == 1

            with pytest.raises(QueuePersistenceError):
                queue.mark_failed(operation_id, "failed")
            pending = queue.get_pending_operations()[0]
            assert pending.retry_count == 0
            assert pending.last_error is None

            with pytest.raises(QueuePersistenceError):
                queue.clear_queue()
            assert queue.get_pending_count() == 1

    def test_completed_operation_stays_removed_after_restart(self, tmp_path):
        path = tmp_path / "queue.json"
        db_path = tmp_path / "db.sqlite"
        queue = DatabaseOperationQueue(path, db_path)
        operation_id = queue.add_operation("save_interrogation", _queue_params())
        assert queue.mark_completed(operation_id)
        assert DatabaseOperationQueue(path, db_path).get_pending_count() == 0

    def test_matching_legacy_queue_is_adopted(self, tmp_path):
        db_path = tmp_path / "interrogations.db"
        legacy_path = tmp_path / "db_queue.json"
        scoped_path = tmp_path / "interrogations.db.queue.json"
        legacy_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "database_path": str(db_path),
                    "saved_at": "now",
                    "queue": [
                        {
                            "id": "legacy-id",
                            "operation": "save_interrogation",
                            "params": _queue_params(),
                            "timestamp": "now",
                            "retry_count": 0,
                            "last_error": None,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        queue = DatabaseOperationQueue(scoped_path, db_path, legacy_path)
        assert scoped_path.exists()
        assert [item.id for item in queue.get_pending_operations()] == ["legacy-id"]

    def test_databases_in_same_directory_use_independent_queue_files(self, tmp_path):
        first = InterrogationDatabase(str(tmp_path / "one.db"))
        second = InterrogationDatabase(str(tmp_path / "two.db"))
        first._operation_queue.add_operation("vacuum", {})
        assert first.get_queue_status()["pending_count"] == 1
        assert second.get_queue_status()["pending_count"] == 0
        assert (tmp_path / "one.db.queue.json").exists()
        assert not (tmp_path / "two.db.queue.json").exists()

    def test_failed_atomic_replace_cleans_temp_file(self, tmp_path):
        queue_path = tmp_path / "queue.json"
        queue = DatabaseOperationQueue(queue_path, tmp_path / "db.sqlite")
        # The queue saves through core.atomic_write, which owns the temp file.
        with mock.patch("core.atomic_write.os.replace", side_effect=OSError("denied")):
            with pytest.raises(QueuePersistenceError):
                queue.add_operation("vacuum", {})
        assert not list(tmp_path.glob("*.tmp"))
        assert queue.get_pending_operations() == []


class TestFileOrganization:
    @staticmethod
    def _image_with_tags(root: Path):
        image = root / "image.png"
        image.write_bytes(b"image")
        image.with_suffix(".txt").write_text("cat, indoor", encoding="utf-8")
        return image

    def test_invalid_and_empty_rules_never_move(self, tmp_path):
        image = self._image_with_tags(tmp_path)
        assert FileManager.organize_by_tags_destination(
            image, [], "organized", match_mode="all"
        ) is None
        with pytest.raises(ValueError):
            FileManager.organize_by_tags_destination(
                image, ["cat"], "organized", match_mode="sometimes"
            )
        with pytest.raises(ValueError):
            FileManager.organize_by_tags_destination(image, ["cat"], "../outside")
        assert image.exists()

    def test_sidecar_collision_selects_shared_suffix(self, tmp_path):
        image = self._image_with_tags(tmp_path)
        target = tmp_path / "organized"
        target.mkdir()
        (target / "image.txt").write_text("existing", encoding="utf-8")

        destination = FileManager.organize_by_tags_destination(
            image, ["cat"], "organized"
        )
        assert destination == target / "image_1.png"
        assert (target / "image_1.txt").read_text(encoding="utf-8") == "cat, indoor"
        assert (target / "image.txt").read_text(encoding="utf-8") == "existing"

    def test_worker_emits_actual_collision_adjusted_destination(self, tmp_path):
        image = self._image_with_tags(tmp_path)
        target = tmp_path / "organized"
        target.mkdir()
        (target / "image.png").write_bytes(b"existing")
        emitted = []
        worker = OrganizationWorker([image], ["cat"], "organized")
        worker.moved.connect(lambda source, destination: emitted.append((source, destination)))
        worker.run()
        assert emitted == [(str(image), str(target / "image_1.png"))]

    def test_read_error_prevents_destructive_operations(self, tmp_path):
        image = self._image_with_tags(tmp_path)
        with mock.patch.object(
            FileManager, "_read_tags_strict", side_effect=OSError("read failed")
        ):
            with pytest.raises(OSError):
                FileManager.delete_tags_from_file(image, ["cat"])
            with pytest.raises(OSError):
                FileManager.organize_by_tags_destination(image, ["cat"], "organized")
        assert image.exists()
        assert image.with_suffix(".txt").read_text(encoding="utf-8") == "cat, indoor"

    def test_double_move_failure_reports_final_locations(self, tmp_path):
        image = self._image_with_tags(tmp_path)
        with mock.patch(
            "core.file_manager.shutil.move",
            side_effect=[None, OSError("sidecar failed"), OSError("rollback failed")],
        ):
            with pytest.raises(RuntimeError, match="image remains at"):
                FileManager.organize_by_tags_destination(
                    image, ["cat"], "organized"
                )


class TestHashCache:
    def setup_method(self):
        with hashing._hash_cache_lock:
            hashing._hash_cache.clear()

    def test_same_size_rewrite_invalidates_cache(self, tmp_path):
        path = tmp_path / "image.bin"
        path.write_bytes(b"first")
        first = hashing.hash_image_content(str(path))
        time.sleep(0.01)
        path.write_bytes(b"other")
        second = hashing.hash_image_content(str(path))
        assert first != second

    def test_canonical_aliases_share_one_cache_entry(self, tmp_path):
        path = tmp_path / "image.bin"
        path.write_bytes(b"content")
        alias = str(path.parent / "." / path.name)
        assert hashing.hash_image_content(str(path)) == hashing.hash_image_content(alias)
        assert len(hashing._hash_cache) == 1

    def test_file_changed_while_hashing_is_not_cached(self, tmp_path):
        path = tmp_path / "image.bin"
        path.write_bytes(b"content")
        real_identity = hashing._identity
        calls = []

        def shifting_identity(image_path):
            key = real_identity(image_path)
            calls.append(key)
            # The post-read check sees a different modification time.
            return key if len(calls) == 1 else (key[0], key[1] + 1, key[2])

        with mock.patch.object(hashing, "_identity", side_effect=shifting_identity):
            hashing.hash_image_content(str(path))
        assert len(calls) == 2
        assert len(hashing._hash_cache) == 0

    def test_cache_is_bounded_and_safe_under_concurrent_reads(self, tmp_path):
        paths = []
        for index in range(3):
            path = tmp_path / f"{index}.bin"
            path.write_bytes(str(index).encode())
            paths.append(path)

        with mock.patch.object(hashing, "_HASH_CACHE_MAX", 2):
            for path in paths:
                hashing.hash_image_content(str(path))
            assert len(hashing._hash_cache) == 2

        with ThreadPoolExecutor(max_workers=8) as executor:
            digests = list(
                executor.map(
                    hashing.hash_image_content,
                    [str(paths[-1])] * 32,
                )
            )
        assert len(set(digests)) == 1
