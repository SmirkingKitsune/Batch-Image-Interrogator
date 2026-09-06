"""Unit tests for llama-server log retention.

The server writes to an inherited file descriptor, so these cover the two
mechanisms that bound it: pruning old run logs, and truncating a live one.
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.llama_cpp_runtime import LlamaCppRuntimeManager  # noqa: E402


def write_log(directory: Path, name: str, size: int, mtime: float) -> Path:
    path = directory / name
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


class TestLogPruning(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_logs(self, count: int, size: int = 1024):
        # Oldest first, so index 0 has the earliest mtime.
        return [
            write_log(self.log_dir, f"llama-server-8080-2026010{i}-000000.log", size, 1_700_000_000 + i)
            for i in range(count)
        ]

    def test_recent_logs_are_kept(self):
        paths = self._make_logs(3)
        LlamaCppRuntimeManager._prune_logs(self.log_dir)
        for path in paths:
            self.assertTrue(path.exists(), f"{path.name} should survive")

    def test_oldest_logs_are_deleted_beyond_the_file_budget(self):
        paths = self._make_logs(10)
        LlamaCppRuntimeManager._prune_logs(self.log_dir)
        survivors = sorted(p.name for p in self.log_dir.glob("*.log"))
        # One slot is reserved for the run about to start.
        self.assertEqual(len(survivors), LlamaCppRuntimeManager.LOG_KEEP_FILES - 1)
        # The newest are the ones kept.
        self.assertIn(paths[-1].name, survivors)
        self.assertNotIn(paths[0].name, survivors)

    def test_total_byte_budget_evicts_even_a_small_file_count(self):
        newest = write_log(self.log_dir, "llama-server-8080-20260102-000000.log", 4096, 1_700_000_100)
        older = write_log(self.log_dir, "llama-server-8080-20260101-000000.log", 3072, 1_700_000_000)
        with (
            patch.object(LlamaCppRuntimeManager, "LOG_KEEP_TOTAL_BYTES", 10 * 1024),
            patch.object(LlamaCppRuntimeManager, "LOG_MAX_BYTES", 4 * 1024),
        ):
            LlamaCppRuntimeManager._prune_logs(self.log_dir)
        self.assertTrue(newest.exists(), "newest log should be kept")
        self.assertFalse(older.exists(), "the next live log must have reserved byte budget")

    def test_oversized_historical_log_is_trimmed_before_retention(self):
        newest = write_log(
            self.log_dir,
            "llama-server-8080-20260102-000000.log",
            12 * 1024,
            1_700_000_100,
        )
        with (
            patch.object(LlamaCppRuntimeManager, "LOG_MAX_BYTES", 8 * 1024),
            patch.object(LlamaCppRuntimeManager, "LOG_TAIL_BYTES", 2 * 1024),
        ):
            LlamaCppRuntimeManager._prune_logs(self.log_dir)
        self.assertTrue(newest.exists())
        self.assertLess(newest.stat().st_size, 3 * 1024)

    def test_unrelated_files_are_untouched(self):
        keep = self.log_dir / "notes.txt"
        keep.write_text("not a run log", encoding="utf-8")
        self._make_logs(10)
        LlamaCppRuntimeManager._prune_logs(self.log_dir)
        self.assertTrue(keep.exists())

    def test_missing_directory_is_not_an_error(self):
        LlamaCppRuntimeManager._prune_logs(self.log_dir / "nonexistent")


class TestLiveLogTruncation(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.log_dir = Path(self._tmp.name)
        self.log_path = self.log_dir / "llama-server-8080-20260101-000000.log"

    def tearDown(self):
        self._tmp.cleanup()

    def test_truncation_keeps_the_tail_and_drops_the_head(self):
        head = b"HEAD-MARKER\n" + b"a" * LlamaCppRuntimeManager.LOG_TAIL_BYTES
        tail = b"\nTAIL-MARKER-LINE\n"
        self.log_path.write_bytes(head + tail)

        LlamaCppRuntimeManager._truncate_log(self.log_path)

        content = self.log_path.read_bytes()
        self.assertLess(len(content), len(head + tail))
        self.assertIn(b"TAIL-MARKER-LINE", content)
        self.assertNotIn(b"HEAD-MARKER", content)
        self.assertIn(b"log truncated at", content)

    def test_truncated_log_starts_on_a_line_boundary(self):
        self.log_path.write_bytes(b"x" * (LlamaCppRuntimeManager.LOG_TAIL_BYTES * 2) + b"\ncomplete line\n")
        LlamaCppRuntimeManager._truncate_log(self.log_path)
        body = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        # First line is the marker; nothing after it is a severed fragment of
        # the padding, because the partial leading line is dropped.
        self.assertTrue(body[0].startswith("# [log truncated"))
        self.assertNotIn("x", "".join(body[1:]))

    def test_a_writing_process_keeps_appending_after_truncation(self):
        """The real scenario: truncate a log a live child is writing to.

        Append mode is what makes this work. Without O_APPEND the child would
        keep writing at its old offset and recreate a multi-gigabyte sparse
        file, which is the bug this test pins down.
        """
        handle = self.log_path.open("a", encoding="utf-8", errors="replace")
        try:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import sys,time\n"
                    "for i in range(400):\n"
                    "    sys.stdout.write('line %d\\n' % i)\n"
                    "    sys.stdout.flush()\n"
                    "    time.sleep(0.005)\n",
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            # Wait until the child has definitely started writing, then
            # truncate underneath it and let it finish.
            deadline = time.monotonic() + 5
            while self.log_path.stat().st_size == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
            size_before = self.log_path.stat().st_size
            os.truncate(self.log_path, 0)
            child.wait(timeout=30)
        finally:
            handle.close()

        size_after = self.log_path.stat().st_size
        self.assertGreater(size_before, 0)
        # Writes after the truncation land at the new end, so the file stays
        # small rather than becoming a sparse file the size of the original.
        self.assertLess(size_after, size_before + 4096)
        content = self.log_path.read_text(encoding="utf-8", errors="replace")
        self.assertIn("line 399", content, "child should keep logging after truncation")
        self.assertEqual(content.count("\x00"), 0, "no sparse-file NUL padding")


class TestWatchdogLifecycle(unittest.TestCase):
    def test_watchdog_starts_and_stops(self):
        manager = LlamaCppRuntimeManager()
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "llama-server-8080-20260101-000000.log"
            log_path.write_bytes(b"")
            manager._start_log_watchdog(log_path)
            self.assertIsNotNone(manager._log_watchdog)
            self.assertTrue(manager._log_watchdog.is_alive())
            self.assertTrue(manager._log_watchdog.daemon, "must not block interpreter exit")

            watchdog = manager._log_watchdog
            manager._stop_log_watchdog()
            watchdog.join(timeout=5)
            self.assertFalse(watchdog.is_alive(), "watchdog should exit promptly when signalled")
            self.assertIsNone(manager._log_watchdog)

    def test_stopping_without_a_running_watchdog_is_safe(self):
        LlamaCppRuntimeManager()._stop_log_watchdog()

    def test_starting_a_new_watchdog_fully_stops_the_previous_one(self):
        manager = LlamaCppRuntimeManager()
        with tempfile.TemporaryDirectory() as tmp:
            first_path = Path(tmp) / "llama-server-8080-first.log"
            second_path = Path(tmp) / "llama-server-8080-second.log"
            first_path.write_bytes(b"")
            second_path.write_bytes(b"")

            manager._start_log_watchdog(first_path)
            first_watchdog = manager._log_watchdog
            manager._start_log_watchdog(second_path)

            self.assertIsNotNone(first_watchdog)
            self.assertFalse(first_watchdog.is_alive())
            self.assertIsNot(manager._log_watchdog, first_watchdog)
            manager._stop_log_watchdog()


if __name__ == "__main__":
    unittest.main()
