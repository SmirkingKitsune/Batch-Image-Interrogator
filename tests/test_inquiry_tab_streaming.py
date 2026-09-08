"""Tests for the Inquiry tab's streaming single-image flow."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt6.QtWidgets import QApplication, QMessageBox

from core import InterrogationDatabase, TagFilterSettings
from ui.tabs.inquiry_tab import InquiryTab
from ui.main_window import MainWindow


class _FakeInterrogator:
    model_name = "LlamaCpp/fake.gguf"
    is_loaded = True

    def __init__(self, fail: bool = False, tag_count: int = 40):
        self.calls = 0
        self.fail = fail
        self.tag_count = tag_count
        self.streamed = False

    def get_model_type(self):
        return "LlamaCpp"

    def get_config(self):
        return {}

    def interrogate(self, image_path, on_stream_delta=None, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("llama-server exploded")
        if on_stream_delta is not None:
            self.streamed = True
            raw = ""
            for piece in ('{"comment": "', "partial answer", '", "tags": []}'):
                raw += piece
                on_stream_delta(raw)
        response = {
            "comment": f"answer {self.calls}",
            "tags": [f"tag{self.calls}_{n}" for n in range(self.tag_count)],
            "warnings": [],
        }
        return {
            "tags": response["tags"],
            "confidence_scores": None,
            "raw_output": f"raw {self.calls}",
            "multimodal_response": response,
        }

    def set_session_history(self, *args, **kwargs):
        pass

    def reset_session(self, *args, **kwargs):
        pass


class InquiryTabStreamingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _pump_until(self, predicate, timeout=15.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate():
                return True
            time.sleep(0.005)
        return False

    def _build_tab(self, tmp: Path, interrogator) -> InquiryTab:
        image_path = tmp / "image.png"
        Image.new("RGB", (16, 16), (20, 40, 60)).save(image_path)

        db = InterrogationDatabase(str(tmp / "interrogations.db"))
        tag_filters = TagFilterSettings(str(tmp / "tag_filters.json"))
        tab = InquiryTab(db, {}, tag_filters)
        tab.current_interrogator = interrogator
        tab.set_directory_context(str(tmp), False)
        tab.set_images_from_interrogation([str(image_path)])
        tab.mode_tabs.setCurrentIndex(0)
        return tab

    def test_consecutive_turns_all_reach_the_transcript(self):
        """Regression: only the first turn used to stay visible."""
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                tab = self._build_tab(Path(tmpdir), _FakeInterrogator())
                for expected in (1, 2, 3):
                    tab.send_single_inquiry()
                    self.assertTrue(self._pump_until(lambda: tab.single_inquiry_worker is None))
                    self.assertEqual(tab.single_transcript.count(), expected)

                history = tab.database.get_multimodal_history(
                    image_hash=tab.current_image_hash,
                    model_name="LlamaCpp/fake.gguf",
                )
                self.assertEqual(len(history), 3)
            finally:
                os.chdir(old_cwd)

    def test_request_card_is_shown_before_the_model_answers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                tab = self._build_tab(Path(tmpdir), _FakeInterrogator())
                tab.send_single_inquiry()

                # The worker has not been given a chance to finish yet.
                self.assertEqual(tab.single_transcript.count(), 1)
                self.assertIsNotNone(tab.pending_single_card)
                self.assertFalse(tab.send_single_button.isEnabled())

                self.assertTrue(self._pump_until(lambda: tab.single_inquiry_worker is None))
                self.assertTrue(tab.send_single_button.isEnabled())
            finally:
                os.chdir(old_cwd)

    def test_model_changes_are_blocked_until_queued_completion_is_processed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                interrogator = _FakeInterrogator()
                tab = self._build_tab(Path(tmpdir), interrogator)
                tab.send_single_inquiry()
                # Finish the native thread without delivering its queued signals.
                self.assertTrue(tab.single_inquiry_worker.wait(5000))
                self.assertFalse(tab.single_inquiry_shutdown_ready())
                with patch.object(tab, "get_llama_config") as config, patch.object(
                    QMessageBox, "critical"
                ) as error:
                    tab.load_model()
                    tab.unload_model()
                    config.assert_not_called()
                    error.assert_not_called()
                self.assertIs(tab.current_interrogator, interrogator)
                self.assertFalse(tab.load_model_button.isEnabled())
                self.assertFalse(tab.unload_model_button.isEnabled())
                self.assertTrue(self._pump_until(lambda: tab.single_inquiry_worker is None))
                self.assertTrue(tab.single_inquiry_shutdown_ready())
                self.assertTrue(tab.load_model_button.isEnabled())
                self.assertTrue(tab.unload_model_button.isEnabled())
            finally:
                os.chdir(old_cwd)

    def test_completion_uses_captured_model_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                tab = self._build_tab(Path(tmpdir), _FakeInterrogator())
                tab.send_single_inquiry()
                self.assertTrue(tab.single_inquiry_worker.wait(5000))
                replacement = _FakeInterrogator()
                replacement.model_name = "LlamaCpp/replacement.gguf"
                tab.current_interrogator = replacement
                self.assertTrue(self._pump_until(lambda: tab.single_inquiry_worker is None))
                history = tab.database.get_multimodal_history(
                    image_hash=tab.current_image_hash, model_name="LlamaCpp/fake.gguf"
                )
                self.assertEqual(len(history), 1)
                self.assertFalse(tab.database.get_multimodal_history(
                    image_hash=tab.current_image_hash, model_name=replacement.model_name
                ))
            finally:
                os.chdir(old_cwd)

    def test_main_window_defers_cleanup_until_inquiry_is_persisted(self):
        inquiry = Mock()
        inquiry.single_inquiry_shutdown_ready.side_effect = [False, True]
        window = SimpleNamespace(
            inquiry_tab=inquiry, interrogation_tab=None, current_interrogator=None,
            gallery_tab=None, _clip_model_worker=None, database=Mock(),
            statusBar=Mock(return_value=Mock()), close=Mock(),
        )
        event = Mock()
        with patch("ui.main_window.QTimer.singleShot") as retry:
            MainWindow.closeEvent(window, event)
            event.ignore.assert_called_once()
            event.accept.assert_not_called()
            inquiry.current_interrogator.unload_model.assert_not_called()
            window.database.close.assert_not_called()
            retry.assert_called_once_with(100, window.close)
            MainWindow.closeEvent(window, event)
        inquiry.current_interrogator.unload_model.assert_called_once()
        window.database.close.assert_called_once()
        event.accept.assert_called_once()

    def test_response_is_streamed_into_the_pending_card(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                interrogator = _FakeInterrogator()
                tab = self._build_tab(Path(tmpdir), interrogator)

                seen = []
                original = tab._on_single_stream_delta
                tab._on_single_stream_delta = lambda text: (seen.append(text), original(text))[1]

                tab.send_single_inquiry()
                self.assertTrue(self._pump_until(lambda: tab.single_inquiry_worker is None))

                self.assertTrue(interrogator.streamed)
                self.assertTrue(any("partial answer" in text for text in seen))
            finally:
                os.chdir(old_cwd)

    def test_failure_replaces_the_pending_card_with_an_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                tab = self._build_tab(Path(tmpdir), _FakeInterrogator(fail=True))
                with patch.object(QMessageBox, "critical", return_value=None):
                    tab.send_single_inquiry()
                    self.assertTrue(self._pump_until(lambda: tab.single_inquiry_worker is None))

                self.assertEqual(tab.single_transcript.count(), 1)
                self.assertIsNone(tab.pending_single_card)
                self.assertTrue(tab.send_single_button.isEnabled())
            finally:
                os.chdir(old_cwd)

    def test_switching_image_mid_request_does_not_break_completion(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                tmp = Path(tmpdir)
                tab = self._build_tab(tmp, _FakeInterrogator())
                second = tmp / "other.png"
                Image.new("RGB", (16, 16), (90, 10, 10)).save(second)

                tab.send_single_inquiry()
                # Clearing the transcript is what an image switch does.
                tab.single_transcript.clear()
                self.assertTrue(self._pump_until(lambda: tab.single_inquiry_worker is None))

                history = tab.database.get_multimodal_history(
                    image_hash=tab.current_image_hash,
                    model_name="LlamaCpp/fake.gguf",
                )
                self.assertEqual(len(history), 1)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
