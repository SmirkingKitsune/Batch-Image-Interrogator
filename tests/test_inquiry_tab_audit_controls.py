import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt6.QtWidgets import QApplication

from core import InterrogationDatabase, TagFilterSettings
from ui.tabs.inquiry_tab import InquiryTab


class InquiryTabAuditControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_audit_disables_text_output_radios_and_preserves_prior_choice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                db = InterrogationDatabase(str(Path(tmpdir) / "interrogations.db"))
                tag_filters = TagFilterSettings(str(Path(tmpdir) / "tag_filters.json"))
                tab = InquiryTab(db, {}, tag_filters)

                tab.no_txt_radio.setChecked(True)
                tab.batch_task_combo.setCurrentText("audit")

                self.assertFalse(tab.no_txt_radio.isEnabled())
                self.assertFalse(tab.merge_txt_radio.isEnabled())
                self.assertFalse(tab.overwrite_txt_radio.isEnabled())
                self.assertTrue(tab.merge_txt_radio.isChecked())
                self.assertEqual(tab._get_txt_output_mode(), "none")

                tab.batch_task_combo.setCurrentText("describe")

                self.assertTrue(tab.no_txt_radio.isEnabled())
                self.assertTrue(tab.merge_txt_radio.isEnabled())
                self.assertTrue(tab.overwrite_txt_radio.isEnabled())
                self.assertTrue(tab.no_txt_radio.isChecked())
            finally:
                os.chdir(old_cwd)

    def test_cancelled_batch_resets_incomplete_rows_and_suppresses_success_popup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                tmp = Path(tmpdir)
                image_paths = []
                for name in ("first.png", "second.png", "third.png"):
                    image_path = tmp / name
                    Image.new("RGB", (8, 8), (20, 40, 60)).save(image_path)
                    image_paths.append(str(image_path))

                db = InterrogationDatabase(str(tmp / "interrogations.db"))
                tag_filters = TagFilterSettings(str(tmp / "tag_filters.json"))
                tab = InquiryTab(db, {}, tag_filters)
                tab.set_images_from_interrogation(image_paths)
                tab.active_batch_total = len(image_paths)

                tab.on_batch_result(image_paths[0], {"raw_output": "", "tags": []})
                tab.on_batch_progress(2, len(image_paths), "Processing: second.png")
                tab.batch_cancel_requested = True

                with patch("ui.tabs.inquiry_tab.QMessageBox.information") as info, patch(
                    "ui.tabs.inquiry_tab.QMessageBox.warning"
                ) as warning:
                    tab.on_batch_finished(True)

                self.assertFalse(info.called)
                self.assertFalse(warning.called)
                self.assertEqual(tab.progress_bar.value(), 1)
                self.assertIn("Batch inquiry canceled after 1 of 3 image(s)", tab.progress_label.text())
                self.assertEqual(tab.image_queue.item(0).text(), "Done - first.png")
                self.assertEqual(tab.image_queue.item(1).text(), "second.png")
                self.assertEqual(tab.image_queue.item(2).text(), "third.png")
            finally:
                # The tab owns a background context scan; stop it before the
                # temporary directory (and its database) is removed.
                tab.close()
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
