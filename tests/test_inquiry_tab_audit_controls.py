import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

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


if __name__ == "__main__":
    unittest.main()
