import json
import tempfile
import unittest
from pathlib import Path

from core.inquiry_settings import InquirySettings


class InquirySettingsTests(unittest.TestCase):
    def test_loads_and_saves_inquiry_options(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "inquiry_settings.json"
            settings = InquirySettings(str(settings_path))

            settings.update_options(
                {
                    "llama_config": {
                        "llama_binary_path": "llama-server",
                        "ctx_size": 4096,
                        "included_model_types": ["WD"],
                    },
                    "single_task": "vqa",
                    "single_prompt": "What is visible?",
                    "batch_task": "ocr",
                    "batch_prompt": "Read text.",
                    "txt_output_mode": "overwrite",
                    "active_tab": 1,
                }
            )

            reloaded = InquirySettings(str(settings_path))

            self.assertEqual(reloaded.get_llama_config()["ctx_size"], 4096)
            self.assertEqual(reloaded.get_options()["single_task"], "vqa")
            self.assertEqual(reloaded.get_options()["txt_output_mode"], "overwrite")
            self.assertEqual(reloaded.get_options()["active_tab"], 1)

    def test_ignores_invalid_option_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "inquiry_settings.json"
            settings_path.write_text(
                json.dumps({"txt_output_mode": "delete", "active_tab": "bad"}),
                encoding="utf-8",
            )

            settings = InquirySettings(str(settings_path))
            options = settings.get_options()

            self.assertEqual(options["txt_output_mode"], "merge")
            self.assertEqual(options["active_tab"], 0)


if __name__ == "__main__":
    unittest.main()
