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
                    },
                    "single_task": "vqa",
                    "single_prompt": "What is visible?",
                    "batch_task": "ocr",
                    "batch_prompt": "Read text.",
                    "batch_include_prior_tables": True,
                    "single_include_prior_transcripts": True,
                    "batch_include_prior_transcripts": True,
                    "batch_included_model_types": ["WD"],
                    "batch_context_source_keys": ["WD\u001fWD14"],
                    "batch_carry_context": True,
                    "batch_use_cache": True,
                    "txt_output_mode": "overwrite",
                    "active_tab": 1,
                }
            )

            reloaded = InquirySettings(str(settings_path))

            self.assertEqual(reloaded.get_llama_config()["ctx_size"], 4096)
            self.assertNotIn("included_model_types", reloaded.get_llama_config())
            self.assertEqual(reloaded.get_options()["single_task"], "vqa")
            self.assertTrue(reloaded.get_options()["batch_include_prior_tables"])
            self.assertTrue(reloaded.get_options()["single_include_prior_transcripts"])
            self.assertTrue(reloaded.get_options()["batch_include_prior_transcripts"])
            self.assertEqual(reloaded.get_options()["batch_included_model_types"], ["WD"])
            self.assertEqual(reloaded.get_options()["batch_context_source_keys"], ["WD\u001fWD14"])
            self.assertTrue(reloaded.get_options()["batch_carry_context"])
            self.assertTrue(reloaded.get_options()["batch_use_cache"])
            self.assertTrue(reloaded.has_saved_option("batch_use_cache"))
            self.assertEqual(reloaded.get_options()["txt_output_mode"], "overwrite")
            self.assertEqual(reloaded.get_options()["active_tab"], 1)

    def test_migrates_legacy_batch_context_from_llama_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "inquiry_settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "llama_config": {
                            "ctx_size": 4096,
                            "include_prior_tables": True,
                            "included_model_types": ["CLIP", "WD"],
                            "carry_batch_context": True,
                        }
                    }
                ),
                encoding="utf-8",
            )

            settings = InquirySettings(str(settings_path))
            options = settings.get_options()

            self.assertEqual(settings.get_llama_config(), {"ctx_size": 4096})
            self.assertTrue(options["batch_include_prior_tables"])
            self.assertEqual(options["batch_included_model_types"], ["CLIP", "WD"])
            self.assertTrue(options["batch_carry_context"])

    def test_ignores_invalid_option_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            settings_path = Path(tmpdir) / "inquiry_settings.json"
            settings_path.write_text(
                json.dumps(
                    {
                        "txt_output_mode": "delete",
                        "active_tab": "bad",
                        "batch_use_cache": "yes",
                    }
                ),
                encoding="utf-8",
            )

            settings = InquirySettings(str(settings_path))
            options = settings.get_options()

            self.assertEqual(options["txt_output_mode"], "merge")
            self.assertEqual(options["active_tab"], 0)
            self.assertIsNone(options["batch_use_cache"])
            self.assertFalse(settings.has_saved_option("batch_use_cache"))


if __name__ == "__main__":
    unittest.main()
