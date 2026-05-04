"""Regression tests for classic interrogation worker cache and txt output behavior."""

import tempfile
import unittest
import sys
from pathlib import Path
from types import ModuleType

from PIL import Image

from core.database import InterrogationDatabase
from core.hashing import hash_image_content
from core.base_interrogator import BaseInterrogator
from interrogators.wd_interrogator import WDInterrogator
from ui.workers import InterrogationWorker


class FakeTagger(BaseInterrogator):
    def __init__(self, model_name="FakeWD", model_type="WD", tags=None):
        super().__init__(model_name)
        self.model_type = model_type
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

            interrogator = FakeTagger("FakeWD", "WD", ["fresh_tag"])
            worker = InterrogationWorker([image_path], interrogator, db, write_files=False)
            worker.run()

            self.assertEqual(interrogator.calls, 1)
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
            db.save_interrogation(image_id, model_id, ["stale_tag"], {"stale_tag": 0.4}, "stale_tag")

            interrogator = FakeTagger("FakeWD", "WD", ["fresh_tag"])
            worker = InterrogationWorker([image_path], interrogator, db, write_files=False)
            worker.run()

            self.assertEqual(interrogator.calls, 1)
            cached = db.get_interrogation(image_hash, "FakeWD")
            self.assertEqual(cached["tags"], ["fresh_tag"])

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


if __name__ == "__main__":
    unittest.main()
