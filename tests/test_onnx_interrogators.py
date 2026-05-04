"""Focused tests for ONNX tagger preprocessing and metadata parsing."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from interrogators.camie_interrogator import CamieInterrogator
from interrogators.wd_interrogator import WDInterrogator


class OnnxInterrogatorTests(unittest.TestCase):
    def test_wd_preprocess_uses_bgr_255_nhwc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "pixel.png"
            Image.new("RGB", (1, 1), (10, 20, 30)).save(image_path)

            interrogator = WDInterrogator()
            interrogator.target_size = 1

            image_array = interrogator.preprocess_image(str(image_path))

            self.assertEqual(image_array.shape, (1, 1, 1, 3))
            np.testing.assert_array_equal(
                image_array[0, 0, 0],
                np.array([30.0, 20.0, 10.0], dtype=np.float32),
            )

    def test_camie_normalizes_v2_tag_mapping_metadata(self):
        metadata = {
            "dataset_info": {
                "tag_mapping": {
                    "idx_to_tag": {"0": "rating_general", "1": "1girl", "2": "blue_hair"},
                    "tag_to_category": {
                        "rating_general": "rating",
                        "1girl": "general",
                        "blue_hair": "general",
                    },
                }
            }
        }

        interrogator = CamieInterrogator()
        normalized = interrogator._normalize_metadata(metadata)

        self.assertEqual(
            normalized["tags"],
            [
                {"name": "rating_general", "category": "rating"},
                {"name": "1girl", "category": "general"},
                {"name": "blue_hair", "category": "general"},
            ],
        )

    def test_camie_interrogate_maps_probabilities_to_normalized_tags(self):
        class FakeModel:
            def get_inputs(self):
                return [SimpleNamespace(name="input")]

            def run(self, _outputs, _inputs):
                return [np.array([[-10.0, 2.0, 3.0]], dtype=np.float32)]

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "image.png"
            Image.new("RGB", (8, 8), (120, 80, 40)).save(image_path)

            interrogator = CamieInterrogator()
            interrogator.is_loaded = True
            interrogator.model = FakeModel()
            interrogator.tags_data = interrogator._normalize_metadata(
                {
                    "dataset_info": {
                        "tag_mapping": {
                            "idx_to_tag": {
                                "0": "rating_general",
                                "1": "1girl",
                                "2": "blue_hair",
                            },
                            "tag_to_category": {
                                "rating_general": "rating",
                                "1girl": "general",
                                "blue_hair": "general",
                            },
                        }
                    }
                }
            )

            results = interrogator.interrogate(str(image_path))

            self.assertEqual(results["tags"], ["blue_hair", "1girl"])
            self.assertGreater(results["confidence_scores"]["blue_hair"], 0.9)


if __name__ == "__main__":
    unittest.main()
