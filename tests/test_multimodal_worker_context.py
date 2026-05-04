"""Unit tests for multimodal batch context-source filtering."""

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "ui" / "workers.py"
SPEC = importlib.util.spec_from_file_location("workers", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
MultimodalInterrogationWorker = MODULE.MultimodalInterrogationWorker


class FakeDatabase:
    def get_all_interrogations_for_image(self, file_hash):
        return [
            {
                "model_name": "WD14",
                "model_type": "WD",
                "tags": ["portrait"],
                "confidence_scores": {"portrait": 0.9},
                "raw_output": "wd raw",
                "interrogated_at": "2026-01-01",
            },
            {
                "model_name": "CLIP ViT",
                "model_type": "CLIP",
                "tags": ["person"],
                "confidence_scores": None,
                "raw_output": "clip raw",
                "interrogated_at": "2026-01-02",
            },
            {
                "model_name": "Camie",
                "model_type": "Camie",
                "tags": ["skip"],
                "confidence_scores": None,
                "raw_output": "camie raw",
                "interrogated_at": "2026-01-03",
            },
            {
                "model_name": "LlamaCpp",
                "model_type": "LlamaCpp",
                "tags": ["current"],
                "confidence_scores": None,
                "raw_output": "current raw",
                "interrogated_at": "2026-01-04",
            },
        ]


class FakeInterrogator:
    model_name = "LlamaCpp"


class MultimodalWorkerContextTests(unittest.TestCase):
    def test_prior_tables_disabled_returns_empty_context(self):
        worker = MultimodalInterrogationWorker(
            image_paths=[],
            interrogator=FakeInterrogator(),
            database=FakeDatabase(),
            task="describe",
            prompt="",
            include_prior_tables=False,
            included_model_types=["WD"],
        )

        self.assertEqual(worker._build_included_tables("hash123"), [])

    def test_prior_tables_filter_by_selected_model_types(self):
        worker = MultimodalInterrogationWorker(
            image_paths=[],
            interrogator=FakeInterrogator(),
            database=FakeDatabase(),
            task="describe",
            prompt="",
            include_prior_tables=True,
            included_model_types=["WD", "CLIP"],
        )

        included = worker._build_included_tables("hash123")

        self.assertEqual([row["model_name"] for row in included], ["WD14", "CLIP ViT"])
        self.assertEqual(included[0]["raw_output_summary"], "wd raw")

    def test_prior_tables_filter_by_selected_sources_and_allow_prior_inquiry(self):
        worker = MultimodalInterrogationWorker(
            image_paths=[],
            interrogator=FakeInterrogator(),
            database=FakeDatabase(),
            task="describe",
            prompt="",
            include_prior_tables=True,
            included_model_types=["WD"],
            included_sources=[
                {"model_name": "LlamaCpp", "model_type": "LlamaCpp"},
                {"model_name": "CLIP ViT", "model_type": "CLIP"},
            ],
        )

        included = worker._build_included_tables("hash123")

        self.assertEqual([row["model_name"] for row in included], ["CLIP ViT", "LlamaCpp"])

    def test_empty_selected_sources_include_no_context(self):
        worker = MultimodalInterrogationWorker(
            image_paths=[],
            interrogator=FakeInterrogator(),
            database=FakeDatabase(),
            task="describe",
            prompt="",
            include_prior_tables=True,
            included_sources=[],
        )

        self.assertEqual(worker._build_included_tables("hash123"), [])


if __name__ == "__main__":
    unittest.main()
