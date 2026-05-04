"""Unit tests for multimodal history persistence in SQLite database."""

import tempfile
import unittest
import importlib.util
from pathlib import Path

from core.database import InterrogationDatabase

MODULE_PATH = Path(__file__).resolve().parents[1] / "interrogators" / "llama_cpp_interrogator.py"
SPEC = importlib.util.spec_from_file_location("llama_cpp_interrogator", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
LlamaCppInterrogator = MODULE.LlamaCppInterrogator


class TestDatabaseMultimodal(unittest.TestCase):
    """Covers session/turn create-read-clear lifecycle."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmpdir.name) / "test_interrogations.db"
        self.db = InterrogationDatabase(str(self.db_path))

        self.image_id = self.db.register_image(
            file_path=str(Path(self.tmpdir.name) / "img.png"),
            file_hash="hash123",
            width=512,
            height=512,
            file_size=1024,
        )
        self.model_id = self.db.register_model(
            model_name="LlamaCpp/test-model.gguf",
            model_type="LlamaCpp",
            config={"ctx_size": 4096},
        )

    def tearDown(self):
        self.db.close()
        self.tmpdir.cleanup()

    def test_session_turn_roundtrip(self):
        session_id = self.db.create_or_get_multimodal_session(
            image_id=self.image_id,
            model_id=self.model_id,
            mode="single",
            session_key="single:hash123",
        )
        self.assertIsInstance(session_id, int)

        turn_id = self.db.append_multimodal_turn(
            session_id=session_id,
            prompt_type="describe",
            prompt_text="Describe this image.",
            included_tables=[{"model_name": "WD", "tags": ["portrait"]}],
            included_transcripts=[{"task": "vqa", "response_summary": "It is indoors."}],
            sidecar_tags=["portrait", "wrong_tag"],
            response_json={
                "tags": ["portrait", "smile"],
                "answer": "A smiling portrait.",
                "ocr_text": "",
                "reasoning_summary": "Face is centered and smiling.",
                "warnings": [],
            },
            tags=["portrait", "smile"],
            reasoning_summary="Face is centered and smiling.",
        )
        self.assertIsInstance(turn_id, int)

        history = self.db.get_multimodal_history(
            session_key="single:hash123",
            mode="single",
            image_hash="hash123",
            model_name="LlamaCpp/test-model.gguf",
        )
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["prompt_type"], "describe")
        self.assertEqual(history[0]["tags"], ["portrait", "smile"])
        self.assertEqual(history[0]["included_transcripts"][0]["task"], "vqa")
        self.assertEqual(history[0]["sidecar_tags"], ["portrait", "wrong_tag"])
        effective_prompt = LlamaCppInterrogator.build_user_prompt_from_turn(history[0])
        self.assertIn("Task: describe", effective_prompt)
        self.assertIn("Prior interrogation tables", effective_prompt)
        self.assertIn("Prior inquiry transcripts", effective_prompt)
        self.assertIn("Current sidecar text-file tags", effective_prompt)
        self.assertIn('"model_name": "WD"', effective_prompt)

        deleted = self.db.clear_multimodal_session(
            session_key="single:hash123",
            mode="single",
            image_hash="hash123",
            model_name="LlamaCpp/test-model.gguf",
        )
        self.assertEqual(deleted, 1)

        history_after = self.db.get_multimodal_history(
            session_key="single:hash123",
            mode="single",
            image_hash="hash123",
            model_name="LlamaCpp/test-model.gguf",
        )
        self.assertEqual(history_after, [])

    def test_exact_cache_entries_are_keyed_by_variant(self):
        self.db.save_interrogation_cache_entry(
            image_id=self.image_id,
            model_id=self.model_id,
            cache_key="cache-key-a",
            cache_metadata={"task": "describe", "prompt": "A"},
            results={
                "tags": ["tag_a"],
                "confidence_scores": None,
                "raw_output": "raw a",
                "multimodal_response": {"comment": "A"},
            },
        )
        self.db.save_interrogation_cache_entry(
            image_id=self.image_id,
            model_id=self.model_id,
            cache_key="cache-key-b",
            cache_metadata={"task": "describe", "prompt": "B"},
            results={
                "tags": ["tag_b"],
                "confidence_scores": None,
                "raw_output": "raw b",
                "multimodal_response": {"comment": "B"},
            },
        )

        cached_a = self.db.get_interrogation_cache_entry(
            "hash123",
            "LlamaCpp/test-model.gguf",
            "cache-key-a",
        )
        cached_b = self.db.get_interrogation_cache_entry(
            "hash123",
            "LlamaCpp/test-model.gguf",
            "cache-key-b",
        )

        self.assertEqual(cached_a["tags"], ["tag_a"])
        self.assertEqual(cached_a["cache_metadata"]["prompt"], "A")
        self.assertEqual(cached_b["tags"], ["tag_b"])
        self.assertEqual(cached_b["cache_metadata"]["prompt"], "B")


if __name__ == "__main__":
    unittest.main()
