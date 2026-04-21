"""Unit tests for multimodal history persistence in SQLite database."""

import tempfile
import unittest
from pathlib import Path

from core.database import InterrogationDatabase


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


if __name__ == "__main__":
    unittest.main()

