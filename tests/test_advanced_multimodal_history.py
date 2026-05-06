import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PyQt6.QtWidgets import QApplication

from core.database import InterrogationDatabase
from core.hashing import hash_image_content
from ui.dialogs_advanced import AdvancedImageInspectionDialog


class AdvancedMultimodalHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _append_turn(
        self,
        db: InterrogationDatabase,
        image_id: int,
        model_id: int,
        session_key: str,
        prompt_text: str,
    ) -> None:
        session_id = db.create_or_get_multimodal_session(
            image_id=image_id,
            model_id=model_id,
            mode="single",
            session_key=session_key,
        )
        db.append_multimodal_turn(
            session_id=session_id,
            prompt_type="describe",
            prompt_text=prompt_text,
            included_tables=[],
            included_transcripts=[],
            sidecar_tags=[],
            response_json={"comment": prompt_text, "warnings": []},
            tags=[],
            reasoning_summary="",
        )

    def test_transcript_history_follows_active_llama_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = tmp / "image.png"
            Image.new("RGB", (8, 8), (20, 40, 60)).save(image_path)

            db = InterrogationDatabase(str(tmp / "interrogations.db"))
            file_hash = hash_image_content(str(image_path))
            image_id = db.register_image(
                str(image_path),
                file_hash,
                8,
                8,
                image_path.stat().st_size,
            )
            session_key = f"single:{file_hash}"
            current_model_id = db.register_model("LlamaCpp/current.gguf", "LlamaCpp", config={})
            other_model_id = db.register_model("LlamaCpp/other.gguf", "LlamaCpp", config={})
            self._append_turn(db, image_id, current_model_id, session_key, "current turn")
            self._append_turn(db, image_id, other_model_id, session_key, "other turn 1")
            self._append_turn(db, image_id, other_model_id, session_key, "other turn 2")

            dialog = AdvancedImageInspectionDialog(
                str(image_path),
                [str(image_path)],
                db,
                llama_config={"llama_model_path": str(tmp / "current.gguf")},
            )
            try:
                self.assertEqual(dialog.mm_transcript.count(), 1)

                dialog.llama_config["llama_model_path"] = str(tmp / "other.gguf")
                dialog._load_multimodal_history()
                self.assertEqual(dialog.mm_transcript.count(), 2)

                dialog.llama_config["llama_model_path"] = str(tmp / "missing.gguf")
                dialog._load_multimodal_history()
                self.assertEqual(dialog.mm_transcript.count(), 0)
            finally:
                dialog.close()
                db.close()


if __name__ == "__main__":
    unittest.main()
