"""Unit tests for multimodal batch context-source filtering."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from core.database import InterrogationDatabase
from core.hashing import hash_image_content

MODULE_PATH = Path(__file__).resolve().parents[1] / "ui" / "workers.py"
SPEC = importlib.util.spec_from_file_location("workers", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
MultimodalInterrogationWorker = MODULE.MultimodalInterrogationWorker


class FakeDatabase:
    def __init__(self):
        self.history_calls = []

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

    def get_multimodal_history(self, **kwargs):
        self.history_calls.append(kwargs)
        return [{"prompt_type": "describe", "model_name": kwargs.get("model_name")}]


class FakeInterrogator:
    model_name = "LlamaCpp"


class FakeLlamaInterrogator:
    model_name = "LlamaCpp/test-model.gguf"

    def __init__(self, tag_prefix="fresh", temperature=0.0):
        self.tag_prefix = tag_prefix
        self.temperature = temperature
        self.calls = 0

    def get_model_type(self):
        return "LlamaCpp"

    def get_config(self):
        return {
            "llama_binary_path": "llama-server",
            "llama_model_path": "test-model.gguf",
            "llama_mmproj_path": "mmproj.gguf",
            "ctx_size": 4096,
            "gpu_layers": -1,
            "temperature": self.temperature,
            "max_tokens": 256,
            "server_host": "127.0.0.1",
            "server_port": 8080,
        }

    def interrogate(self, image_path, **kwargs):
        self.calls += 1
        tag = f"{self.tag_prefix}_{self.calls}"
        return {
            "tags": [tag],
            "confidence_scores": None,
            "raw_output": tag,
            "multimodal_response": {
                "tags": [tag],
                "comment": f"Generated {tag}",
                "warnings": [],
                "reasoning_summary": "",
            },
        }


class FakeAuditInterrogator(FakeLlamaInterrogator):
    def interrogate(self, image_path, **kwargs):
        self.calls += 1
        sidecar_tags = kwargs.get("sidecar_tags") or []
        delete_tags = ["dog"] if "dog" in sidecar_tags else []
        remaining = [tag for tag in sidecar_tags if tag not in delete_tags]
        return {
            "tags": remaining,
            "confidence_scores": None,
            "raw_output": "audit raw",
            "multimodal_response": {
                "tags": remaining,
                "delete_tags": delete_tags,
                "comment": "Audit complete.",
                "warnings": [],
                "reasoning_summary": "Dog is unsupported.",
            },
        }


class MultimodalWorkerContextTests(unittest.TestCase):
    def _make_image(self, directory: Path) -> Path:
        image_path = directory / "image.png"
        Image.new("RGB", (8, 8), (20, 40, 60)).save(image_path)
        return image_path

    def _make_named_image(self, directory: Path, name: str) -> Path:
        image_path = directory / name
        Image.new("RGB", (8, 8), (20, 40, 60)).save(image_path)
        return image_path

    def _run_worker(
        self,
        db,
        image_path,
        interrogator,
        task="describe",
        prompt="",
        include_prior_tables=False,
        included_sources=None,
        carry_context_across_batch=False,
        use_cache=True,
        write_files=False,
        overwrite_files=False,
        txt_output_mode=None,
    ):
        worker = MultimodalInterrogationWorker(
            image_paths=[image_path],
            interrogator=interrogator,
            database=db,
            task=task,
            prompt=prompt,
            write_files=write_files,
            overwrite_files=overwrite_files,
            txt_output_mode=txt_output_mode,
            include_prior_tables=include_prior_tables,
            included_sources=included_sources,
            carry_context_across_batch=carry_context_across_batch,
            use_cache=use_cache,
        )
        worker.run()

    def _save_prior_result(self, db, image_path, model_name, model_type, tags):
        image_hash = hash_image_content(str(image_path))
        image_id = db.register_image(str(image_path), image_hash, 8, 8, image_path.stat().st_size)
        model_id = db.register_model(model_name, model_type, config={})
        db.save_interrogation(image_id, model_id, tags, None, ", ".join(tags))

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

    def test_prior_transcripts_are_scoped_to_worker_model(self):
        db = FakeDatabase()
        worker = MultimodalInterrogationWorker(
            image_paths=[],
            interrogator=FakeInterrogator(),
            database=db,
            task="describe",
            prompt="",
            include_prior_transcripts=True,
        )

        history = worker._build_included_transcripts("hash123")

        self.assertEqual(
            db.history_calls,
            [{"image_hash": "hash123", "model_name": "LlamaCpp"}],
        )
        self.assertEqual(history[0]["model_name"], "LlamaCpp")

    def test_cancel_before_run_reports_cancelled(self):
        worker = MultimodalInterrogationWorker(
            image_paths=[],
            interrogator=FakeInterrogator(),
            database=FakeDatabase(),
            task="describe",
            prompt="",
        )
        finished_states = []
        worker.finished.connect(finished_states.append)

        worker.cancel()
        worker.run()

        self.assertTrue(worker.was_cancelled)
        self.assertEqual(finished_states, [True])

    def test_cancel_after_result_reports_cancelled_and_stops_remaining_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            first_image = self._make_named_image(tmp, "first.png")
            second_image = self._make_named_image(tmp, "second.png")
            db = InterrogationDatabase(str(tmp / "interrogations.db"))
            interrogator = FakeLlamaInterrogator("cancel")
            worker = MultimodalInterrogationWorker(
                image_paths=[first_image, second_image],
                interrogator=interrogator,
                database=db,
                task="describe",
                prompt="",
                write_files=False,
                use_cache=False,
            )
            finished_states = []
            worker.result.connect(lambda _path, _results: worker.cancel())
            worker.finished.connect(finished_states.append)

            worker.run()

            self.assertEqual(interrogator.calls, 1)
            self.assertEqual(finished_states, [True])
            self.assertTrue(worker.was_cancelled)

    def test_exact_cache_hit_skips_interrogator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))

            first = FakeLlamaInterrogator("first")
            self._run_worker(db, image_path, first, prompt="same")
            self.assertEqual(first.calls, 1)

            second = FakeLlamaInterrogator("second")
            self._run_worker(db, image_path, second, prompt="same")

            self.assertEqual(second.calls, 0)
            cached = db.get_interrogation(hash_image_content(str(image_path)), second.model_name)
            self.assertEqual(cached["tags"], ["first_1"])

    def test_different_prompt_misses_exact_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))

            self._run_worker(db, image_path, FakeLlamaInterrogator("seed"), prompt="prompt A")
            second = FakeLlamaInterrogator("miss")
            self._run_worker(db, image_path, second, prompt="prompt B")

            self.assertEqual(second.calls, 1)

    def test_different_task_misses_exact_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))

            self._run_worker(db, image_path, FakeLlamaInterrogator("seed"), task="describe")
            second = FakeLlamaInterrogator("miss")
            self._run_worker(db, image_path, second, task="ocr")

            self.assertEqual(second.calls, 1)

    def test_different_context_source_misses_exact_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))
            self._save_prior_result(db, image_path, "WD14", "WD", ["wd_tag"])
            self._save_prior_result(db, image_path, "CLIP ViT", "CLIP", ["clip_tag"])

            self._run_worker(
                db,
                image_path,
                FakeLlamaInterrogator("seed"),
                include_prior_tables=True,
                included_sources=[{"model_name": "WD14", "model_type": "WD"}],
            )
            second = FakeLlamaInterrogator("miss")
            self._run_worker(
                db,
                image_path,
                second,
                include_prior_tables=True,
                included_sources=[{"model_name": "CLIP ViT", "model_type": "CLIP"}],
            )

            self.assertEqual(second.calls, 1)

    def test_different_context_content_misses_exact_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))
            self._save_prior_result(db, image_path, "WD14", "WD", ["wd_tag"])

            source = [{"model_name": "WD14", "model_type": "WD"}]
            self._run_worker(
                db,
                image_path,
                FakeLlamaInterrogator("seed"),
                include_prior_tables=True,
                included_sources=source,
            )
            self._save_prior_result(db, image_path, "WD14", "WD", ["changed_wd_tag"])

            second = FakeLlamaInterrogator("miss")
            self._run_worker(
                db,
                image_path,
                second,
                include_prior_tables=True,
                included_sources=source,
            )

            self.assertEqual(second.calls, 1)

    def test_carry_context_bypasses_exact_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            db = InterrogationDatabase(str(tmp / "interrogations.db"))

            self._run_worker(db, image_path, FakeLlamaInterrogator("seed"), prompt="same")
            second = FakeLlamaInterrogator("live")
            self._run_worker(
                db,
                image_path,
                second,
                prompt="same",
                carry_context_across_batch=True,
                use_cache=True,
            )

            self.assertEqual(second.calls, 1)

    def test_default_txt_output_mode_merges_existing_sidecar_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            image_path.with_suffix(".txt").write_text("existing_tag", encoding="utf-8")
            db = InterrogationDatabase(str(tmp / "interrogations.db"))

            self._run_worker(
                db,
                image_path,
                FakeLlamaInterrogator("new"),
                use_cache=False,
                write_files=True,
                overwrite_files=False,
            )

            self.assertEqual(
                image_path.with_suffix(".txt").read_text(encoding="utf-8"),
                "existing_tag, new_1",
            )

    def test_none_txt_output_mode_leaves_sidecar_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            image_path.with_suffix(".txt").write_text("existing_tag", encoding="utf-8")
            db = InterrogationDatabase(str(tmp / "interrogations.db"))

            self._run_worker(
                db,
                image_path,
                FakeLlamaInterrogator("new"),
                use_cache=False,
                write_files=True,
                overwrite_files=True,
                txt_output_mode="none",
            )

            self.assertEqual(
                image_path.with_suffix(".txt").read_text(encoding="utf-8"),
                "existing_tag",
            )

    def test_overwrite_txt_output_mode_replaces_sidecar_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            image_path.with_suffix(".txt").write_text("existing_tag", encoding="utf-8")
            db = InterrogationDatabase(str(tmp / "interrogations.db"))

            self._run_worker(
                db,
                image_path,
                FakeLlamaInterrogator("new"),
                use_cache=False,
                txt_output_mode="overwrite",
            )

            self.assertEqual(
                image_path.with_suffix(".txt").read_text(encoding="utf-8"),
                "new_1",
            )

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

    def test_audit_deletes_erroneous_sidecar_tags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            image_path = self._make_image(tmp)
            image_path.with_suffix(".txt").write_text("cat, dog, indoor", encoding="utf-8")
            db = InterrogationDatabase(str(tmp / "interrogations.db"))

            self._run_worker(
                db,
                image_path,
                FakeAuditInterrogator("audit"),
                task="audit",
                use_cache=False,
                write_files=True,
            )

            self.assertEqual(image_path.with_suffix(".txt").read_text(encoding="utf-8"), "cat, indoor")
            cached = db.get_interrogation(hash_image_content(str(image_path)), "LlamaCpp/test-model.gguf")
            self.assertEqual(cached["tags"], ["cat", "indoor"])


if __name__ == "__main__":
    unittest.main()
