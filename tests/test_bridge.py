"""Tests for the headless bridge behind the opt-in Electron front end."""

import http.client
import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from bridge.events import EventBus
from bridge.server import BridgeServer, RpcError, StaticRoutes
from core.base_interrogator import BaseInterrogator
from core.database import InterrogationDatabase
from core.file_manager import FileManager
from core.inquiry_settings import InquirySettings
from core.onnx_providers import ONNXProviderSettings
from core.tag_filters import TagFilterSettings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKEN = "test-token-123"


class FakeTagger(BaseInterrogator):
    def __init__(self, tags=None):
        super().__init__("FakeCamie")
        self.tags = tags or ["cat", "watermark"]
        self.is_loaded = True
        self.unloaded = False

    def load_model(self, **kwargs):
        self.is_loaded = True

    def unload_model(self):
        self.is_loaded = False
        self.unloaded = True

    def interrogate(self, image_path, **kwargs):
        return {
            "tags": list(self.tags),
            "confidence_scores": {tag: 0.9 for tag in self.tags},
            "raw_output": "",
        }

    def get_model_type(self):
        return "Camie"

    def get_config(self):
        return {"threshold": 0.35}


THOUGHTS = ("The image ", "shows a ", "red square.")


class FakeLlama:
    """A loaded llama model that thinks out loud, then answers."""

    model_name = "LlamaCpp/fake.gguf"

    def __init__(self):
        self.is_loaded = True
        self.disabled = False
        self.effort = None
        self.guard = True
        self.loop_first = False  # simulate a first attempt abandoned for looping
        self.reasoning_controls = {
            "thinking_toggle": True,
            "effort": {"values": ["low", "medium", "xhigh"], "default": "xhigh", "aliases": {"high": "xhigh"}},
        }

    def get_model_type(self):
        return "LlamaCpp"

    def get_config(self):
        return {"temperature": 0.0, "reasoning_effort": self.effort}

    def set_session_history(self, session_key, turns):
        pass

    def reset_session(self, session_key):
        pass

    def unload_model(self):
        self.is_loaded = False

    def set_disable_reasoning(self, disabled):
        self.disabled = bool(disabled)

    def set_repetition_guard(self, enabled):
        self.guard = bool(enabled)

    def set_reasoning_effort(self, effort):
        from core.reasoning_controls import normalize_effort

        self.effort = normalize_effort(self.reasoning_controls, effort)
        return self.effort

    def interrogate(self, image_path, on_stream_delta=None, on_reasoning_delta=None, on_restart=None, **kwargs):
        if self.loop_first:
            if on_reasoning_delta:
                on_reasoning_delta('"@Inaro' + "to" * 50)
            if on_stream_delta:
                on_stream_delta("")
            if on_restart:
                on_restart("Stuck repeating “totototo…” in the thinking; retrying.")
        for fragment in THOUGHTS:
            if on_reasoning_delta:
                on_reasoning_delta(fragment)
        if on_stream_delta:
            on_stream_delta('{"comment": "A red square"')
        response = {"comment": "A red square", "tags": ["red", "square"], "reasoning_summary": "",
                    "_parse_mode": "primary_json"}
        return {"tags": ["red", "square"], "confidence_scores": None,
                "raw_output": json.dumps(response), "multimodal_response": response}


# ── HTTP transport ────────────────────────────────────────────────────────────


class BridgeServerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "renderer").mkdir()
        (root / "renderer" / "index.html").write_text("<html>ok</html>", encoding="utf-8")
        (root / "secret.txt").write_text("do not serve", encoding="utf-8")
        routes = StaticRoutes()
        routes.add_dir("/app", root / "renderer")

        def dispatch(method, params):
            if method == "echo.value":
                return {"value": params.get("value")}
            raise RpcError("no such thing", code="missing")

        self.bus = EventBus()
        self.server = BridgeServer(dispatch, self.bus, TOKEN, routes)
        self.server.start()
        self.port = self.server.port

    def tearDown(self):
        self.server.stop()
        self._tmp.cleanup()

    def request(self, method, path, body=None, headers=None, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        all_headers = {"Host": host or f"127.0.0.1:{self.port}"}
        all_headers.update(headers or {})
        conn.request(method, path, body=body, headers=all_headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return response, data

    def rpc(self, method, params=None, headers=None):
        base = {"Content-Type": "application/json", "X-II-Request": "1", "X-Bridge-Token": TOKEN}
        base.update(headers or {})
        return self.request("POST", "/rpc", json.dumps({"method": method, "params": params or {}}), base)

    def test_requests_without_the_token_are_refused(self):
        response, _ = self.request("GET", "/app/")
        self.assertEqual(response.status, 401)
        response, _ = self.request("POST", "/rpc", "{}", {"X-II-Request": "1"})
        self.assertEqual(response.status, 401)

    def test_thumbnails_and_images_revalidate_so_edits_show(self):
        from bridge.thumbnails import Thumbnailer, resolve_image_path

        root = Path(self._tmp.name)
        image = root / "photo.png"
        Image.new("RGB", (64, 48), (200, 30, 30)).save(image)
        server = BridgeServer(
            lambda method, params: None, EventBus(), TOKEN, StaticRoutes(),
            thumbnailer=Thumbnailer(cache_dir=root / "cache"), image_reader=resolve_image_path,
        )
        server.start()
        self.addCleanup(server.stop)

        def get(path, etag=None):
            conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
            headers = {"Host": f"127.0.0.1:{server.port}", "X-Bridge-Token": TOKEN}
            if etag:
                headers["If-None-Match"] = etag
            conn.request("GET", path, headers=headers)
            response = conn.getresponse()
            data = response.read()
            conn.close()
            return response, data

        for url in (f"/thumb?size=32&path={image}", f"/image?path={image}"):
            with self.subTest(url=url.split("?")[0]):
                first, body = get(url)
                self.assertEqual(first.status, 200)
                self.assertTrue(body)
                self.assertEqual(first.getheader("Cache-Control"), "private, no-cache")
                etag = first.getheader("ETag")
                self.assertTrue(etag)

                again, body = get(url, etag)
                self.assertEqual((again.status, body), (304, b""))

                stat = image.stat()
                Image.new("RGB", (64, 48), (30, 200, 30)).save(image)
                os.utime(image, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
                edited, body = get(url, etag)
                self.assertEqual(edited.status, 200, "an edited image must not be served from cache")
                self.assertNotEqual(edited.getheader("ETag"), etag)
                self.assertTrue(body)

    def test_foreign_host_header_is_refused(self):
        response, _ = self.request("GET", "/app/", headers={"X-Bridge-Token": TOKEN}, host="evil.example:80")
        self.assertEqual(response.status, 403)

    def test_auth_exchanges_the_token_for_an_httponly_cookie(self):
        response, _ = self.request("GET", "/auth?token=wrong")
        self.assertEqual(response.status, 401)

        response, _ = self.request("GET", f"/auth?token={TOKEN}")
        self.assertEqual(response.status, 302)
        cookie = response.getheader("Set-Cookie")
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)

        response, body = self.request("GET", "/app/", headers={"Cookie": cookie.split(";")[0]})
        self.assertEqual(response.status, 200)
        self.assertEqual(body, b"<html>ok</html>")
        self.assertIn("script-src 'self'", response.getheader("Content-Security-Policy"))

    def test_rpc_needs_the_custom_header_and_a_local_origin(self):
        response, _ = self.rpc("echo.value", {"value": 1}, headers={"X-II-Request": ""})
        self.assertEqual(response.status, 403)
        response, _ = self.rpc("echo.value", {"value": 1}, headers={"Origin": "https://evil.example"})
        self.assertEqual(response.status, 403)
        response, body = self.rpc("echo.value", {"value": 5}, headers={"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(response.status, 200)
        self.assertEqual(json.loads(body), {"ok": True, "result": {"value": 5}})

    def test_rpc_errors_reach_the_ui_with_their_code(self):
        _, body = self.rpc("nope.nothing")
        payload = json.loads(body)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["error"]["code"], "missing")

    def test_static_paths_cannot_escape_their_directory(self):
        response, _ = self.request("GET", "/app/../secret.txt", headers={"X-Bridge-Token": TOKEN})
        self.assertEqual(response.status, 404)

    def test_events_stream_published_messages(self):
        received = queue.Queue()

        def listen():
            conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
            conn.request("GET", "/events", headers={"Host": f"127.0.0.1:{self.port}", "X-Bridge-Token": TOKEN})
            response = conn.getresponse()
            buffer = b""
            while True:
                chunk = response.read1(1024) if hasattr(response, "read1") else response.fp.read1(1024)
                if not chunk:
                    return
                buffer += chunk
                if b"event: image_result_ready" in buffer:
                    received.put(buffer.decode("utf-8"))
                    conn.close()
                    return

        thread = threading.Thread(target=listen, daemon=True)
        thread.start()
        deadline = time.time() + 5
        while self.bus.subscriber_count == 0 and time.time() < deadline:
            time.sleep(0.02)
        self.bus.publish("image_result_ready", {"path": "/x.png", "has_txt": True})
        text = received.get(timeout=5)
        self.assertIn('data: {"path": "/x.png", "has_txt": true}', text)


# ── Service ───────────────────────────────────────────────────────────────────


class BridgeServiceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        # Relative settings and database paths resolve inside the temp dir, so
        # nothing here can touch the working copy's real files.
        self._cwd = os.getcwd()
        os.chdir(self.tmp)
        self.images_dir = self.tmp / "images"
        (self.images_dir / "sub").mkdir(parents=True)
        # Distinct pixels: identical files share a content hash and would be
        # served from the cache.
        for index, name in enumerate(("a.png", "b.png", "sub/c.png")):
            Image.new("RGB", (16, 12), (10 + index * 60, 20, 30)).save(self.images_dir / name)

        from bridge.service import BridgeService
        from bridge.ui_settings import UiSettings

        self.bus = EventBus()
        self.events = self.bus.subscribe()
        self.filters = TagFilterSettings(str(self.tmp / "tag_filters.json"))
        self.service = BridgeService(
            device_status={},
            bus=self.bus,
            database=InterrogationDatabase(str(self.tmp / "interrogations.db")),
            tag_filters=self.filters,
            provider_settings=ONNXProviderSettings(str(self.tmp / "providers.json")),
            inquiry_settings=InquirySettings(str(self.tmp / "inquiry.json")),
            ui_settings=UiSettings(str(self.tmp / "electron_settings.json")),
            telemetry=False,
        )

    def tearDown(self):
        self.service.shutdown()
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def wait_for(self, name, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                item = self.events.get(timeout=0.2)
            except queue.Empty:
                continue
            if item and item[0] == name:
                return json.loads(item[1])
        self.fail(f"event {name} never arrived")

    def open_directory(self, recursive=True):
        self.service.dispatch("dir.open", {"path": str(self.images_dir), "recursive": recursive})
        return self.wait_for("dir.loaded")

    def test_dispatch_rejects_unknown_methods_and_bad_parameters(self):
        with self.assertRaises(RpcError):
            self.service.dispatch("nothing.here", {})
        with self.assertRaises(RpcError):
            self.service.dispatch("__class__.__init__", {})
        with self.assertRaises(RpcError) as ctx:
            self.service.dispatch("dir.open", {"unexpected": 1})
        self.assertEqual(ctx.exception.code, "bad_params")

    def test_directory_scan_publishes_the_image_list(self):
        loaded = self.open_directory()
        self.assertEqual(len(loaded["paths"]), 3)
        self.assertEqual(loaded["dir_count"], 2)
        self.assertEqual(self.service.ui_settings.get("last_directory"), str(self.images_dir.resolve()))

    def test_opening_a_directory_keeps_the_configured_global_database(self):
        configured = self.service.database.db_path
        self.open_directory()
        self.assertEqual(self.service.database.db_path, configured)
        self.assertEqual(Path(self.service.database.get_db_location()), (self.tmp / "interrogations.db").resolve())

    def test_batch_interrogation_streams_states_and_writes_filtered_sidecars(self):
        self.open_directory()
        self.filters.add_remove_tag("watermark")
        tagger = FakeTagger()
        self.service.interrogator = tagger
        self.service.model_info = {"type": "Camie", "label": "FakeCamie", "short": "FakeCamie"}

        self.service.dispatch("interrogate.start", {"txt_mode": "merge"})
        states = []
        finished = None
        deadline = time.time() + 20
        while finished is None and time.time() < deadline:
            item = self.events.get(timeout=5)
            name, payload = item[0], json.loads(item[1])
            if name == "interrogate.item":
                states.append(payload["state"])
            elif name == "interrogate.finished":
                finished = payload

        self.assertEqual(finished["done"], 3)
        self.assertEqual(finished["failed"], 0)
        self.assertTrue(finished["auto_unloaded"], "auto-unload is on by default")
        self.assertTrue(tagger.unloaded)
        self.assertEqual(states.count("running"), 3)
        self.assertEqual(states.count("done"), 3)
        self.assertEqual(FileManager.read_tags_from_file(self.images_dir / "a.png"), ["cat"])

        # A second run is served from the database.
        self.service.interrogator = FakeTagger()
        self.service.model_info = {"type": "Camie", "label": "FakeCamie", "short": "FakeCamie"}
        self.service.dispatch("interrogate.start", {"txt_mode": "none"})
        self.assertEqual(self.wait_for("interrogate.finished")["cached"], 3)

    def test_gallery_listing_detail_and_tag_saving(self):
        self.open_directory()
        FileManager.write_tags_to_file(self.images_dir / "a.png", ["cat", "sky"])
        listing = self.service.dispatch("gallery.list", {"show": "tagged"})
        self.assertEqual([Path(item["p"]).name for item in listing["items"]], ["a.png"])
        self.assertEqual(dict(listing["tags"]), {"cat": 1, "sky": 1})

        filtered = self.service.dispatch("gallery.list", {"tags": ["sky"]})
        self.assertEqual(len(filtered["items"]), 1)

        path = str(self.images_dir / "a.png")
        meta = self.service.dispatch("gallery.meta", {"paths": [path]})[0]
        self.assertEqual((meta["w"], meta["h"], meta["tags"]), (16, 12, 2))

        saved = self.service.dispatch("gallery.save_tags", {"path": path, "tags": ["cat", "new tag"]})
        self.assertEqual(saved["tags"], ["cat", "new tag"])
        refreshed = self.service.dispatch("gallery.meta", {"paths": [path]})[0]
        self.assertEqual(refreshed["tags"], 2)
        detail = self.service.dispatch("gallery.detail", {"path": path})
        self.assertEqual(detail["editor"]["selected"], ["cat", "new tag"])

    def test_inspection_diff_and_apply(self):
        self.open_directory()
        self.service.interrogator = FakeTagger(["cat", "dog"])
        self.service.model_info = {"type": "Camie", "label": "FakeCamie", "short": "FakeCamie"}
        self.service.dispatch("interrogate.start", {"txt_mode": "none"})
        self.wait_for("interrogate.finished")

        path = str(self.images_dir / "a.png")
        FileManager.write_tags_to_file(path_obj := Path(path), ["cat", "hand_made"])
        report = self.service.dispatch("inspect.image", {"path": path})
        model = report["models"][0]
        statuses = {row["tag"]: row["status"] for row in model["comparison"]}
        self.assertEqual(statuses, {"cat": "in_both", "dog": "db_only", "hand_made": "file_only"})
        self.assertTrue(model["plan"]["changes"])

        applied = self.service.dispatch("inspect.apply", {"path": path, "model_name": "FakeCamie"})
        self.assertEqual(applied["tags"], ["cat", "hand_made", "dog"])
        self.assertEqual(FileManager.read_tags_from_file(path_obj), ["cat", "hand_made", "dog"])

    def test_organize_plan_and_move(self):
        self.open_directory()
        FileManager.write_tags_to_file(self.images_dir / "a.png", ["cat"])
        FileManager.write_tags_to_file(self.images_dir / "sub" / "c.png", ["cat", "dog"])
        options = {"tags": ["cat"], "match_mode": "any", "target": "organized", "recursive": True,
                   "move_text": True, "selected_dirs": [".", "sub"]}

        plan = self.service.dispatch("organize.plan", options)
        self.assertEqual((plan["count"], plan["txt_count"], plan["sources_used"]), (2, 2, 2))
        narrowed = self.service.dispatch("organize.plan", {**options, "selected_dirs": ["sub"]})
        self.assertEqual(narrowed["count"], 1)
        with self.assertRaises(RpcError):
            self.service.dispatch("organize.plan", {**options, "target": "../escape"})

        self.service.dispatch("organize.run", options)
        finished = self.wait_for("organize.finished")
        self.assertEqual(finished["moved"], 2)
        self.assertTrue((self.images_dir / "organized" / "a.png").exists())
        self.assertTrue((self.images_dir / "organized" / "a.txt").exists())
        self.assertTrue((self.images_dir / "sub" / "organized" / "c.png").exists())

    def test_filters_are_edited_and_broadcast(self):
        state = self.service.dispatch("filters.add", {"kind": "replace", "tag": "girl", "replacement": "female"})
        self.assertEqual(state["replace"], {"girl": "female"})
        self.assertEqual(self.wait_for("filters.changed")["replace"], {"girl": "female"})
        state = self.service.dispatch("filters.remove", {"kind": "replace", "tag": "girl"})
        self.assertEqual(state["replace"], {})
        with self.assertRaises(RpcError):
            self.service.dispatch("filters.add", {"kind": "bogus", "tag": "x"})

    def test_database_busy_is_a_request_reply_with_the_window(self):
        answers = []
        worker = threading.Thread(
            target=lambda: answers.append(self.service._on_database_busy("save_interrogation", {}, 3))
        )
        worker.start()
        request = self.wait_for("database_busy")
        self.assertEqual(request["operation"], "save_interrogation")
        self.service.dispatch("db.busy_reply", {"id": request["id"], "response": "queue"})
        worker.join(timeout=5)
        self.assertEqual(answers, ["queue"])

    def collect(self, until, timeout=15):
        """Every event up to and including the first `until` event."""
        seen = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                item = self.events.get(timeout=0.2)
            except queue.Empty:
                continue
            if item:
                seen.append((item[0], json.loads(item[1])))
                if item[0] == until:
                    return seen
        self.fail(f"event {until} never arrived")

    def use_fake_llama(self):
        llama = FakeLlama()
        self.service.llama = llama
        self.service.llama_info = {"file": "fake.gguf", "reasoning": llama.reasoning_controls, "reasoning_effort": None}
        return llama

    def test_single_inquiry_streams_thinking_before_the_answer(self):
        self.open_directory()
        self.use_fake_llama()
        image = str(self.images_dir / "a.png")
        self.service.dispatch("inquiry.send", {"path": image, "task": "describe"})
        events = self.collect("inquiry.single.done")

        names = [name for name, _ in events if name.startswith("inquiry.single.")]
        self.assertLess(names.index("inquiry.single.reasoning"), names.index("inquiry.single.stream"))
        thinking = ""
        for name, data in events:
            if name == "inquiry.single.reasoning":
                thinking = thinking[:data["offset"]] + data["text"]
        self.assertEqual(thinking, "".join(THOUGHTS), "every thought arrives before the answer")
        markers = [i for i, (name, data) in enumerate(events) if name == "inquiry.single.reasoning" and data.get("done")]
        first_answer = next(i for i, (name, _) in enumerate(events) if name == "inquiry.single.stream")
        self.assertEqual(len(markers), 1, "one end-of-thinking marker")
        self.assertLess(markers[0], first_answer, "thinking ends when the answer starts")
        self.assertIn("seconds", events[markers[0]][1])
        self.assertEqual(
            [name for name in names if name != "inquiry.single.reasoning"][-2:],
            ["inquiry.single.stream", "inquiry.single.done"],
        )
        done = events[-1][1]
        self.assertEqual(done["turn"]["thinking"], "".join(THOUGHTS))
        self.assertIn("thinking_seconds", done["turn"])
        self.assertEqual(done["turn"]["image_path"], image)

        # Thinking is for the session only: the stored turn has none.
        history = self.service.dispatch("inquiry.image", {"path": image})["history"]
        self.assertEqual([turn["comment"] for turn in history], ["A red square"])
        self.assertNotIn("thinking", history[0])

    def test_batch_inquiry_streams_thinking_for_each_image(self):
        self.open_directory(recursive=False)
        self.use_fake_llama()
        self.service.dispatch("inquiry.batch_start", {"task": "describe", "txt_mode": "none"})
        events = self.collect("inquiry.batch.finished", timeout=20)

        thinking, results = {}, {}
        for name, data in events:
            if name == "inquiry.batch.reasoning":
                thinking[data["path"]] = thinking.get(data["path"], "")[:data["offset"]] + data["text"]
            elif name == "inquiry.batch.result":
                results[data["path"]] = data["turn"]
        expected = {str(self.images_dir / "a.png"), str(self.images_dir / "b.png")}
        self.assertEqual(set(results), expected)
        for path in expected:
            self.assertEqual(thinking[path], "".join(THOUGHTS))
            self.assertEqual(results[path]["thinking"], "".join(THOUGHTS))

    def test_reasoning_effort_is_saved_and_applied_without_a_reload(self):
        llama = self.use_fake_llama()
        result = self.service.dispatch("inquiry.set_reasoning", {"effort": "high"})
        self.assertEqual((result["reasoning_effort"], result["effective_effort"]), ("high", "xhigh"))
        self.assertTrue(result["live"])
        self.assertEqual(llama.effort, "xhigh")
        self.assertEqual(self.service.llama_info["reasoning_effort"], "xhigh")
        self.assertEqual(self.service.dispatch("inquiry.state", {})["config"]["reasoning_effort"], "high")

        result = self.service.dispatch("inquiry.set_reasoning", {"disabled": True})
        self.assertTrue(llama.disabled)
        self.assertTrue(result["disable_reasoning"])
        self.assertEqual(result["reasoning_effort"], "high", "switching thinking off keeps the chosen effort")

        # The PyQt6 tab shares the settings file; it gains the key, loses nothing.
        saved = self.service.inquiry_settings.get_llama_config()
        self.assertEqual((saved["reasoning_effort"], saved["disable_reasoning"]), ("high", True))

    def test_a_restarted_turn_clears_its_thoughts_and_says_why(self):
        self.open_directory()
        self.use_fake_llama().loop_first = True
        self.service.dispatch("inquiry.send", {"path": str(self.images_dir / "a.png"), "task": "describe"})
        events = self.collect("inquiry.single.done")

        thinking, notice = "", None
        for name, data in events:
            if name != "inquiry.single.reasoning":
                continue
            if data.get("reset"):
                thinking, notice = "", data["notice"]
            elif not data.get("done"):
                thinking = thinking[:data["offset"]] + data["text"]
        self.assertIn("Stuck repeating", notice)
        self.assertEqual(thinking, "".join(THOUGHTS), "only the retry's thoughts remain")
        self.assertEqual(events[-1][1]["turn"]["thinking"], "".join(THOUGHTS))

    def test_repetition_guard_is_saved_and_applied_without_a_reload(self):
        llama = self.use_fake_llama()
        self.assertTrue(self.service.dispatch("inquiry.state", {})["config"]["repetition_guard"])
        result = self.service.dispatch("inquiry.set_guard", {"enabled": False})
        self.assertEqual(result, {"repetition_guard": False, "live": True})
        self.assertFalse(llama.guard)
        self.assertIs(self.service.inquiry_settings.get_llama_config()["repetition_guard"], False)

    def test_llama_models_display_by_file_name_not_blob_hash(self):
        class LoadedLlama:
            model_name = "LlamaCpp/af36ecb6b5db1407953345b746c14ac93f0657dda413910b4348683a2d990377"

        self.service.llama = LoadedLlama()
        self.service.llama_info = {"file": "Qwen3.8-27B-UD-Q8_K_XL.gguf"}
        self.assertEqual(self.service._model_display(LoadedLlama.model_name), "Qwen3.8-27B-UD-Q8_K_XL")
        self.assertEqual(self.service._model_display("LlamaCpp/other-model.gguf"), "other-model")
        self.assertEqual(self.service._model_display("SmilingWolf/wd-vit-tagger-v3"), "SmilingWolf/wd-vit-tagger-v3")
        self.service.llama = None
        self.service.llama_info = None

    def test_ui_settings_are_validated(self):
        self.assertEqual(self.service.dispatch("app.set_setting", {"key": "gallery_thumb", "value": 999})["value"], 400)
        with self.assertRaises(RpcError):
            self.service.dispatch("app.set_setting", {"key": "txt_mode", "value": "shred"})
        with self.assertRaises(RpcError):
            self.service.dispatch("app.set_setting", {"key": "not_a_key", "value": 1})


# ── Launch ────────────────────────────────────────────────────────────────────


class LaunchTests(unittest.TestCase):
    def test_missing_electron_stops_with_the_setup_hint(self):
        from bridge import app as bridge_app

        with patch.object(bridge_app, "electron_binary", return_value=None), \
                patch("sys.stderr") as stderr:
            code = bridge_app.start_bridge({}, ["--electron"])
        self.assertEqual(code, 1)
        written = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn("--electron requested but Electron is not installed", written)
        self.assertIn("./setup.sh --electron", written)

    def test_electron_flag_hands_over_to_the_bridge_without_qt(self):
        script = (
            "import sys\n"
            "sys.argv = ['main.py', '--electron']\n"
            "import core.device_detector as dd\n"
            "dd.detect_devices_early = lambda: {'pytorch_cuda_available': False, 'pytorch_error': None,"
            " 'onnx_cuda_available': False, 'onnx_error': None}\n"
            "import bridge\n"
            "seen = {}\n"
            "def fake(status, argv):\n"
            "    seen['argv'] = argv\n"
            "    return 7\n"
            "bridge.start_bridge = fake\n"
            "import main\n"
            "try:\n"
            "    main.main()\n"
            "except SystemExit as exc:\n"
            "    code = exc.code\n"
            "assert code == 7, code\n"
            "assert seen['argv'] == ['--electron'], seen\n"
            "assert not any(m.startswith('PyQt6') for m in sys.modules), 'Qt was imported'\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_token_travels_by_environment_not_argv(self):
        from bridge import electron

        with patch.object(electron.subprocess, "Popen") as popen, \
                patch.object(electron, "sandbox_flags", return_value=[]):
            electron.spawn_electron(Path("/bin/electron"), "http://127.0.0.1:1", "secret-token")
        argv = popen.call_args.args[0]
        env = popen.call_args.kwargs["env"]
        self.assertNotIn("secret-token", " ".join(argv))
        self.assertEqual(env["II_BRIDGE_TOKEN"], "secret-token")
        self.assertEqual(env["II_BRIDGE_URL"], "http://127.0.0.1:1")


if __name__ == "__main__":
    unittest.main()
