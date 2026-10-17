"""Unit tests for llama.cpp runtime port resolution."""

import importlib.util
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "core" / "llama_cpp_runtime.py"
SPEC = importlib.util.spec_from_file_location("llama_cpp_runtime", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
LlamaCppRuntimeError = MODULE.LlamaCppRuntimeError
LlamaCppRuntimeManager = MODULE.LlamaCppRuntimeManager
LlamaCppRequestAborted = MODULE.LlamaCppRequestAborted


def _can_symlink() -> bool:
    """Creating symlinks on Windows needs Developer Mode or admin rights."""
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "target"
        target.write_bytes(b"")
        try:
            (Path(tmp) / "link").symlink_to(target)
        except (OSError, NotImplementedError):
            return False
    return True


requires_symlinks = unittest.skipUnless(
    _can_symlink(), "creating symlinks needs Windows Developer Mode or admin rights"
)


class TestLlamaCppRuntimePortResolution(unittest.TestCase):
    """Tests for llama.cpp runtime port selection and overlap avoidance."""

    @staticmethod
    def _get_free_local_port() -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
        finally:
            sock.close()

    def setUp(self):
        self.manager = LlamaCppRuntimeManager()

    def test_resolve_server_port_returns_requested_when_available(self):
        preferred = self._get_free_local_port()
        resolved = self.manager.resolve_server_port("127.0.0.1", preferred)
        self.assertEqual(resolved, preferred)

    def test_resolve_server_port_avoids_busy_requested_port(self):
        busy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            busy_sock.bind(("127.0.0.1", 0))
            busy_sock.listen(1)
            busy_port = int(busy_sock.getsockname()[1])

            resolved = self.manager.resolve_server_port("127.0.0.1", busy_port)
            self.assertNotEqual(resolved, busy_port)
            self.assertTrue(self.manager._is_port_available("127.0.0.1", resolved))
        finally:
            busy_sock.close()

    def test_resolve_server_port_reuses_active_managed_runtime_port(self):
        active_port = self._get_free_local_port()
        self.manager._base_url = f"http://127.0.0.1:{active_port}"
        with mock.patch.object(self.manager, "_is_process_running", return_value=True):
            with mock.patch.object(self.manager, "_check_health", return_value=True):
                resolved = self.manager.resolve_server_port("127.0.0.1", requested_port=8080)
        self.assertEqual(resolved, active_port)

    def test_resolve_server_port_rejects_invalid_port(self):
        with self.assertRaises(LlamaCppRuntimeError):
            self.manager.resolve_server_port("127.0.0.1", requested_port=70000)


class TestLlamaCppRuntimeServerCommand(unittest.TestCase):
    """Tests for llama-server launch arguments."""

    def setUp(self):
        self.manager = LlamaCppRuntimeManager()
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        root = Path(temp_dir.name)
        self.binary = root / "llama-server.exe"
        self.model = root / "model.gguf"
        self.binary.touch()
        self.model.touch()

        process = mock.Mock()
        process.poll.return_value = None
        patches = [
            mock.patch.object(self.manager, "resolve_server_port", return_value=18080),
            mock.patch.object(self.manager, "_check_health", return_value=True),
            # Keep the runtime from writing launch logs into the repo cache.
            mock.patch.object(MODULE.Path, "mkdir"),
            mock.patch.object(MODULE.Path, "open", mock.mock_open()),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        popen_patcher = mock.patch.object(MODULE.subprocess, "Popen", return_value=process)
        self.popen = popen_patcher.start()
        self.addCleanup(popen_patcher.stop)
        self.addCleanup(setattr, self.manager, "_process", None)

    def _ensure(self, **kwargs) -> None:
        self.manager.ensure_server(
            binary_path=str(self.binary),
            model_path=str(self.model),
            **kwargs,
        )

    def test_no_repack_disabled_by_default(self):
        self._ensure()
        self.assertNotIn("--no-repack", self.popen.call_args.args[0])

    def test_no_repack_enabled_appends_flag(self):
        self._ensure(no_repack=True)
        self.assertIn("--no-repack", self.popen.call_args.args[0])

    def test_toggling_no_repack_restarts_server(self):
        self._ensure(no_repack=False)
        self._ensure(no_repack=False)
        self.assertEqual(self.popen.call_count, 1)

        self._ensure(no_repack=True)
        self.assertEqual(self.popen.call_count, 2)
        self.assertIn("--no-repack", self.popen.call_args.args[0])


class TestAbortActiveRequests(unittest.TestCase):
    """Force stop must cut off a request even before its first token arrives."""

    def setUp(self):
        release = self.release = threading.Event()
        received = self.received = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path.endswith("/quick"):
                    body = b'{"choices": [{"message": {"content": "ok"}}]}'
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                if self.path.endswith("/delayed-headers"):
                    received.set()
                    release.wait(10)
                    return
                # Prompt processing: headers go out, then silence.
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.flush()
                release.wait(10)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.manager = LlamaCppRuntimeManager()
        self.manager._base_url = f"http://127.0.0.1:{self.server.server_address[1]}"
        patcher = mock.patch.object(self.manager, "_is_process_running", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()

    def _complete(self, **kwargs):
        return self.manager.chat_completion(
            messages=[{"role": "user", "content": "hi"}], temperature=0.0, max_tokens=8,
            timeout=30.0, **kwargs,
        )

    def test_abort_unblocks_a_request_waiting_for_its_first_token(self):
        outcome = {}

        def run():
            started = time.monotonic()
            try:
                self._complete(on_delta=lambda _fragment: None, stall_timeout=30.0)
            except Exception as exc:  # the exception type is what is asserted
                outcome["error"] = exc
            outcome["seconds"] = time.monotonic() - started

        worker = threading.Thread(target=run)
        worker.start()
        deadline = time.monotonic() + 5
        while not self.manager._active_responses and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(self.manager._active_responses, "request never reached the read")

        self.manager.abort_active_requests()
        worker.join(5)

        self.assertFalse(worker.is_alive(), "the blocked read was not released")
        self.assertIsInstance(outcome.get("error"), LlamaCppRequestAborted)
        self.assertLess(outcome["seconds"], 5)
        self.assertEqual(self.manager._active_responses, set())

    def test_abort_unblocks_before_response_headers(self):
        outcome = []
        original = MODULE.request.Request

        def delayed(url, **kwargs):
            return original(url + "/delayed-headers", **kwargs)

        def run():
            try:
                self._complete(on_delta=lambda _: None, stall_timeout=30.0)
            except Exception as exc:
                outcome.append(exc)

        with mock.patch.object(MODULE.request, "Request", side_effect=delayed):
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            try:
                self.assertTrue(self.received.wait(5))
                self.assertFalse(self.manager._active_responses)
                self.manager.abort_active_requests()
                worker.join(2)
                self.assertFalse(worker.is_alive(), "header read survived force stop")
                self.assertIsInstance(outcome[0], LlamaCppRequestAborted)
                self.assertFalse(self.manager._active_sockets)
            finally:
                self.release.set()
                worker.join(5)

    def test_a_request_started_after_an_abort_is_unaffected(self):
        self.manager.abort_active_requests()
        original = MODULE.request.Request

        def quick(url, **kwargs):
            return original(url + "/quick", **kwargs)

        with mock.patch.object(MODULE.request, "Request", side_effect=quick):
            response = self._complete()
        self.assertEqual(response["choices"][0]["message"]["content"], "ok")


if __name__ == "__main__":
    unittest.main()


class TestModelPathNormalization(unittest.TestCase):
    """Model paths must reach llama-server with their symlinks intact.

    llama.cpp locates the remaining parts of a split GGUF by pattern-matching
    the filename it is given. A HuggingFace snapshot entry is a symlink to a
    content-addressed blob, so dereferencing it renames
    `Model-00001-of-00002.gguf` to a sha256 and the load fails with
    "invalid split file name" — pointing at a path the user never chose.
    """

    @requires_symlinks
    def test_symlinked_model_path_is_not_dereferenced(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blob = root / "blobs" / ("aa" * 32)
            blob.parent.mkdir(parents=True)
            blob.write_bytes(b"GGUF")

            snapshot = root / "snapshots" / "rev"
            snapshot.mkdir(parents=True)
            link = snapshot / "Model-BF16-00001-of-00002.gguf"
            link.symlink_to(blob)

            normalized = MODULE._absolute_keep_symlinks(str(link))
            self.assertEqual(normalized, link)
            self.assertTrue(normalized.name.endswith("-00001-of-00002.gguf"))
            self.assertNotEqual(normalized, blob)
            # Still reaches the bytes, so existence checks keep working.
            self.assertTrue(normalized.exists())

    def test_relative_and_user_paths_are_made_absolute(self):
        normalized = MODULE._absolute_keep_symlinks("~/models/../models/x.gguf")
        self.assertTrue(normalized.is_absolute())
        self.assertNotIn("..", normalized.parts)
        self.assertNotIn("~", str(normalized))
