"""Tests for the reasoning toggles and the streaming stall guard."""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest import mock

from core.llama_cpp_runtime import (
    LlamaCppRuntimeManager,
    LlamaCppStallError,
    is_llama_timeout_error,
)
from interrogators import LlamaCppInterrogator


def _sse(chunk: dict) -> bytes:
    return f"data: {json.dumps(chunk)}\n\n".encode("utf-8")


def _content(text: str) -> bytes:
    return _sse({"choices": [{"delta": {"content": text}}]})


class _ScriptedHandler(BaseHTTPRequestHandler):
    """llama-server stand-in that can pause partway through a stream."""

    chunks: list = []
    pause_after: int = -1
    pause_seconds: float = 0.0
    gap_seconds: float = 0.0

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.server.last_payload = json.loads(self.rfile.read(length) or b"{}")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            for index, chunk in enumerate(self.chunks):
                self.wfile.write(chunk)
                self.wfile.flush()
                if index == self.pause_after:
                    # Silence on an open connection: the wedged-generation case.
                    time.sleep(self.pause_seconds)
                elif self.gap_seconds:
                    time.sleep(self.gap_seconds)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except BrokenPipeError:
            # Expected once the stall guard fires and the client hangs up.
            pass

    def log_message(self, *args):
        pass


class _ServerCase(unittest.TestCase):
    def _serve(self, **attrs) -> LlamaCppRuntimeManager:
        handler = type("Handler", (_ScriptedHandler,), attrs)
        server = HTTPServer(("127.0.0.1", 0), handler)
        server.last_payload = None
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)

        manager = LlamaCppRuntimeManager()
        manager._base_url = f"http://127.0.0.1:{server.server_port}"
        manager._model_alias = "test-model"
        manager._is_process_running = lambda: True
        self.server = server
        return manager


class ChatTemplateKwargsTests(_ServerCase):
    def test_kwargs_are_forwarded_to_the_server(self):
        manager = self._serve(chunks=[_content("hi")])

        manager.chat_completion(
            messages=[{"role": "user", "content": "x"}],
            temperature=0.0,
            max_tokens=32,
            on_delta=lambda _: None,
            chat_template_kwargs={"enable_thinking": False},
        )

        self.assertEqual(
            self.server.last_payload["chat_template_kwargs"],
            {"enable_thinking": False},
        )

    def test_payload_is_unchanged_when_no_kwargs_are_given(self):
        manager = self._serve(chunks=[_content("hi")])

        manager.chat_completion(
            messages=[{"role": "user", "content": "x"}],
            temperature=0.0,
            max_tokens=32,
            on_delta=lambda _: None,
        )

        self.assertNotIn("chat_template_kwargs", self.server.last_payload)


class StallGuardTests(_ServerCase):
    def test_reasoning_tokens_keep_a_healthy_stream_alive(self):
        thinking = _sse({"choices": [{"delta": {"reasoning_content": "thinking"}}]})
        manager = self._serve(chunks=[thinking] * 4 + [_content("done")], gap_seconds=0.2)
        deltas = []
        manager.chat_completion(
            messages=[{"role": "user", "content": "x"}], temperature=0.0,
            max_tokens=32, on_delta=deltas.append, stall_timeout=0.5,
        )
        self.assertEqual(deltas, ["done"])

    def test_silence_mid_stream_raises_a_stall_error(self):
        manager = self._serve(
            chunks=[_content("a"), _content("b")],
            pause_after=0,
            pause_seconds=1.5,
        )

        with self.assertRaises(LlamaCppStallError):
            manager.chat_completion(
                messages=[{"role": "user", "content": "x"}],
                temperature=0.0,
                max_tokens=32,
                on_delta=lambda _: None,
                stall_timeout=0.4,
            )

    def test_steady_tokens_do_not_trip_the_guard(self):
        # The point of a stall budget over a wall clock: this run takes ~0.9s,
        # well past the 0.4s budget, but no single gap between tokens is. Slow
        # hardware looks like this; a wedged server does not.
        manager = self._serve(
            chunks=[_content("a"), _content("b"), _content("c"), _content("d")],
            gap_seconds=0.3,
        )

        deltas = []
        manager.chat_completion(
            messages=[{"role": "user", "content": "x"}],
            temperature=0.0,
            max_tokens=32,
            on_delta=deltas.append,
            stall_timeout=0.4,
        )

        self.assertEqual("".join(deltas), "abcd")

    def test_keepalives_without_tokens_still_trip_the_guard(self):
        # The reason the guard counts tokens rather than trusting the socket
        # timeout: these chunks keep the connection warm indefinitely while the
        # model produces nothing, so a socket-level limit alone never fires.
        idle = _sse({"choices": [{"delta": {}}]})
        manager = self._serve(
            chunks=[_content("a")] + [idle] * 40,
            gap_seconds=0.05,
        )

        with self.assertRaises(LlamaCppStallError) as caught:
            manager.chat_completion(
                messages=[{"role": "user", "content": "x"}],
                temperature=0.0,
                max_tokens=32,
                on_delta=lambda _: None,
                stall_timeout=0.5,
            )

        self.assertIn("no tokens", str(caught.exception))

    def test_a_stall_is_not_classified_as_a_timeout(self):
        # The retry ladder keys off this: a stall must not buy a second,
        # longer run at a wedged generation.
        self.assertFalse(is_llama_timeout_error(LlamaCppStallError("no tokens for 9s")))

    def test_stall_error_survives_a_timeout_shaped_cause(self):
        error = LlamaCppStallError("no tokens")
        error.__cause__ = TimeoutError("timed out")
        self.assertFalse(is_llama_timeout_error(error))


class ReasoningWiringTests(unittest.TestCase):
    def _load(self, **kwargs) -> tuple:
        interrogator = LlamaCppInterrogator(model_name="LlamaCpp")
        runtime_mock = mock.Mock()
        runtime_mock.resolve_server_port.return_value = 8080
        runtime_mock.ensure_server.return_value = "http://127.0.0.1:8080"
        interrogator.runtime = runtime_mock

        interrogator.load_model(
            llama_binary_path="/tmp/llama-server",
            llama_model_path="/tmp/model.gguf",
            llama_mmproj_path=None,
            ctx_size=4096,
            gpu_layers=-1,
            temperature=0.0,
            max_tokens=256,
            server_port=8080,
            server_host="127.0.0.1",
            **kwargs,
        )
        return interrogator, runtime_mock

    def test_disable_reasoning_produces_template_kwargs(self):
        interrogator, _ = self._load(disable_reasoning=True)
        self.assertEqual(
            interrogator._build_chat_template_kwargs(),
            {"enable_thinking": False},
        )

    def test_reasoning_left_on_sends_nothing_extra(self):
        interrogator, _ = self._load(disable_reasoning=False)
        self.assertIsNone(interrogator._build_chat_template_kwargs())

    def test_reasoning_can_be_changed_without_reloading(self):
        # The tooltip promises no reload. Before this setter existed the value
        # was only read by load_model(), so a mid-session toggle did nothing.
        interrogator, runtime_mock = self._load(disable_reasoning=False)
        self.assertIsNone(interrogator._build_chat_template_kwargs())

        interrogator.set_disable_reasoning(True)

        self.assertEqual(
            interrogator._build_chat_template_kwargs(),
            {"enable_thinking": False},
        )
        self.assertTrue(interrogator.get_config()["disable_reasoning"])
        # No reload: the server was started once, at load_model().
        self.assertEqual(runtime_mock.ensure_server.call_count, 1)

    def test_reasoning_can_be_turned_back_on_without_reloading(self):
        interrogator, _ = self._load(disable_reasoning=True)
        interrogator.set_disable_reasoning(False)

        self.assertIsNone(interrogator._build_chat_template_kwargs())
        self.assertFalse(interrogator.get_config()["disable_reasoning"])

    def test_no_reasoning_preserve_reaches_ensure_server(self):
        _, runtime_mock = self._load(no_reasoning_preserve=True)
        kwargs = runtime_mock.ensure_server.call_args.kwargs
        self.assertTrue(kwargs["no_reasoning_preserve"])

    def test_no_reasoning_preserve_defaults_off(self):
        _, runtime_mock = self._load()
        kwargs = runtime_mock.ensure_server.call_args.kwargs
        self.assertFalse(kwargs["no_reasoning_preserve"])

    def test_reasoning_budget_reaches_ensure_server(self):
        interrogator, runtime_mock = self._load(reasoning_budget=512)
        self.assertEqual(runtime_mock.ensure_server.call_args.kwargs["reasoning_budget"], 512)
        self.assertEqual(interrogator.get_config()["reasoning_budget"], 512)

    def test_reasoning_budget_defaults_to_unrestricted(self):
        _, runtime_mock = self._load()
        self.assertEqual(runtime_mock.ensure_server.call_args.kwargs["reasoning_budget"], -1)


class ServerFlagTests(unittest.TestCase):
    """--no-reasoning-preserve must reach the command line and the config key.

    The config key matters as much as the flag: it is what forces a restart
    when the setting changes, and a launch-time flag cannot be applied to an
    already-running server.
    """

    def _launch_argv(self, **kwargs) -> list:
        manager = LlamaCppRuntimeManager()
        captured = {}

        def fake_popen(cmd, *args, **popen_kwargs):
            captured["cmd"] = cmd
            raise RuntimeError("stop here: the command line is all we need")

        with mock.patch("core.llama_cpp_runtime.Path.exists", return_value=True), \
             mock.patch.object(manager, "resolve_server_port", return_value=8080), \
             mock.patch("subprocess.Popen", side_effect=fake_popen):
            try:
                manager.ensure_server(
                    binary_path="/tmp/llama-server",
                    model_path="/tmp/model.gguf",
                    **kwargs,
                )
            except Exception:
                pass
        return captured.get("cmd", [])

    def test_flag_is_passed_when_enabled(self):
        self.assertIn("--no-reasoning-preserve", self._launch_argv(no_reasoning_preserve=True))

    def test_flag_is_absent_by_default(self):
        self.assertNotIn("--no-reasoning-preserve", self._launch_argv())

    def test_reasoning_budget_is_passed_as_a_pair(self):
        argv = self._launch_argv(reasoning_budget=512)
        self.assertIn("--reasoning-budget", argv)
        self.assertEqual(argv[argv.index("--reasoning-budget") + 1], "512")

    def test_zero_budget_is_passed_not_treated_as_unset(self):
        # 0 is meaningful to llama.cpp ("end thinking immediately"), so it must
        # not be swallowed by a falsy check the way -1 is deliberately skipped.
        argv = self._launch_argv(reasoning_budget=0)
        self.assertIn("--reasoning-budget", argv)
        self.assertEqual(argv[argv.index("--reasoning-budget") + 1], "0")

    def test_unrestricted_budget_leaves_the_command_line_untouched(self):
        # -1 must add no flag at all, so upgrading does not change the command
        # for anyone who never sets this.
        self.assertNotIn("--reasoning-budget", self._launch_argv(reasoning_budget=-1))
        self.assertNotIn("--reasoning-budget", self._launch_argv())


if __name__ == "__main__":
    unittest.main()
