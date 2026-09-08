"""Tests for streamed llama-server responses."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from core.llama_cpp_runtime import LlamaCppRuntimeManager
from interrogators import LlamaCppInterrogator


def _sse(chunk: dict) -> bytes:
    return f"data: {json.dumps(chunk)}\n\n".encode("utf-8")


class _StreamingHandler(BaseHTTPRequestHandler):
    """Minimal llama-server stand-in that answers with an SSE tool call."""

    chunks: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.last_payload = body

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in self.chunks:
            self.wfile.write(chunk)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *args):
        pass


class StreamedCompletionTests(unittest.TestCase):
    def _serve(self, chunks) -> LlamaCppRuntimeManager:
        handler = type("Handler", (_StreamingHandler,), {"chunks": chunks})
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

    def test_tool_call_arguments_are_reassembled_from_deltas(self):
        chunks = [
            _sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "call_1",
                 "function": {"name": "submit_multimodal_response", "arguments": '{"comment": "'}},
            ]}}]}),
            _sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": 'a cat'}},
            ]}}]}),
            _sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {"arguments": '", "tags": ["cat"]}'}},
            ]}, "finish_reason": "tool_calls"}]}),
        ]
        manager = self._serve(chunks)

        seen = []
        response = manager.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=64,
            on_delta=seen.append,
        )

        self.assertTrue(self.server.last_payload["stream"])
        self.assertEqual(len(seen), 3)
        content = LlamaCppInterrogator._extract_assistant_content(response)
        self.assertEqual(json.loads(content), {"comment": "a cat", "tags": ["cat"]})
        self.assertEqual(response["choices"][0]["finish_reason"], "tool_calls")

    def test_plain_content_deltas_are_reassembled(self):
        chunks = [
            _sse({"choices": [{"delta": {"content": "Hello "}}]}),
            _sse({"choices": [{"delta": {"content": "world"}}, ]}),
        ]
        manager = self._serve(chunks)

        seen = []
        response = manager.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=64,
            on_delta=seen.append,
        )

        self.assertEqual(seen, ["Hello ", "world"])
        self.assertEqual(response["choices"][0]["message"]["content"], "Hello world")

    def test_malformed_chunk_does_not_abort_the_stream(self):
        chunks = [
            _sse({"choices": [{"delta": {"content": "good "}}]}),
            b"data: {not json}\n\n",
            b": keepalive\n\n",
            _sse({"choices": [{"delta": {"content": "tail"}}]}),
        ]
        manager = self._serve(chunks)

        response = manager.chat_completion(
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=64,
            on_delta=lambda _: None,
        )

        self.assertEqual(response["choices"][0]["message"]["content"], "good tail")

    def test_non_streaming_request_omits_the_stream_flag(self):
        manager = self._serve([])
        # The stand-in always answers with SSE, so only inspect the request.
        try:
            manager.chat_completion(
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.0,
                max_tokens=64,
            )
        except Exception:
            pass
        self.assertNotIn("stream", self.server.last_payload)


class StreamPreviewTests(unittest.TestCase):
    def test_partial_comment_is_surfaced_while_streaming(self):
        raw = '{"tags": ["cat"], "comment": "A cat sits on a'
        self.assertEqual(
            LlamaCppInterrogator.extract_stream_preview(raw),
            "A cat sits on a",
        )

    def test_escapes_are_decoded(self):
        raw = '{"comment": "line one\\nline two'
        self.assertEqual(
            LlamaCppInterrogator.extract_stream_preview(raw),
            "line one\nline two",
        )

    def test_nothing_is_shown_before_the_comment_field_arrives(self):
        self.assertEqual(LlamaCppInterrogator.extract_stream_preview('{"tags": ["ca'), "")
        self.assertEqual(LlamaCppInterrogator.extract_stream_preview(""), "")

    def test_answer_is_used_when_comment_is_absent(self):
        self.assertEqual(
            LlamaCppInterrogator.extract_stream_preview('{"answer": "42"}'),
            "42",
        )

    def test_plain_text_responses_pass_through(self):
        self.assertEqual(
            LlamaCppInterrogator.extract_stream_preview("just prose"),
            "just prose",
        )


class StreamFallbackTests(unittest.TestCase):
    class _Runtime:
        def __init__(self, streamed_response, buffered_response):
            self.streamed_response = streamed_response
            self.buffered_response = buffered_response
            self.calls = []

        def chat_completion(self, **kwargs):
            self.calls.append(kwargs)
            if kwargs.get("on_delta") is not None:
                return self.streamed_response
            return self.buffered_response

    def test_empty_stream_falls_back_to_the_buffered_endpoint(self):
        """Not every llama.cpp build emits tool-call deltas over SSE."""
        empty = {"choices": [{"message": {"role": "assistant", "content": ""}}]}
        buffered = {"choices": [{"message": {"role": "assistant", "content": '{"comment": "ok"}'}}]}
        interrogator = LlamaCppInterrogator()
        interrogator.runtime = self._Runtime(empty, buffered)

        response = interrogator._chat_completion_with_timeout_retry(
            messages=[],
            temperature=0.0,
            max_tokens=32,
            on_delta=lambda _: None,
        )

        self.assertEqual(len(interrogator.runtime.calls), 2)
        self.assertIsNone(interrogator.runtime.calls[1].get("on_delta"))
        self.assertEqual(
            LlamaCppInterrogator._extract_assistant_content(response),
            '{"comment": "ok"}',
        )

    def test_streamed_response_with_content_is_kept(self):
        good = {"choices": [{"message": {"role": "assistant", "content": '{"comment": "ok"}'}}]}
        interrogator = LlamaCppInterrogator()
        interrogator.runtime = self._Runtime(good, {})

        interrogator._chat_completion_with_timeout_retry(
            messages=[],
            temperature=0.0,
            max_tokens=32,
            on_delta=lambda _: None,
        )

        self.assertEqual(len(interrogator.runtime.calls), 1)


if __name__ == "__main__":
    unittest.main()
