"""The repetition guard: DRY on every request, loops abandoned and retried."""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from PIL import Image

from core.gguf_metadata import recommended_sampling
from core.llama_cpp_runtime import LlamaCppRuntimeManager
from core.pipelines import MultimodalBatchRunner, ReasoningRelay
from core.repetition import (
    DRY_SAMPLING,
    LoopDetector,
    find_loop,
    loop_excerpt,
    rescue_sampling,
)
from interrogators import LlamaCppInterrogator

HANDLE_LOOP = 'There is a handle at top right:\n"@Inaro' + "to" * 200
PADDING = " ".join(f"word{i}" for i in range(80))


class FindLoopTests(unittest.TestCase):
    def test_a_looping_handle_is_caught(self):
        self.assertEqual(find_loop(HANDLE_LOOP), "to")

    def test_meant_repetition_is_left_alone(self):
        for text in (
            PADDING,
            PADDING + " Poster text: " + "BUY NOW! " * 3,
            PADDING + " Hmm" + "." * 30,
            PADDING + ' "tags": ["mountain", "lake", "snow", "sunset", "calm water", "tree line"]',
            "short",
        ):
            with self.subTest(text=text[-40:]):
                self.assertIsNone(find_loop(text))

    def test_sentences_loop_from_the_fourth_repeat(self):
        sentence = "Wait, let me re-check the colors of the sky and the water. "
        self.assertIsNone(find_loop(PADDING + sentence * 3))
        self.assertEqual(find_loop(PADDING + sentence * 4), sentence)

    def test_long_units_loop_from_the_third_repeat(self):
        paragraph = " ".join(f"step{i}" for i in range(25)) + ". "
        self.assertGreaterEqual(len(paragraph), 100)
        self.assertIsNone(find_loop(PADDING + paragraph * 2))
        self.assertEqual(find_loop(PADDING + paragraph * 3), paragraph)

    def test_a_runaway_character_is_caught(self):
        self.assertEqual(find_loop("Hmm" + "." * 250), ".")
        self.assertEqual(find_loop("ok" + "\n" * 250), "\n")

    def test_excerpt_is_one_line(self):
        self.assertEqual(loop_excerpt("to"), "“" + "to" * 16 + "…”")
        self.assertEqual(loop_excerpt("a b\n"), "“a b a b a b a b a b a b a b a b…”")


class LoopDetectorTests(unittest.TestCase):
    def test_fires_soon_after_the_loop_becomes_long_enough(self):
        detector = LoopDetector()
        fired_at = next(i for i, char in enumerate(HANDLE_LOOP) if detector.feed(char))
        loop_start = HANDLE_LOOP.index("to")
        self.assertLess(fired_at - loop_start, 200 + LoopDetector.CHECK_EVERY_CHARS)

    def test_keeps_only_a_bounded_tail(self):
        detector = LoopDetector()
        for i in range(2000):
            detector.feed(f"word{i} ")
        self.assertLessEqual(len(detector._tail), LoopDetector.TAIL_CHARS)


class SamplingTests(unittest.TestCase):
    def test_recommended_sampling_comes_from_the_gguf(self):
        metadata = {
            "general.sampling.temp": 1.0,
            "general.sampling.top_k": 20,
            "general.sampling.top_p": 0.949999988079071,
            "general.sampling.min_p": True,  # not a number: ignored
        }
        self.assertEqual(recommended_sampling(metadata), {"temperature": 1.0, "top_k": 20, "top_p": 0.95})
        self.assertEqual(recommended_sampling({}), {})

    def test_the_retry_is_never_colder_than_configured(self):
        temperature, sampling = rescue_sampling({"temperature": 1.0, "top_k": 20, "top_p": 0.95}, 0.0)
        self.assertEqual((temperature, sampling["top_k"], sampling["top_p"]), (1.0, 20, 0.95))
        self.assertGreater(sampling["dry_multiplier"], DRY_SAMPLING["dry_multiplier"])
        self.assertEqual(rescue_sampling({}, 0.0)[0], 0.6)
        self.assertEqual(rescue_sampling({}, 0.9)[0], 0.9)


class CacheAndRelayTests(unittest.TestCase):
    def test_unguarded_results_are_cached_apart(self):
        base = {"llama_model_path": "/m.gguf", "temperature": 0.0}
        self.assertEqual(MultimodalBatchRunner.normalize_cache_config(dict(base, repetition_guard=True)), base)
        self.assertEqual(
            MultimodalBatchRunner.normalize_cache_config(dict(base, repetition_guard=False))["repetition_guard"],
            False,
        )

    def test_restart_voids_the_shown_thoughts(self):
        sent = []
        relay = ReasoningRelay(sent.append)
        relay("looping")
        relay.restart("stuck; retrying")
        self.assertEqual(sent[-1], {"offset": 0, "text": "", "reset": True, "notice": "stuck; retrying"})
        self.assertEqual(relay.text, "")
        relay("fresh")
        relay.finish()
        self.assertEqual(sent[-2], {"offset": 0, "text": "fresh"})
        self.assertTrue(sent[-1]["done"], "a restarted relay can finish again")


def _sse(delta: dict) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode("utf-8")


ANSWER = json.dumps({"tags": ["watermark"], "comment": "A handle in the corner."})


class _LoopingServer(BaseHTTPRequestHandler):
    """Streams a thinking loop to greedy requests, an answer to sampled ones.

    `loop_always` makes every request loop.
    """

    loop_always = False

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.server.payloads.append(payload)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            if self.loop_always or payload.get("temperature", 0) == 0:
                self.wfile.write(_sse({"reasoning_content": 'The handle reads "@Inaro'}))
                for _ in range(3000):
                    self.wfile.write(_sse({"reasoning_content": "to"}))
                self.server.loops_finished += 1  # only if the client never hung up
            else:
                self.wfile.write(_sse({"reasoning_content": "A handle in the corner."}))
                self.wfile.write(_sse({"content": ANSWER}))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the guard hung up on the loop

    def log_message(self, *args):
        pass


class RetryOverHttpTests(unittest.TestCase):
    def serve(self, **attrs):
        handler = type("Handler", (_LoopingServer,), attrs)
        server = HTTPServer(("127.0.0.1", 0), handler)
        server.payloads, server.loops_finished = [], 0
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 5)
        self.addCleanup(server.shutdown)
        runtime = LlamaCppRuntimeManager()
        runtime._base_url = f"http://127.0.0.1:{server.server_port}"
        runtime._model_alias = "test-model"
        runtime._is_process_running = lambda: True
        interrogator = LlamaCppInterrogator()
        interrogator.runtime = runtime
        interrogator.is_loaded = True
        interrogator.recommended_sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95}
        self.server = server
        return interrogator

    def image(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "a.png"
        Image.new("RGB", (8, 8)).save(path)
        return str(path)

    def test_a_loop_is_abandoned_and_retried_with_the_models_sampling(self):
        interrogator = self.serve()
        previews, notices, thoughts = [], [], []
        result = interrogator.interrogate(
            self.image(), task="describe",
            on_stream_delta=previews.append, on_reasoning_delta=thoughts.append, on_restart=notices.append,
        )

        self.assertEqual(result["tags"], ["watermark"])
        self.assertTrue(result["multimodal_response"]["_loop_retry"])
        first, second = self.server.payloads
        self.assertEqual(self.server.loops_finished, 0, "the loop was cut off, not streamed to the end")
        for key, value in DRY_SAMPLING.items():
            self.assertEqual(first[key], value)
        self.assertEqual((first["temperature"], second["temperature"]), (0.0, 1.0))
        self.assertEqual((second["top_k"], second["top_p"]), (20, 0.95))
        self.assertEqual(len(notices), 1)
        self.assertIn("“toto", notices[0])
        self.assertIn("", previews, "the answer preview was reset for the retry")

    def test_a_second_loop_fails_the_image_clearly(self):
        interrogator = self.serve(loop_always=True)
        with self.assertRaises(RuntimeError) as caught:
            interrogator.interrogate(self.image(), task="describe")
        self.assertIn("kept repeating", str(caught.exception))
        self.assertIn("thinking", str(caught.exception))
        self.assertEqual(len(self.server.payloads), 2)

    def test_unguarded_requests_carry_no_dry(self):
        interrogator = self.serve()
        interrogator.set_repetition_guard(False)
        interrogator.temperature = 0.7  # the stand-in answers sampled requests
        interrogator.interrogate(self.image(), task="describe", on_stream_delta=lambda _t: None)
        self.assertNotIn("dry_multiplier", self.server.payloads[0])


if __name__ == "__main__":
    unittest.main()
