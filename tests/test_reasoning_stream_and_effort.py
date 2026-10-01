"""Reasoning effort read from the chat template, and the streamed thinking."""

import json
import struct
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

from PIL import Image

from core.llama_cpp_runtime import LlamaCppRuntimeManager
from core.pipelines import MultimodalBatchRunner, ReasoningRelay
from core.reasoning_controls import (
    describe_reasoning_controls,
    detect_reasoning_controls,
    normalize_effort,
    reasoning_controls_for_model,
)
from interrogators import LlamaCppInterrogator

# Shaped like the Qwen3.8 template: a default, an alias, a validated list.
VALIDATING_TEMPLATE = """
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set effort_level = reasoning_effort|default('xhigh') %}
    {%- if effort_level == 'high' %}
        {%- set effort_level = 'xhigh' %}
    {%- endif %}
    {%- if effort_level not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('Unsupported reasoning effort ' ~ reasoning_effort) }}
    {%- endif %}
    {%- if effort_level == 'low' %}{{- 'Think briefly.' }}{%- endif %}
{%- endif %}
{%- if add_generation_prompt %}<|im_start|>assistant
{%- if enable_thinking is defined and enable_thinking is false %}<think></think>{%- endif %}
{%- endif %}
"""

# Shaped like gpt-oss: a default, then the value is interpolated as-is.
INTERPOLATING_TEMPLATE = """
{%- if reasoning_effort is not defined %}
    {%- set reasoning_effort = "medium" %}
{%- endif %}
{{- "Reasoning: " + reasoning_effort }}
"""


def write_gguf(path: Path, template: str) -> None:
    entries = [("general.architecture", "qwen35"), ("tokenizer.chat_template", template)]
    with path.open("wb") as handle:
        handle.write(b"GGUF")
        handle.write(struct.pack("<I", 3))
        handle.write(struct.pack("<Q", 0))
        handle.write(struct.pack("<Q", len(entries)))
        for key, value in entries:
            for text, prefix in ((key, None), (value, 8)):
                encoded = text.encode("utf-8")
                if prefix is not None:
                    handle.write(struct.pack("<I", prefix))
                handle.write(struct.pack("<Q", len(encoded)))
                handle.write(encoded)


class TemplateDetectionTests(unittest.TestCase):
    def test_validated_values_default_and_alias_are_read(self):
        controls = detect_reasoning_controls(VALIDATING_TEMPLATE)
        self.assertTrue(controls["thinking_toggle"])
        self.assertEqual(controls["effort"], {
            "values": ["low", "medium", "xhigh"],
            "default": "xhigh",
            "aliases": {"high": "xhigh"},
        })

    def test_only_accepted_values_are_sent(self):
        controls = detect_reasoning_controls(VALIDATING_TEMPLATE)
        self.assertEqual(normalize_effort(controls, "low"), "low")
        self.assertEqual(normalize_effort(controls, "high"), "xhigh")
        self.assertIsNone(normalize_effort(controls, "max"), "the template would raise")
        self.assertIsNone(normalize_effort(controls, ""))
        self.assertIsNone(normalize_effort(None, "low"), "unknown template: send nothing")

    def test_interpolated_effort_falls_back_to_the_usual_values(self):
        controls = detect_reasoning_controls(INTERPOLATING_TEMPLATE)
        self.assertFalse(controls["thinking_toggle"])
        self.assertEqual(controls["effort"]["values"], ["low", "medium", "high"])
        self.assertEqual(controls["effort"]["default"], "medium")

    def test_templates_without_controls(self):
        self.assertEqual(
            detect_reasoning_controls("{% if enable_thinking is false %}x{% endif %}"),
            {"thinking_toggle": True, "effort": None},
        )
        self.assertEqual(detect_reasoning_controls("{{ messages }}"), {"thinking_toggle": False, "effort": None})
        self.assertEqual(describe_reasoning_controls(detect_reasoning_controls("{{ messages }}")), [])

    def test_report_lines(self):
        lines = describe_reasoning_controls(detect_reasoning_controls(VALIDATING_TEMPLATE))
        self.assertEqual(lines[0], "Reasoning effort: low · medium · xhigh (template default xhigh).")
        self.assertIn("enable_thinking", lines[1])

    def test_read_from_the_gguf_chat_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "model.gguf"
            write_gguf(model, VALIDATING_TEMPLATE)
            self.assertEqual(reasoning_controls_for_model(str(model))["effort"]["default"], "xhigh")
            self.assertIsNone(reasoning_controls_for_model(str(Path(tmp) / "missing.gguf")))
            (Path(tmp) / "bad.gguf").write_bytes(b"not a gguf")
            self.assertIsNone(reasoning_controls_for_model(str(Path(tmp) / "bad.gguf")))


class InterrogatorEffortTests(unittest.TestCase):
    def load(self, template=VALIDATING_TEMPLATE, **kwargs):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        model = Path(tmp.name) / "model.gguf"
        if template is not None:
            write_gguf(model, template)
        interrogator = LlamaCppInterrogator(model_name="LlamaCpp")
        runtime = mock.Mock()
        runtime.resolve_server_port.return_value = 8080
        runtime.ensure_server.return_value = "http://127.0.0.1:8080"
        interrogator.runtime = runtime
        interrogator.load_model(
            llama_binary_path="/tmp/llama-server", llama_model_path=str(model),
            ctx_size=4096, max_tokens=256, **kwargs,
        )
        return interrogator

    def test_effort_rides_on_the_request_once_loaded(self):
        interrogator = self.load(reasoning_effort="low")
        self.assertEqual(interrogator._build_chat_template_kwargs(), {"reasoning_effort": "low"})
        self.assertEqual(interrogator.get_config()["reasoning_effort"], "low")

    def test_effort_changes_without_a_reload(self):
        interrogator = self.load()
        self.assertIsNone(interrogator._build_chat_template_kwargs(), "template default: payload unchanged")
        self.assertEqual(interrogator.set_reasoning_effort("high"), "xhigh")
        self.assertEqual(interrogator._build_chat_template_kwargs(), {"reasoning_effort": "xhigh"})
        self.assertEqual(interrogator.runtime.ensure_server.call_count, 1)

    def test_an_unsupported_effort_is_never_sent(self):
        interrogator = self.load(reasoning_effort="max")
        self.assertIsNone(interrogator._build_chat_template_kwargs())
        self.assertIsNone(interrogator.get_config()["reasoning_effort"])

    def test_a_model_without_effort_never_gets_one(self):
        interrogator = self.load(template="{{ messages }}", reasoning_effort="low")
        self.assertIsNone(interrogator._build_chat_template_kwargs())
        unreadable = self.load(template=None, reasoning_effort="low")
        self.assertIsNone(unreadable.reasoning_controls)
        self.assertIsNone(unreadable._build_chat_template_kwargs())

    def test_disabling_thinking_wins_over_effort(self):
        interrogator = self.load(reasoning_effort="low")
        interrogator.set_disable_reasoning(True)
        self.assertEqual(interrogator._build_chat_template_kwargs(), {"enable_thinking": False})
        self.assertIsNone(interrogator.get_config()["reasoning_effort"])
        interrogator.set_disable_reasoning(False)
        self.assertEqual(interrogator._build_chat_template_kwargs(), {"reasoning_effort": "low"})

    def test_thinking_callback_reaches_the_runtime(self):
        interrogator = self.load()
        content = json.dumps({"comment": "a cat", "tags": ["cat"]})
        interrogator.runtime.chat_completion.return_value = {
            "choices": [{"message": {"role": "assistant", "content": content}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "a.png"
            Image.new("RGB", (8, 8)).save(image)
            thoughts = []

            # Guarded (the default): the runtime gets a wrapper that forwards.
            interrogator.interrogate(str(image), on_stream_delta=lambda _t: None, on_reasoning_delta=thoughts.append)
            interrogator.runtime.chat_completion.call_args.kwargs["on_reasoning_delta"]("hmm")
            self.assertEqual(thoughts, ["hmm"])

            # Unguarded: exactly the call the runtime always got.
            interrogator.set_repetition_guard(False)

            def on_thought(fragment):
                pass

            interrogator.interrogate(str(image), on_stream_delta=lambda _t: None, on_reasoning_delta=on_thought)
            self.assertIs(interrogator.runtime.chat_completion.call_args.kwargs["on_reasoning_delta"], on_thought)
            interrogator.interrogate(str(image), on_stream_delta=lambda _t: None)
            kwargs = interrogator.runtime.chat_completion.call_args.kwargs
            self.assertNotIn("on_reasoning_delta", kwargs)
            self.assertNotIn("sampling", kwargs)


def _sse(delta: dict) -> bytes:
    return f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode("utf-8")


class _StreamHandler(BaseHTTPRequestHandler):
    chunks: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.server.last_payload = json.loads(self.rfile.read(length) or b"{}")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for chunk in self.chunks:
            self.wfile.write(chunk)
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


class RuntimeThinkingStreamTests(unittest.TestCase):
    def serve(self, chunks) -> LlamaCppRuntimeManager:
        handler = type("Handler", (_StreamHandler,), {"chunks": chunks})
        server = HTTPServer(("127.0.0.1", 0), handler)
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

    def test_thinking_streams_first_and_stays_out_of_the_answer(self):
        manager = self.serve([
            _sse({"reasoning_content": "Looking at "}),
            _sse({"reasoning_content": "the image."}),
            _sse({"content": '{"comment": "a cat"}'}),
        ])
        order, thoughts, answer = [], [], []
        response = manager.chat_completion(
            messages=[{"role": "user", "content": "x"}], temperature=0.0, max_tokens=32,
            on_delta=lambda f: (answer.append(f), order.append("answer")),
            on_reasoning_delta=lambda f: (thoughts.append(f), order.append("thinking")),
        )
        self.assertEqual("".join(thoughts), "Looking at the image.")
        self.assertEqual(answer, ['{"comment": "a cat"}'])
        self.assertEqual(order, ["thinking", "thinking", "answer"])
        message = response["choices"][0]["message"]
        self.assertEqual(message["content"], '{"comment": "a cat"}')
        self.assertNotIn("reasoning_content", message, "parsing must see the same message as before")

    def test_thinking_alone_turns_streaming_on(self):
        manager = self.serve([_sse({"reasoning": "hmm"}), _sse({"content": "ok"})])
        thoughts = []
        response = manager.chat_completion(
            messages=[{"role": "user", "content": "x"}], temperature=0.0, max_tokens=32,
            on_reasoning_delta=thoughts.append,
        )
        self.assertTrue(self.server.last_payload["stream"])
        self.assertEqual(thoughts, ["hmm"])
        self.assertEqual(response["choices"][0]["message"]["content"], "ok")


class ReasoningRelayTests(unittest.TestCase):
    def test_emits_appended_text_with_its_offset(self):
        sent = []
        relay = ReasoningRelay(sent.append)
        relay("abc")  # first fragment goes out at once
        relay("def")  # held back by the rate limit
        relay("")
        relay.flush()
        relay.flush()  # nothing new: no empty emit
        self.assertEqual(sent, [{"offset": 0, "text": "abc"}, {"offset": 3, "text": "def"}])
        self.assertEqual(relay.text, "abcdef")
        rebuilt = ""
        for payload in sent:
            rebuilt = rebuilt[:payload["offset"]] + payload["text"]
        self.assertEqual(rebuilt, relay.text)

    def test_finish_sends_what_was_held_back_then_the_duration_once(self):
        sent = []
        relay = ReasoningRelay(sent.append)
        relay("abc")
        relay("def")
        relay.finish()
        relay.finish()
        self.assertEqual(sent[1], {"offset": 3, "text": "def"})
        self.assertEqual(sent[2]["done"], True)
        self.assertEqual((sent[2]["offset"], sent[2]["text"]), (6, ""))
        self.assertIn("seconds", sent[2])
        self.assertEqual(len(sent), 3)

    def test_no_thinking_means_no_done_marker(self):
        sent = []
        ReasoningRelay(sent.append).finish()
        self.assertEqual(sent, [])

    def test_measures_the_thinking_span(self):
        relay = ReasoningRelay(lambda _payload: None)
        self.assertEqual(relay.seconds, 0.0)
        relay("a")
        time.sleep(0.05)
        relay("b")
        self.assertGreaterEqual(relay.seconds, 0.04)


class CacheIdentityTests(unittest.TestCase):
    BASE = {"llama_model_path": "/m.gguf", "temperature": 0.0, "max_tokens": 256}

    def test_default_reasoning_keeps_existing_cache_keys(self):
        config = dict(self.BASE, disable_reasoning=False, reasoning_budget=-1, reasoning_effort=None)
        self.assertEqual(MultimodalBatchRunner.normalize_cache_config(config), self.BASE)

    def test_reasoning_settings_that_change_the_answer_are_keyed(self):
        normalized = MultimodalBatchRunner.normalize_cache_config(
            dict(self.BASE, disable_reasoning=True, reasoning_budget=512, reasoning_effort="low"),
        )
        self.assertEqual(normalized["disable_reasoning"], True)
        self.assertEqual(normalized["reasoning_budget"], 512)
        self.assertEqual(normalized["reasoning_effort"], "low")


if __name__ == "__main__":
    unittest.main()
