"""Unit tests for llama.cpp multimodal interrogator helpers."""

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "interrogators" / "llama_cpp_interrogator.py"
SPEC = importlib.util.spec_from_file_location("llama_cpp_interrogator", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
LlamaCppInterrogator = MODULE.LlamaCppInterrogator


class TestLlamaCppInterrogator(unittest.TestCase):
    """Tests for parsing and prompt helpers that don't require runtime server."""

    def test_parse_valid_json_response(self):
        raw = (
            '{"tags":["cat","indoor"],'
            '"comment":"A cat on a sofa.",'
            '"ocr_text":"",'
            '"reasoning_summary":"Visible whiskers and sofa texture.",'
            '"warnings":[]}'
        )
        parsed = LlamaCppInterrogator._parse_and_validate_json_response(raw, task="describe")
        self.assertEqual(parsed["tags"], ["cat", "indoor"])
        self.assertEqual(parsed["comment"], "A cat on a sofa.")
        self.assertEqual(parsed["answer"], "A cat on a sofa.")

    def test_parse_fenced_json_response(self):
        raw = (
            "```json\n"
            '{"tags":["receipt"],"comment":"Contains totals.","ocr_text":"Subtotal 12.00",'
            '"reasoning_summary":"Readable printed receipt text.","warnings":["partial blur"]}\n'
            "```"
        )
        parsed = LlamaCppInterrogator._parse_and_validate_json_response(raw, task="describe")
        self.assertEqual(parsed["ocr_text"], "Subtotal 12.00")
        self.assertEqual(parsed["warnings"], ["partial blur"])

    def test_parse_invalid_schema_raises(self):
        raw = '{"tags":["ok"],"warnings":[]}'
        with self.assertRaises(ValueError):
            LlamaCppInterrogator._parse_and_validate_json_response(raw, task="describe")

    def test_parse_ocr_example_schema(self):
        raw = (
            '{"OCR":["line one","line two"],'
            '"comment":"This looks like a sign.","warnings":[]}'
        )
        parsed = LlamaCppInterrogator._parse_and_validate_json_response(raw, task="ocr")
        self.assertEqual(parsed["ocr_text"], "line one\nline two")
        self.assertEqual(parsed["tags"], [])

    def test_parse_vqa_example_schema(self):
        raw = (
            '{"VQA":["night"],'
            '"comment":"It appears to be night.","warnings":[]}'
        )
        parsed = LlamaCppInterrogator._parse_and_validate_json_response(raw, task="vqa")
        self.assertEqual(parsed["tags"], ["night"])
        self.assertEqual(parsed["ocr_text"], "")

    def test_parse_custom_example_schema(self):
        raw = (
            '{"custom":["person","outdoor"],'
            '"comment":"Responding to a custom prompt.","warnings":[]}'
        )
        parsed = LlamaCppInterrogator._parse_and_validate_json_response(raw, task="custom")
        self.assertEqual(parsed["tags"], ["person", "outdoor"])
        self.assertEqual(parsed["ocr_text"], "")

    def test_parse_legacy_answer_schema_is_still_accepted(self):
        raw = '{"tags":["portrait"],"answer":"Legacy shape.","warnings":[]}'
        parsed = LlamaCppInterrogator._parse_and_validate_json_response(raw, task="describe")
        self.assertEqual(parsed["comment"], "Legacy shape.")
        self.assertEqual(parsed["answer"], "Legacy shape.")

    def test_build_non_json_fallback_response(self):
        parsed = LlamaCppInterrogator._build_non_json_fallback_response(
            "Plain text comment without JSON",
            warnings=["model_returned_non_json_response"],
        )
        self.assertEqual(parsed["tags"], [])
        self.assertIn("Plain text comment", parsed["comment"])
        self.assertIn("Plain text comment", parsed["answer"])
        self.assertEqual(parsed["ocr_text"], "")
        self.assertTrue(parsed["warnings"])

    def test_extract_assistant_content_accepts_choice_text_shape(self):
        response = {
            "choices": [
                {
                    "text": (
                        '{"tags":["dog"],"answer":"A dog.","ocr_text":"",'
                        '"reasoning_summary":"Clear silhouette.","warnings":[]}'
                    )
                }
            ]
        }
        content = LlamaCppInterrogator._extract_assistant_content(response)
        self.assertIn('"tags":["dog"]', content)

    def test_extract_assistant_content_prefers_tool_call_arguments(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "submit_multimodal_response",
                                    "arguments": (
                                        '{"tags":["city"],"answer":"Urban scene.","ocr_text":"",'
                                        '"reasoning_summary":"Buildings visible.","warnings":[]}'
                                    ),
                                },
                            }
                        ],
                        "content": "Ignored prose content",
                    }
                }
            ]
        }
        content = LlamaCppInterrogator._extract_assistant_content(response)
        self.assertIn('"tags":["city"]', content)

    def test_extract_assistant_content_empty_raises_informative_error(self):
        response = {"choices": [{"message": {"role": "assistant", "content": ""}}]}
        with self.assertRaises(ValueError) as ctx:
            LlamaCppInterrogator._extract_assistant_content(response)
        self.assertIn("choice keys", str(ctx.exception))

    def test_build_user_prompt_includes_prior_tables(self):
        prompt = LlamaCppInterrogator._build_user_prompt(
            task="vqa",
            prompt="What brand is on the sign?",
            included_tables=[
                {"model_name": "WD", "tags": ["street", "signage"]},
                {"model_name": "CLIP", "tags": ["night city"]},
            ],
        )
        self.assertIn("Task: vqa", prompt)
        self.assertIn("What brand is on the sign?", prompt)
        self.assertIn("Prior interrogation tables", prompt)
        self.assertIn('"model_name": "WD"', prompt)

    def test_build_user_prompt_task_templates_are_specific(self):
        describe_prompt = LlamaCppInterrogator._build_user_prompt(
            task="describe",
            prompt="",
            included_tables=[],
        )
        ocr_prompt = LlamaCppInterrogator._build_user_prompt(
            task="ocr",
            prompt="",
            included_tables=[],
        )
        vqa_prompt = LlamaCppInterrogator._build_user_prompt(
            task="vqa",
            prompt="What time is shown?",
            included_tables=[],
        )

        self.assertIn("Goal: describe the visible scene and subjects.", describe_prompt)
        self.assertIn("Output key for labels: tags (string[]).", describe_prompt)
        self.assertIn("Goal: extract readable text from the image.", ocr_prompt)
        self.assertIn("Preserve line breaks", ocr_prompt)
        self.assertIn("Goal: answer the user's visual question.", vqa_prompt)
        self.assertIn("Put the direct final answer in comment.", vqa_prompt)

    def test_build_user_prompt_with_format_example(self):
        prompt = LlamaCppInterrogator._build_user_prompt(
            task="ocr",
            prompt="",
            included_tables=[],
            include_format_example=True,
        )
        self.assertIn("Previous response was invalid JSON.", prompt)
        self.assertIn('"OCR": [', prompt)
        self.assertIn('"comment":', prompt)
        self.assertIn("Do not include markdown fences", prompt)

    def test_prompt_display_summary_uses_task_when_user_prompt_empty(self):
        summary = LlamaCppInterrogator.build_prompt_display_summary(
            task="describe",
            prompt="",
            included_tables=[],
        )

        self.assertEqual(summary, "Task: describe")
        self.assertNotIn("[no prompt]", summary)

    def test_prompt_display_summary_includes_context_sources(self):
        summary = LlamaCppInterrogator.build_prompt_display_summary(
            task="describe",
            prompt="",
            included_tables=[
                {"model_name": "WD14", "model_type": "WD", "tags": ["portrait"]},
                {"model_name": "CLIP ViT", "model_type": "CLIP", "tags": ["person"]},
            ],
        )

        self.assertIn("Task: describe", summary)
        self.assertIn("Context sources: 2 prior results", summary)
        self.assertIn("WD14 (WD)", summary)
        self.assertIn("CLIP ViT (CLIP)", summary)

    def test_prompt_display_summary_includes_user_request_and_context(self):
        summary = LlamaCppInterrogator.build_prompt_display_summary(
            task="vqa",
            prompt="What brand is visible?",
            included_tables=[{"model_name": "WD14", "model_type": "WD"}],
        )

        self.assertIn("Task: vqa", summary)
        self.assertIn("User request: What brand is visible?", summary)
        self.assertIn("Context sources: 1 prior result", summary)

    def test_build_user_prompt_from_turn_reconstructs_effective_prompt(self):
        prompt = LlamaCppInterrogator.build_user_prompt_from_turn(
            {
                "prompt_type": "describe",
                "prompt_text": "",
                "included_tables": [{"model_name": "WD14", "tags": ["portrait"]}],
            }
        )

        self.assertIn("Task: describe", prompt)
        self.assertIn("Goal: describe the visible scene and subjects.", prompt)
        self.assertIn("Prior interrogation tables", prompt)
        self.assertIn('"model_name": "WD14"', prompt)

    def test_load_model_uses_runtime_resolved_server_port(self):
        interrogator = LlamaCppInterrogator(model_name="LlamaCpp")
        runtime_mock = mock.Mock()
        runtime_mock.resolve_server_port.return_value = 18081
        runtime_mock.ensure_server.return_value = "http://127.0.0.1:18081"
        interrogator.runtime = runtime_mock

        interrogator.load_model(
            llama_binary_path="C:/tmp/llama-server.exe",
            llama_model_path="C:/tmp/model.gguf",
            llama_mmproj_path=None,
            ctx_size=4096,
            gpu_layers=-1,
            temperature=0.2,
            max_tokens=256,
            server_port=8080,
            server_host="127.0.0.1",
        )

        cfg = interrogator.get_config()
        self.assertEqual(cfg["server_port"], 18081)
        runtime_mock.resolve_server_port.assert_called_once_with(
            host="127.0.0.1",
            requested_port=8080,
        )
        ensure_kwargs = runtime_mock.ensure_server.call_args.kwargs
        self.assertEqual(ensure_kwargs["port"], 18081)


if __name__ == "__main__":
    unittest.main()
