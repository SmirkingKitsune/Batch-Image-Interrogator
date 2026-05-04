"""llama.cpp multimodal interrogator implementation."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.base_interrogator import BaseInterrogator
from core.llama_cpp_runtime import (
    LlamaCppRuntimeManager,
    LlamaCppRuntimeError,
    is_llama_timeout_error,
)


class LlamaCppInterrogator(BaseInterrogator):
    """Multimodal interrogator backed by a managed llama.cpp server."""

    TASK_MODES = ["describe", "ocr", "vqa", "custom"]
    REQUIRED_FIELDS = ["comment"]
    RESPONSE_TOOL_NAME = "submit_multimodal_response"
    REQUEST_TIMEOUT_SECONDS = 120.0
    REQUEST_TIMEOUT_RETRY_SECONDS = 300.0

    def __init__(self, model_name: str = "LlamaCpp"):
        super().__init__(model_name)
        self.runtime = LlamaCppRuntimeManager.get_instance()
        self.temperature = 0.0
        self.max_tokens = 4096
        self.server_url: Optional[str] = None
        self._owns_runtime = False
        self._session_history: Dict[str, List[Dict[str, Any]]] = {}

    def load_model(
        self,
        llama_binary_path: str,
        llama_model_path: str,
        llama_mmproj_path: Optional[str] = None,
        ctx_size: int = 4096,
        gpu_layers: int = -1,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        server_port: int = 8080,
        server_host: str = "127.0.0.1",
        **kwargs,
    ):
        """Start or reuse managed llama.cpp server and load multimodal model."""
        model_path = Path(llama_model_path).expanduser().resolve()
        resolved_port = self.runtime.resolve_server_port(
            host=str(server_host),
            requested_port=int(server_port),
        )
        model_label = f"LlamaCpp/{model_path.name}"
        self.model_name = model_label
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens if max_tokens is not None else ctx_size)

        self.config = {
            "llama_binary_path": str(Path(llama_binary_path).expanduser().resolve()),
            "llama_model_path": str(model_path),
            "llama_mmproj_path": (
                str(Path(llama_mmproj_path).expanduser().resolve())
                if llama_mmproj_path
                else None
            ),
            "ctx_size": int(ctx_size),
            "gpu_layers": int(gpu_layers),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "server_port": int(resolved_port),
            "server_host": str(server_host),
            **kwargs,
        }

        try:
            self.server_url = self.runtime.ensure_server(
                binary_path=self.config["llama_binary_path"],
                model_path=self.config["llama_model_path"],
                mmproj_path=self.config["llama_mmproj_path"],
                host=self.config["server_host"],
                port=self.config["server_port"],
                ctx_size=self.config["ctx_size"],
                gpu_layers=self.config["gpu_layers"],
            )
            self._owns_runtime = True
            self.is_loaded = True
        except LlamaCppRuntimeError as exc:
            raise RuntimeError(f"Failed to load llama.cpp model: {exc}") from exc

    def interrogate(
        self,
        image_path: str,
        task: str = "describe",
        prompt: str = "",
        session_key: Optional[str] = None,
        keep_context: bool = False,
        included_tables: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Interrogate image using llama.cpp multimodal model.

        Returns:
            Dict with 'tags', 'confidence_scores', 'raw_output', and parsed response.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if task not in self.TASK_MODES:
            raise ValueError(f"Invalid task '{task}'. Choose from {self.TASK_MODES}")

        system_prompt = self._build_system_prompt()
        user_text = self._build_user_prompt(task, prompt, included_tables or [])
        image_data_url = self._encode_image_as_data_url(image_path)

        if keep_context and session_key:
            history = self._session_history.setdefault(
                session_key, [{"role": "system", "content": system_prompt}]
            )
            messages = list(history)
        else:
            messages = [{"role": "system", "content": system_prompt}]

        user_message = {
            "role": "user",
            "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": image_data_url}},
            ],
        }
        messages.append(user_message)

        try:
            response = self._chat_completion_with_timeout_retry(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                tools=self._build_response_tools(),
                tool_choice={
                    "type": "function",
                    "function": {"name": self.RESPONSE_TOOL_NAME},
                },
            )
        except LlamaCppRuntimeError as exc:
            raise RuntimeError(f"Multimodal inference failed: {exc}") from exc

        parsed = None
        primary_error: Optional[Exception] = None
        primary_content: Optional[str] = None
        retry_content: Optional[str] = None
        fallback_content: Optional[str] = None
        parse_mode = "primary_json"

        try:
            primary_content = self._extract_assistant_content(response)
            parsed = self._parse_and_validate_json_response(primary_content, task=task)
        except Exception as exc:
            primary_error = exc

        retry_error: Optional[Exception] = None
        retry_messages: Optional[List[Dict[str, Any]]] = None
        if parsed is None:
            # Retry with an explicit JSON format example for lightweight models.
            try:
                retry_user_text = self._build_user_prompt(
                    task=task,
                    prompt=prompt,
                    included_tables=included_tables or [],
                    include_format_example=True,
                )
                retry_messages = list(messages[:-1])
                retry_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": retry_user_text},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    }
                )
                retry_response = self._chat_completion_with_timeout_retry(
                    messages=retry_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    tools=self._build_response_tools(),
                    tool_choice={
                        "type": "function",
                        "function": {"name": self.RESPONSE_TOOL_NAME},
                    },
                )
                retry_content = self._extract_assistant_content(retry_response)
                parsed = self._parse_and_validate_json_response(retry_content, task=task)
                parse_mode = "retry_with_format_example_json"
            except Exception as exc:
                retry_error = exc

        fallback_error: Optional[Exception] = None
        if parsed is None:
            # Fallback: retry without response_format for servers/models that ignore or mishandle it.
            try:
                fallback_messages = retry_messages if retry_messages else messages
                fallback_response = self._chat_completion_with_timeout_retry(
                    messages=fallback_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format=None,
                    tools=self._build_response_tools(),
                    tool_choice={
                        "type": "function",
                        "function": {"name": self.RESPONSE_TOOL_NAME},
                    },
                )
                fallback_content = self._extract_assistant_content(fallback_response)
                parsed = self._parse_and_validate_json_response(fallback_content, task=task)
                parse_mode = "fallback_no_response_format_json"
            except Exception as exc:
                fallback_error = exc

        if parsed is None:
            raw_text = (fallback_content or retry_content or primary_content or "").strip()
            if raw_text:
                try:
                    parsed = self._repair_response_to_json(raw_text, task=task)
                    parse_mode = "repair_json"
                except Exception:
                    parsed = self._build_non_json_fallback_response(
                        raw_text=raw_text,
                        warnings=[
                            "model_returned_non_json_response",
                            f"primary_parse_error: {primary_error}",
                            f"retry_parse_error: {retry_error}",
                            f"fallback_parse_error: {fallback_error}",
                        ],
                    )
                    parse_mode = "non_json_fallback"
            else:
                first_keys = list(response.keys()) if isinstance(response, dict) else []
                raise RuntimeError(
                    "Failed to parse llama response. "
                    f"Primary parse error: {primary_error}. "
                    f"Retry parse error: {retry_error}. "
                    f"Fallback parse error: {fallback_error}. "
                    f"First response keys: {first_keys}"
                ) from fallback_error or primary_error

        debug_raw = (fallback_content or retry_content or primary_content or "").strip()
        if debug_raw:
            parsed["_debug_raw_response"] = debug_raw[:20000]
        parsed["_parse_mode"] = parse_mode

        if keep_context and session_key:
            compact_user = {"role": "user", "content": user_text}
            compact_assistant = {
                "role": "assistant",
                "content": json.dumps(parsed, ensure_ascii=False),
            }
            history = self._session_history.setdefault(
                session_key, [{"role": "system", "content": system_prompt}]
            )
            history.append(compact_user)
            history.append(compact_assistant)

        return {
            "tags": parsed["tags"],
            "confidence_scores": None,
            "raw_output": json.dumps(parsed, indent=2, ensure_ascii=False),
            "multimodal_response": parsed,
        }

    def set_session_history(self, session_key: str, turns: List[Dict[str, Any]]) -> None:
        """Prime single-image session context from persisted turn history."""
        history: List[Dict[str, Any]] = [{"role": "system", "content": self._build_system_prompt()}]
        for turn in turns:
            prompt_text = self.build_user_prompt_from_turn(turn)
            response_json = turn.get("response_json")
            if isinstance(response_json, str):
                assistant_content = response_json
            else:
                assistant_content = json.dumps(response_json or {}, ensure_ascii=False)
            history.append({"role": "user", "content": prompt_text})
            history.append({"role": "assistant", "content": assistant_content})
        self._session_history[session_key] = history

    def reset_session(self, session_key: str) -> None:
        """Clear context for a single session."""
        self._session_history.pop(session_key, None)

    def get_model_type(self) -> str:
        """Return model type identifier."""
        return "LlamaCpp"

    def unload_model(self):
        """Unload model/runtime references."""
        self._session_history.clear()
        if self._owns_runtime:
            self.runtime.release_server()
            self._owns_runtime = False
        self.is_loaded = False
        self.server_url = None

    def _chat_completion_with_timeout_retry(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run completion with one timeout-specific retry at a longer timeout."""
        try:
            return self.runtime.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
            )
        except LlamaCppRuntimeError as exc:
            if not is_llama_timeout_error(exc):
                raise

        try:
            return self.runtime.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                timeout=self.REQUEST_TIMEOUT_RETRY_SECONDS,
            )
        except LlamaCppRuntimeError as retry_exc:
            if is_llama_timeout_error(retry_exc):
                raise LlamaCppRuntimeError(
                    "llama-server request failed after timeout retry "
                    f"({int(self.REQUEST_TIMEOUT_SECONDS)}s then "
                    f"{int(self.REQUEST_TIMEOUT_RETRY_SECONDS)}s): {retry_exc}"
                ) from retry_exc
            raise

    @classmethod
    def _build_system_prompt(cls) -> str:
        return (
            "You are a multimodal image analysis assistant. "
            "Always return ONLY valid JSON. "
            "Use the task-specific JSON key as instructed by the user prompt. "
            "Always include: comment (string) and warnings (string[]). "
            "Do not include analysis steps, only final JSON content. "
            "Do not include markdown fences or extra prose."
        )

    @classmethod
    def _build_user_prompt(
        cls,
        task: str,
        prompt: str,
        included_tables: List[Dict[str, Any]],
        include_format_example: bool = False,
    ) -> str:
        task_instructions = {
            "describe": (
                "Goal: describe the visible scene and subjects.\n"
                "- Focus on objects, people, actions, setting, and style.\n"
                "- Keep comment concise (2-5 sentences) and concrete.\n"
                "- tags should be visual concepts only (no meta commentary).\n"
                "- Output key for labels: tags (string[])."
            ),
            "ocr": (
                "Goal: extract readable text from the image.\n"
                "- Prioritize exact text extraction in OCR.\n"
                "- Preserve line breaks when possible; do not invent unreadable text.\n"
                "- Keep comment concise: summarize what the extracted text indicates.\n"
                "- Output key for extracted text lines: OCR (string[])."
            ),
            "vqa": (
                "Goal: answer the user's visual question.\n"
                "- Put the direct final answer in comment.\n"
                "- Use only image evidence; if uncertain, state uncertainty briefly.\n"
                "- Output key for short supporting labels: VQA (string[])."
            ),
            "custom": (
                "Goal: follow the custom user request while returning the required JSON schema.\n"
                "- Keep comment concise and grounded in visible image content.\n"
                "- Output key for custom labels: custom (string[])."
            ),
        }
        base = task_instructions.get(task, task_instructions["describe"])
        user_prompt = prompt.strip() if prompt else ""

        parts = [f"Task: {task}", base]
        if user_prompt:
            parts.append(f"User request: {user_prompt}")
        if included_tables:
            tables_json = json.dumps(included_tables, ensure_ascii=False, indent=2)
            parts.append("Prior interrogation tables (use as context when helpful):")
            parts.append(tables_json)
        parts.append("Task output JSON schema template (placeholders):")
        parts.append(cls._build_format_example_json(task=task))
        if include_format_example:
            parts.append(
                "Previous response was invalid JSON. Follow the schema template above strictly and replace placeholders with real values."
            )
        parts.append(
            "Return the final output by calling the function/tool with JSON arguments only."
        )
        parts.append("Do not include markdown fences or prose outside the JSON/tool arguments.")
        return "\n\n".join(parts)

    @classmethod
    def build_user_prompt_from_turn(cls, turn: Dict[str, Any]) -> str:
        """Reconstruct the effective user prompt from a persisted multimodal turn."""
        return cls._build_user_prompt(
            task=turn.get("prompt_type") or "describe",
            prompt=turn.get("prompt_text") or "",
            included_tables=turn.get("included_tables") or [],
        )

    @classmethod
    def build_prompt_display_summary(
        cls,
        task: str,
        prompt: str,
        included_tables: List[Dict[str, Any]],
    ) -> str:
        """Build a readable transcript summary of the effective request."""
        clean_task = (task or "describe").strip() or "describe"
        clean_prompt = (prompt or "").strip()
        parts = [f"Task: {clean_task}"]
        if clean_prompt:
            parts.append(f"User request: {clean_prompt}")

        table_count = len(included_tables or [])
        if table_count:
            labels: List[str] = []
            seen = set()
            for table in included_tables:
                if not isinstance(table, dict):
                    continue
                model_name = table.get("model_name") or "Unknown"
                model_type = table.get("model_type")
                label = f"{model_name} ({model_type})" if model_type else str(model_name)
                if label not in seen:
                    labels.append(label)
                    seen.add(label)

            source_text = ", ".join(labels[:4]) if labels else "selected prior results"
            if len(labels) > 4:
                source_text = f"{source_text}, +{len(labels) - 4} more"
            plural = "result" if table_count == 1 else "results"
            parts.append(f"Context sources: {table_count} prior {plural} from {source_text}")

        return "\n".join(parts)

    @classmethod
    def _build_format_example_json(cls, task: str = "describe") -> str:
        templates = {
            "describe": (
                "{\n"
                '  "tags": [list],\n'
                '  "comment": "string",\n'
                '  "warnings": []\n'
                "}"
            ),
            "ocr": (
                "{\n"
                '  "OCR": [list],\n'
                '  "comment": "string",\n'
                '  "warnings": []\n'
                "}"
            ),
            "vqa": (
                "{\n"
                '  "VQA": [list],\n'
                '  "comment": "string",\n'
                '  "warnings": []\n'
                "}"
            ),
            "custom": (
                "{\n"
                '  "custom": [list],\n'
                '  "comment": "string",\n'
                '  "warnings": []\n'
                "}"
            ),
        }
        return templates.get(task, templates["describe"])

    @staticmethod
    def _encode_image_as_data_url(image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            raise ValueError(f"Image does not exist: {image_path}")
        suffix = path.suffix.lower().lstrip(".") or "png"
        if suffix == "jpg":
            suffix = "jpeg"
        with path.open("rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:image/{suffix};base64,{encoded}"

    @staticmethod
    def _extract_assistant_content(response: Dict[str, Any]) -> str:
        choices = response.get("choices", [])
        if not choices:
            raise ValueError("No completion choices returned by llama-server")

        first = choices[0]
        candidates: List[str] = []

        message = first.get("message", {})
        if isinstance(message, dict):
            tool_calls = message.get("tool_calls", [])
            if isinstance(tool_calls, list):
                for call in tool_calls:
                    if not isinstance(call, dict):
                        continue
                    function_obj = call.get("function", {})
                    if isinstance(function_obj, dict):
                        args = function_obj.get("arguments")
                        if isinstance(args, str) and args.strip():
                            candidates.append(args)

            candidates.extend(LlamaCppInterrogator._collect_text_candidates(message.get("content")))
            candidates.extend(LlamaCppInterrogator._collect_text_candidates(message.get("reasoning_content")))

        candidates.extend(LlamaCppInterrogator._collect_text_candidates(first.get("text")))

        delta = first.get("delta", {})
        if isinstance(delta, dict):
            candidates.extend(LlamaCppInterrogator._collect_text_candidates(delta.get("content")))

        content = "\n".join(part for part in candidates if isinstance(part, str) and part.strip()).strip()
        if content:
            return content

        choice_keys = list(first.keys()) if isinstance(first, dict) else []
        raise ValueError(f"Assistant response content was empty (choice keys: {choice_keys})")

    @staticmethod
    def _collect_text_candidates(value: Any) -> List[str]:
        """Collect text fragments from common OpenAI-compatible response shapes."""
        parts: List[str] = []
        if value is None:
            return parts
        if isinstance(value, str):
            return [value]
        if isinstance(value, dict):
            for key in ("text", "content", "value"):
                v = value.get(key)
                if isinstance(v, str):
                    parts.append(v)
                elif isinstance(v, list):
                    parts.extend(LlamaCppInterrogator._collect_text_candidates(v))
            return parts
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    parts.extend(LlamaCppInterrogator._collect_text_candidates(item))
        return parts

    @classmethod
    def _build_response_tools(cls) -> List[Dict[str, Any]]:
        """OpenAI-compatible tool schema for structured multimodal responses."""
        return [
            {
                "type": "function",
                "function": {
                    "name": cls.RESPONSE_TOOL_NAME,
                    "description": "Submit structured multimodal analysis response.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "OCR": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "VQA": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "custom": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "comment": {"type": "string"},
                            "answer": {"type": "string"},
                            "ocr_text": {"type": "string"},
                            "reasoning_summary": {"type": "string"},
                            "warnings": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": cls.REQUIRED_FIELDS,
                    },
                },
            }
        ]

    @classmethod
    def _parse_and_validate_json_response(cls, content: str, task: Optional[str] = None) -> Dict[str, Any]:
        candidate = cls._extract_json_candidate(content)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model response was not valid JSON: {exc}") from exc

        normalized = cls._normalize_task_response(parsed, task=task)
        return normalized

    @classmethod
    def _normalize_task_response(cls, parsed: Dict[str, Any], task: Optional[str] = None) -> Dict[str, Any]:
        """Accept task-specific lightweight JSON and normalize to app-internal shape."""
        if not isinstance(parsed, dict):
            raise ValueError("JSON response must be an object")

        comment = parsed.get("comment")
        if not isinstance(comment, str):
            legacy_answer = parsed.get("answer")
            if isinstance(legacy_answer, str):
                comment = legacy_answer
            else:
                raise ValueError("'comment' must be a string")

        warnings = parsed.get("warnings", [])
        if warnings is None:
            warnings = []
        if not isinstance(warnings, list) or any(not isinstance(w, str) for w in warnings):
            raise ValueError("'warnings' must be an array of strings when provided")

        tags: List[str] = []
        if isinstance(parsed.get("tags"), list) and all(isinstance(x, str) for x in parsed["tags"]):
            tags = list(parsed["tags"])

        ocr_lines = parsed.get("OCR")
        ocr_text = ""
        if isinstance(ocr_lines, list) and all(isinstance(x, str) for x in ocr_lines):
            ocr_text = "\n".join([line for line in ocr_lines if line.strip()]).strip()

        if not ocr_text and isinstance(parsed.get("ocr_text"), str):
            ocr_text = parsed.get("ocr_text", "")

        if task == "vqa" and not tags:
            vqa = parsed.get("VQA")
            if isinstance(vqa, list) and all(isinstance(x, str) for x in vqa):
                tags = list(vqa)

        if task == "custom" and not tags:
            custom = parsed.get("custom")
            if isinstance(custom, list) and all(isinstance(x, str) for x in custom):
                tags = list(custom)

        if task == "describe" and not tags:
            describe_custom = parsed.get("custom")
            if isinstance(describe_custom, list) and all(isinstance(x, str) for x in describe_custom):
                tags = list(describe_custom)

        reasoning_summary = parsed.get("reasoning_summary", "")
        if not isinstance(reasoning_summary, str):
            reasoning_summary = ""

        normalized = {
            "tags": tags,
            "comment": comment,
            # Backward-compat alias for existing persistence/consumers.
            "answer": comment,
            "ocr_text": ocr_text,
            "reasoning_summary": reasoning_summary,
            "warnings": warnings,
        }
        return normalized

    def _repair_response_to_json(self, raw_text: str, task: Optional[str] = None) -> Dict[str, Any]:
        """
        Ask the model to normalize previously returned text into strict JSON schema.
        This pass does not send image content; it only repairs the format.
        """
        repair_messages = [
            {
                "role": "system",
                "content": (
                    "Convert the user content into valid JSON with keys: "
                    "tags (string[]), comment (string), ocr_text (string), "
                    "reasoning_summary (string), warnings (string[]). "
                    "Return only JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Normalize this prior assistant output into the JSON schema:\n\n"
                    f"{raw_text}"
                ),
            },
        ]
        repair_resp = self._chat_completion_with_timeout_retry(
            messages=repair_messages,
            temperature=0.0,
            max_tokens=max(256, min(1024, self.max_tokens)),
            response_format={"type": "json_object"},
        )
        repair_content = self._extract_assistant_content(repair_resp)
        return self._parse_and_validate_json_response(repair_content, task=task)

    @staticmethod
    def _build_non_json_fallback_response(raw_text: str, warnings: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a safe structured response when model output cannot be parsed as JSON."""
        text = raw_text.strip()
        comment = text[:4000] if text else "No parsable comment returned."
        summary = "Model returned non-JSON content; response preserved as comment text."
        warning_list = [w for w in (warnings or []) if w]
        return {
            "tags": [],
            "comment": comment,
            "answer": comment,
            "ocr_text": "",
            "reasoning_summary": summary,
            "warnings": warning_list or ["model_returned_non_json_response"],
        }

    @staticmethod
    def _extract_json_candidate(content: str) -> str:
        stripped = content.strip()

        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
                body = "\n".join(lines[1:-1]).strip()
                if body:
                    return body

        first = stripped.find("{")
        last = stripped.rfind("}")
        if first != -1 and last != -1 and last > first:
            return stripped[first : last + 1]

        return stripped
