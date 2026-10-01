"""llama.cpp multimodal interrogator implementation."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.base_interrogator import BaseInterrogator
from core.gguf_metadata import read_gguf_metadata, recommended_sampling
from core.llama_cpp_runtime import (
    LlamaCppRepetitionError,
    LlamaCppRuntimeManager,
    LlamaCppRuntimeError,
    LlamaCppStallError,
    is_llama_timeout_error,
)
from core.reasoning_controls import normalize_effort, reasoning_controls_from_metadata
from core.repetition import DRY_SAMPLING, LoopDetector, loop_excerpt, rescue_sampling


# Formats llama.cpp decodes itself (stb_image). Anything else -- WebP above
# all, which some builds hand to an external ffmpeg that may not be installed --
# is converted with Pillow before it is sent.
LLAMA_NATIVE_IMAGE_FORMATS = frozenset({"JPEG", "MPO", "PNG", "GIF", "BMP"})
# How llama-server words a request whose image it could not decode.
_IMAGE_LOAD_ERRORS = ("failed to load image", "failed to decode image", "unable to load image")


class LlamaCppInterrogator(BaseInterrogator):
    """Multimodal interrogator backed by a managed llama.cpp server."""

    TASK_MODES = ["describe", "ocr", "vqa", "custom", "audit"]
    REQUIRED_FIELDS = ["comment"]
    RESPONSE_TOOL_NAME = "submit_multimodal_response"
    REQUEST_TIMEOUT_SECONDS = 120.0
    REQUEST_TIMEOUT_RETRY_SECONDS = 300.0
    # Generous on purpose. Prompt prefill on a large image runs for tens of
    # seconds before the first token appears, and that is normal on modest
    # hardware; only genuine silence should trip this.
    STREAM_STALL_SECONDS = 180.0

    def __init__(self, model_name: str = "LlamaCpp"):
        super().__init__(model_name)
        self.runtime = LlamaCppRuntimeManager.get_instance()
        self.temperature = 0.0
        self.max_tokens = 4096
        self.disable_reasoning = False
        # Requested effort; only sent when the loaded template accepts it.
        self.reasoning_effort: Optional[str] = None
        self.reasoning_controls: Optional[Dict[str, Any]] = None
        # DRY on every request, and loops in the stream abandoned and retried.
        self.repetition_guard = True
        self.recommended_sampling: Dict[str, Any] = {}
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
        disable_reasoning: bool = False,
        no_reasoning_preserve: bool = False,
        reasoning_budget: int = -1,
        reasoning_effort: Optional[str] = None,
        repetition_guard: bool = True,
        **kwargs,
    ):
        """Start or reuse managed llama.cpp server and load multimodal model.

        Args:
            disable_reasoning: Ask the chat template to skip thinking, per
                request. Models that emit reasoning spend most of their token
                budget on it and the app keeps only the final JSON, so turning
                it off is what makes a modest `max_tokens` viable.
            no_reasoning_preserve: Launch the server with
                `--no-reasoning-preserve`. Changing it restarts the server.
            reasoning_budget: Cap thinking at N tokens rather than removing it.
                -1 is unrestricted. Quality and speed depend on the model and
                task. This launch flag requires reloading the model to apply.
            reasoning_effort: Per-request `reasoning_effort` for templates that
                read one (gpt-oss, Qwen3.8). Sent only when the model's chat
                template accepts the value; see `reasoning_controls`.
            repetition_guard: Send the DRY anti-repetition sampler with every
                request, and abandon a reply that streams a loop, retrying it
                once with the model's recommended sampling. Per request; see
                `set_repetition_guard`.
        """
        model_path = Path(llama_model_path).expanduser().resolve()
        resolved_port = self.runtime.resolve_server_port(
            host=str(server_host),
            requested_port=int(server_port),
        )
        model_label = f"LlamaCpp/{model_path.name}"
        self.model_name = model_label
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens if max_tokens is not None else ctx_size)
        self.disable_reasoning = bool(disable_reasoning)
        self.reasoning_effort = str(reasoning_effort).strip() if reasoning_effort else None
        self.reasoning_controls = None
        self.repetition_guard = bool(repetition_guard)
        self.recommended_sampling = {}

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
            "disable_reasoning": self.disable_reasoning,
            "no_reasoning_preserve": bool(no_reasoning_preserve),
            "reasoning_budget": int(reasoning_budget),
            "reasoning_effort": None,
            "repetition_guard": self.repetition_guard,
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
                no_reasoning_preserve=self.config["no_reasoning_preserve"],
                reasoning_budget=self.config["reasoning_budget"],
            )
            self._owns_runtime = True
            self.is_loaded = True
        except LlamaCppRuntimeError as exc:
            raise RuntimeError(f"Failed to load llama.cpp model: {exc}") from exc
        # Read once per load (a fraction of a second): the chat template's
        # reasoning controls and the publisher's recommended sampling. A file
        # it cannot read leaves effort unsent and the retry on defaults.
        metadata = self._read_model_metadata(self.config["llama_model_path"])
        self.reasoning_controls = reasoning_controls_from_metadata(metadata)
        self.recommended_sampling = recommended_sampling(metadata)
        self.config["reasoning_effort"] = self._effective_reasoning_effort()

    @staticmethod
    def _read_model_metadata(model_path: str) -> Dict[str, Any]:
        try:
            return read_gguf_metadata(model_path)
        except Exception:  # noqa: BLE001 - a missing or foreign file is "no metadata"
            return {}

    def interrogate(
        self,
        image_path: str,
        task: str = "describe",
        prompt: str = "",
        session_key: Optional[str] = None,
        keep_context: bool = False,
        included_tables: Optional[List[Dict[str, Any]]] = None,
        included_transcripts: Optional[List[Dict[str, Any]]] = None,
        sidecar_tags: Optional[List[str]] = None,
        on_stream_delta: Optional[Callable[[str], None]] = None,
        on_reasoning_delta: Optional[Callable[[str], None]] = None,
        on_restart: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Interrogate image using llama.cpp multimodal model.

        Args:
            on_stream_delta: Called with the accumulated raw response text as it
                streams back. Only the first attempt streams; a reparse retry
                calls it with an empty string to reset any partial display.
            on_reasoning_delta: Called with each fragment of the model's
                thinking as it streams, ahead of the answer. Only the first
                attempt streams it. Thinking is never part of the result.
            on_restart: Called with a short notice when the first attempt is
                abandoned for looping and starts over; whatever streamed so far
                is void. The answer stream is also reset with "".

        Returns:
            Dict with 'tags', 'confidence_scores', 'raw_output', and parsed response.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if task not in self.TASK_MODES:
            raise ValueError(f"Invalid task '{task}'. Choose from {self.TASK_MODES}")

        system_prompt = self._build_system_prompt()
        user_text = self._build_user_prompt(
            task,
            prompt,
            included_tables or [],
            included_transcripts=included_transcripts or [],
            sidecar_tags=sidecar_tags or [],
        )
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

        on_delta = self._build_stream_relay(on_stream_delta)
        first_request = dict(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            on_delta=on_delta,
            on_reasoning_delta=on_reasoning_delta,
        )

        image_resent = loop_retried = False
        try:
            while True:
                try:
                    response = self._chat_completion_with_timeout_retry(**first_request)
                    break
                except LlamaCppRepetitionError as loop:
                    if loop_retried:
                        raise RuntimeError(
                            f"Stopped: the model kept repeating {loop_excerpt(loop.unit)} in its "
                            f"{loop.stream}, even after a retry with its recommended sampling."
                        ) from loop
                    # Greedy decoding repeats a loop exactly, so start over with
                    # the sampling the model was tuned for and a firmer DRY.
                    loop_retried = True
                    first_request["temperature"], first_request["sampling"] = rescue_sampling(
                        self.recommended_sampling, self.temperature,
                    )
                    first_request["on_delta"] = self._build_stream_relay(on_stream_delta)
                    if on_stream_delta is not None:
                        on_stream_delta("")
                    if on_restart is not None:
                        on_restart(
                            f"Stuck repeating {loop_excerpt(loop.unit)} in the {loop.stream}; "
                            "retrying with the model's recommended sampling."
                        )
                except LlamaCppRuntimeError as exc:
                    if image_resent or not self._is_image_load_error(exc):
                        raise
                    # llama.cpp's decoder refused bytes Pillow can read (an
                    # unusual JPEG or BMP, say). Send a re-encoded copy, once;
                    # the later attempts below build on `messages`.
                    image_resent = True
                    image_data_url = self._encode_image_as_data_url(image_path, reencode=True)
                    messages = messages[:-1] + [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": image_data_url}},
                        ],
                    }]
                    first_request["messages"] = messages
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
            if on_stream_delta is not None:
                # The streamed text is being discarded; clear the live view so
                # it does not sit there looking like a finished answer.
                on_stream_delta("")
            # Retry with an explicit JSON format example for lightweight models.
            try:
                retry_user_text = self._build_user_prompt(
                    task=task,
                    prompt=prompt,
                    included_tables=included_tables or [],
                    included_transcripts=included_transcripts or [],
                    sidecar_tags=sidecar_tags or [],
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
                # After a loop, the later attempts keep the rescue sampling.
                retry_response = self._chat_completion_with_timeout_retry(
                    messages=retry_messages,
                    temperature=first_request["temperature"],
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    sampling=first_request.get("sampling"),
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
                    temperature=first_request["temperature"],
                    max_tokens=self.max_tokens,
                    response_format=None,
                    sampling=first_request.get("sampling"),
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

        # Only keep the raw text when the parse needed repair, retry, or a
        # fallback. A clean primary parse is already faithfully represented by
        # `parsed`, and the transcript view only surfaces raw text for unusual
        # parses -- persisting it for every turn dominated the database size.
        debug_raw = (fallback_content or retry_content or primary_content or "").strip()
        if debug_raw and parse_mode != "primary_json":
            parsed["_debug_raw_response"] = debug_raw[:20000]
        parsed["_parse_mode"] = parse_mode
        if loop_retried:
            parsed["_loop_retry"] = True
        if task == "audit":
            delete_tags = self._normalize_tag_list(parsed.get("delete_tags", []))
            parsed["delete_tags"] = delete_tags
            parsed["sidecar_tags"] = list(sidecar_tags or [])
            if sidecar_tags and not parsed.get("tags"):
                delete_lookup = {tag.strip().casefold() for tag in delete_tags}
                parsed["tags"] = [
                    tag for tag in sidecar_tags
                    if tag.strip().casefold() not in delete_lookup
                ]

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

    def set_disable_reasoning(self, disable_reasoning: bool) -> None:
        """Change the reasoning setting on an already-loaded model.

        `enable_thinking` rides on each request rather than the server command
        line, so this needs no reload — unlike `no_reasoning_preserve`, which is
        a launch flag and does restart the server. Takes effect on the next
        request, which mid-batch means the next image rather than the one
        currently generating.
        """
        self.disable_reasoning = bool(disable_reasoning)
        if isinstance(getattr(self, "config", None), dict):
            self.config["disable_reasoning"] = self.disable_reasoning
            self.config["reasoning_effort"] = self._effective_reasoning_effort()

    def set_repetition_guard(self, enabled: bool) -> None:
        """Turn the repetition guard on or off for the next request; no reload."""
        self.repetition_guard = bool(enabled)
        if isinstance(getattr(self, "config", None), dict):
            self.config["repetition_guard"] = self.repetition_guard

    def set_reasoning_effort(self, reasoning_effort: Optional[str]) -> Optional[str]:
        """Change the requested effort on a loaded model; no reload needed.

        Like `enable_thinking`, `reasoning_effort` rides on each request. Returns
        the value that will actually be sent, which is None when this model's
        template does not accept it (or reasoning is disabled).
        """
        self.reasoning_effort = str(reasoning_effort).strip() if reasoning_effort else None
        effective = self._effective_reasoning_effort()
        if isinstance(getattr(self, "config", None), dict):
            self.config["reasoning_effort"] = effective
        return effective

    def _effective_reasoning_effort(self) -> Optional[str]:
        if self.disable_reasoning:
            return None
        return normalize_effort(self.reasoning_controls, self.reasoning_effort)

    def _build_chat_template_kwargs(self) -> Optional[Dict[str, Any]]:
        """Template variables for the current reasoning setting.

        `enable_thinking` is the Qwen3-family switch and is the one llama.cpp
        forwards verbatim to the jinja template. Templates that do not declare
        it ignore the variable, so sending it is safe across models; returning
        None when reasoning is left on keeps the payload byte-identical to
        before this option existed.

        `reasoning_effort` is different: a template that validates it (Qwen3.8)
        fails the request on an unknown value, so it is only sent when the
        loaded template lists it.
        """
        if self.disable_reasoning:
            return {"enable_thinking": False}
        effort = self._effective_reasoning_effort()
        if effort:
            return {"reasoning_effort": effort}
        return None

    def _chat_completion_with_timeout_retry(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        on_delta: Optional[Callable[[str], None]] = None,
        on_reasoning_delta: Optional[Callable[[str], None]] = None,
        sampling: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Retry rejected output constraints once using prompt-only JSON.

        Keep this separate from timeout retries: unrelated HTTP errors and
        stalled generations must still propagate without another attempt.

        Every request comes through here, so this is where the repetition
        guard applies: DRY rides along unless `sampling` overrides it, and the
        reply streams through loop detectors even when nobody is watching, so
        a loop raises `LlamaCppRepetitionError` instead of running to
        max_tokens.
        """
        if self.repetition_guard:
            if sampling is None:
                sampling = dict(DRY_SAMPLING)
            on_delta, on_reasoning_delta = self._guard_streams(on_delta, on_reasoning_delta)
        try:
            return self._chat_completion_with_transport_retry(
                messages, temperature, max_tokens, response_format,
                tools, tool_choice, on_delta, on_reasoning_delta, sampling,
            )
        except LlamaCppRuntimeError as exc:
            detail = str(exc).lower()
            if not (
                (response_format or tools or tool_choice)
                and "http error 400:" in detail
                and any(marker in detail for marker in (
                    "grammar", "response_format", "tool_choice",
                ))
            ):
                raise
        return self._chat_completion_with_transport_retry(
            messages, temperature, max_tokens, on_delta=on_delta,
            on_reasoning_delta=on_reasoning_delta, sampling=sampling,
        )

    @staticmethod
    def _guard_streams(
        on_delta: Optional[Callable[[str], None]],
        on_reasoning_delta: Optional[Callable[[str], None]],
    ) -> Tuple[Callable[[str], None], Callable[[str], None]]:
        """Stream callbacks that raise once the answer or the thinking loops.

        Both are always returned, so every guarded request streams and can be
        abandoned; raising inside the stream closes the connection, which
        llama-server takes as the cue to stop generating.
        """
        detectors = {"answer": LoopDetector(), "thinking": LoopDetector()}

        def watch(stream: str, forward: Optional[Callable[[str], None]]) -> Callable[[str], None]:
            def guarded(fragment: str) -> None:
                if forward is not None:
                    forward(fragment)
                unit = detectors[stream].feed(fragment)
                if unit is not None:
                    raise LlamaCppRepetitionError(unit, stream)
            return guarded

        return watch("answer", on_delta), watch("thinking", on_reasoning_delta)

    def _chat_completion_with_transport_retry(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        on_delta: Optional[Callable[[str], None]] = None,
        on_reasoning_delta: Optional[Callable[[str], None]] = None,
        sampling: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run completion with one timeout-specific retry at a longer timeout.

        A stall is not retried. `LlamaCppStallError` is outside the
        `is_llama_timeout_error` family precisely so it propagates on the first
        occurrence instead of buying a wedged generation a second, longer run.
        """
        chat_template_kwargs = self._build_chat_template_kwargs()
        # Only passed when wanted, so callers and doubles that predate the
        # thinking stream see the exact call they always did.
        thinking = {"on_reasoning_delta": on_reasoning_delta} if on_reasoning_delta is not None else {}
        samplers = {"sampling": sampling} if sampling else {}
        try:
            response = self.runtime.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                tools=tools,
                tool_choice=tool_choice,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
                on_delta=on_delta,
                chat_template_kwargs=chat_template_kwargs,
                stall_timeout=self.STREAM_STALL_SECONDS,
                **thinking,
                **samplers,
            )
            if (on_delta is not None or thinking) and self._is_empty_completion(response):
                # Not every llama.cpp build emits tool-call deltas over SSE.
                # Fall back to the buffered endpoint rather than lose the turn.
                return self.runtime.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                    tools=tools,
                    tool_choice=tool_choice,
                    timeout=self.REQUEST_TIMEOUT_SECONDS,
                    chat_template_kwargs=chat_template_kwargs,
                    **samplers,
                )
            return response
        except LlamaCppStallError:
            raise
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
                chat_template_kwargs=chat_template_kwargs,
                **samplers,
            )
        except LlamaCppRuntimeError as retry_exc:
            if is_llama_timeout_error(retry_exc):
                raise LlamaCppRuntimeError(
                    "llama-server request failed after timeout retry "
                    f"({int(self.REQUEST_TIMEOUT_SECONDS)}s then "
                    f"{int(self.REQUEST_TIMEOUT_RETRY_SECONDS)}s): {retry_exc}"
                ) from retry_exc
            raise

    @staticmethod
    def _build_stream_relay(
        on_stream_delta: Optional[Callable[[str], None]],
    ) -> Optional[Callable[[str], None]]:
        """Turn per-fragment callbacks into running-total callbacks."""
        if on_stream_delta is None:
            return None

        state = {"text": ""}

        def relay(fragment: str) -> None:
            state["text"] += fragment
            on_stream_delta(state["text"])

        return relay

    @staticmethod
    def _is_empty_completion(response: Dict[str, Any]) -> bool:
        """True when a completion carries no assistant text at all."""
        try:
            LlamaCppInterrogator._extract_assistant_content(response)
        except Exception:
            return True
        return False

    @classmethod
    def extract_stream_preview(cls, raw_text: str) -> str:
        """Pull readable prose out of a partially streamed response.

        Responses arrive as JSON tool arguments, so the raw stream is mostly
        scaffolding. This surfaces the `comment` value as it is being written
        and shows nothing until there is something worth reading.
        """
        text = (raw_text or "").strip()
        if not text:
            return ""

        comment = cls._partial_json_string_value(raw_text, "comment")
        if comment is None:
            comment = cls._partial_json_string_value(raw_text, "answer")
        if comment is not None:
            return comment

        if text.startswith("{") or text.startswith("[") or '"' in text[:40]:
            # Structured output that has not reached the comment field yet.
            return ""
        return text

    @staticmethod
    def _partial_json_string_value(raw_text: str, key: str) -> Optional[str]:
        """Read a possibly-unterminated JSON string value for `key`.

        Returns None when the key has not been streamed yet, so callers can
        tell "no value yet" apart from "value is still empty".
        """
        marker = f'"{key}"'
        start = raw_text.find(marker)
        if start == -1:
            return None

        index = start + len(marker)
        length = len(raw_text)
        while index < length and raw_text[index] in " \t\r\n":
            index += 1
        if index >= length or raw_text[index] != ":":
            return None
        index += 1
        while index < length and raw_text[index] in " \t\r\n":
            index += 1
        if index >= length or raw_text[index] != '"':
            return None
        index += 1

        chunk: List[str] = []
        while index < length:
            char = raw_text[index]
            if char == "\\":
                # An escape split across chunks resolves on the next update.
                if index + 1 >= length:
                    break
                chunk.append(raw_text[index : index + 2])
                index += 2
                continue
            if char == '"':
                break
            chunk.append(char)
            index += 1

        candidate = "".join(chunk)
        try:
            return json.loads(f'"{candidate}"')
        except json.JSONDecodeError:
            return candidate

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
        included_transcripts: Optional[List[Dict[str, Any]]] = None,
        sidecar_tags: Optional[List[str]] = None,
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
            "audit": (
                "Goal: audit the sidecar text-file tags against the image.\n"
                "- Treat sidecar_tags as the current tag file contents.\n"
                "- Delete only tags that are clearly erroneous, contradicted by visible image evidence, or unsupported after reviewing optional context.\n"
                "- If uncertain, keep the tag.\n"
                "- tags should be the sidecar tags that should remain.\n"
                "- delete_tags should contain exact sidecar tag strings to remove from the .txt file.\n"
                "- Do not add new tags in this task."
            ),
        }
        base = task_instructions.get(task, task_instructions["describe"])
        user_prompt = prompt.strip() if prompt else ""

        parts = [f"Task: {task}", base]
        if sidecar_tags:
            parts.append("Current sidecar text-file tags:")
            parts.append(json.dumps(list(sidecar_tags), ensure_ascii=False, indent=2))
        if user_prompt:
            parts.append(f"User request: {user_prompt}")
        if included_tables:
            tables_json = json.dumps(included_tables, ensure_ascii=False, indent=2)
            parts.append("Prior interrogation tables (use as context when helpful):")
            parts.append(tables_json)
        if included_transcripts:
            transcripts_json = json.dumps(included_transcripts, ensure_ascii=False, indent=2)
            parts.append("Prior inquiry transcripts (use as context when helpful):")
            parts.append(transcripts_json)
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
            included_transcripts=turn.get("included_transcripts") or [],
            sidecar_tags=turn.get("sidecar_tags") or [],
        )

    @classmethod
    def build_prompt_display_summary(
        cls,
        task: str,
        prompt: str,
        included_tables: List[Dict[str, Any]],
        included_transcripts: Optional[List[Dict[str, Any]]] = None,
        sidecar_tags: Optional[List[str]] = None,
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

        transcript_count = len(included_transcripts or [])
        if transcript_count:
            plural = "turn" if transcript_count == 1 else "turns"
            parts.append(f"Transcript context: {transcript_count} prior inquiry {plural}")

        sidecar_count = len(sidecar_tags or [])
        if sidecar_count:
            plural = "tag" if sidecar_count == 1 else "tags"
            parts.append(f"Sidecar tags: {sidecar_count} {plural}")

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
            "audit": (
                "{\n"
                '  "tags": [list],\n'
                '  "delete_tags": [list],\n'
                '  "comment": "string",\n'
                '  "reasoning_summary": "string",\n'
                '  "warnings": []\n'
                "}"
            ),
        }
        return templates.get(task, templates["describe"])

    @classmethod
    def build_transcript_context(cls, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Compact persisted multimodal turns into prompt-safe transcript context."""
        context: List[Dict[str, Any]] = []
        for turn in turns or []:
            if not isinstance(turn, dict):
                continue
            response = turn.get("response_json", {}) or {}
            if not isinstance(response, dict):
                response = {}
            context.append(
                {
                    "model_name": turn.get("model_name"),
                    "mode": turn.get("mode"),
                    "turn_index": turn.get("turn_index"),
                    "task": turn.get("prompt_type"),
                    "prompt": turn.get("prompt_text") or "",
                    "response_summary": (
                        response.get("comment")
                        or response.get("answer")
                        or response.get("reasoning_summary")
                        or ""
                    ),
                    "tags": turn.get("tags", []) or [],
                    "delete_tags": response.get("delete_tags", []) or [],
                    "warnings": response.get("warnings", []) or [],
                    "created_at": turn.get("created_at"),
                }
            )
        return context

    @classmethod
    def _encode_image_as_data_url(cls, image_path: str, reencode: bool = False) -> str:
        """Encode an image as a data URL llama-server can decode.

        A file in a format llama.cpp reads itself is sent byte for byte. Any
        other format -- judged by content, not extension, so a WebP saved as
        .png counts -- is converted with Pillow, as is a CMYK JPEG or an image
        whose EXIF orientation llama.cpp would ignore (the model then sees it
        upright, as the gallery shows it). `reencode` forces the conversion.
        """
        path = Path(image_path)
        if not path.exists():
            raise ValueError(f"Image does not exist: {image_path}")
        from PIL import Image, UnidentifiedImageError

        try:
            with Image.open(path) as image:
                image_format = (image.format or "").upper()
                try:
                    orientation = image.getexif().get(0x0112, 1)
                except Exception:  # noqa: BLE001 - malformed EXIF just means "upright"
                    orientation = 1
                if (
                    reencode
                    or image_format not in LLAMA_NATIVE_IMAGE_FORMATS
                    or image.mode == "CMYK"
                    or orientation not in (0, 1)
                ):
                    data, mime = cls._reencode_image(image)
                    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
        except Image.DecompressionBombError:
            # Too many pixels for Pillow to open even to look; send it as before.
            image_format = path.suffix.lower().lstrip(".").replace("jpg", "jpeg").upper() or "PNG"
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Cannot read image {path.name}: {exc}") from exc

        mime = "image/jpeg" if image_format in ("JPEG", "MPO") else f"image/{image_format.lower()}"
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"

    @staticmethod
    def _reencode_image(image: Any) -> Tuple[bytes, str]:
        """Upright first frame as PNG when it has transparency, else JPEG."""
        from PIL import ImageOps

        image = ImageOps.exif_transpose(image)
        buffer = io.BytesIO()
        if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
            image.convert("RGBA").save(buffer, "PNG", compress_level=1)
            return buffer.getvalue(), "image/png"
        image.convert("RGB").save(buffer, "JPEG", quality=95, subsampling=0)
        return buffer.getvalue(), "image/jpeg"

    @staticmethod
    def _is_image_load_error(exc: Exception) -> bool:
        detail = str(exc).lower()
        return "http error 400" in detail and any(marker in detail for marker in _IMAGE_LOAD_ERRORS)

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
                            "delete_tags": {
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

        delete_tags = cls._normalize_tag_list(parsed.get("delete_tags", []))

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
            "delete_tags": delete_tags,
            "warnings": warnings,
        }
        return normalized

    @staticmethod
    def _normalize_tag_list(value: Any) -> List[str]:
        if not isinstance(value, list):
            return []
        normalized: List[str] = []
        seen = set()
        for item in value:
            if not isinstance(item, str):
                continue
            clean = item.strip()
            if not clean:
                continue
            key = clean.casefold()
            if key in seen:
                continue
            normalized.append(clean)
            seen.add(key)
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
                    "reasoning_summary (string), delete_tags (string[]), warnings (string[]). "
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
            "delete_tags": [],
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
