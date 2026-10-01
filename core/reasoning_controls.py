"""What a model's chat template lets a request change about its thinking.

Reasoning models expose their switches as jinja template variables, and
llama-server forwards per-request values for them from `chat_template_kwargs`:

* ``enable_thinking`` (the Qwen3 family and others) turns thinking off.
* ``reasoning_effort`` sets how hard the model thinks. The accepted values
  differ by model -- gpt-oss takes low/medium/high, Qwen3.8 takes
  low/medium/xhigh and raises on anything else -- so they are read from the
  template rather than assumed. Sending a value a template rejects fails the
  whole request.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

EFFORT_VARIABLE = "reasoning_effort"
THINKING_VARIABLE = "enable_thinking"

# Used when a template reads reasoning_effort without listing what it accepts
# (gpt-oss interpolates it into the system prompt as-is).
FALLBACK_EFFORT_VALUES = ["low", "medium", "high"]

_EFFORT_ORDER = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_STRING = r"""(?:'([^'\\]*)'|"([^"\\]*)")"""


def detect_reasoning_controls(template: Optional[str]) -> Dict[str, Any]:
    """Describe the reasoning variables a chat template reads.

    Returns ``{"thinking_toggle": bool, "effort": None | {"values": [...],
    "default": str | None, "aliases": {...}}}``. Values are ordered from least
    to most effort; aliases map accepted spellings onto listed values (Qwen3.8
    treats "high" as "xhigh").
    """
    text = template or ""
    controls: Dict[str, Any] = {
        "thinking_toggle": bool(re.search(rf"\b{THINKING_VARIABLE}\b", text)),
        "effort": None,
    }
    if re.search(rf"\b{EFFORT_VARIABLE}\b", text):
        controls["effort"] = _detect_effort(text)
    return controls


def normalize_effort(controls: Optional[Dict[str, Any]], value: Optional[str]) -> Optional[str]:
    """The effort to send for `value`, or None when this model does not take it."""
    effort = (controls or {}).get("effort")
    if not effort or not value:
        return None
    value = str(value).strip()
    if value in effort["values"]:
        return value
    return effort["aliases"].get(value)


def describe_reasoning_controls(controls: Optional[Dict[str, Any]]) -> List[str]:
    """Human-readable lines for a metadata report."""
    if not controls:
        return []
    lines = []
    effort = controls.get("effort")
    if effort:
        default = f" (template default {effort['default']})" if effort.get("default") else ""
        lines.append(f"Reasoning effort: {' · '.join(effort['values'])}{default}.")
    if controls.get("thinking_toggle"):
        lines.append("Thinking can be switched off per request (enable_thinking).")
    return lines


def reasoning_controls_for_model(model_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Read the controls from a GGUF's embedded chat template; None if unreadable."""
    if not model_path:
        return None
    from core.gguf_metadata import read_gguf_metadata

    try:
        metadata = read_gguf_metadata(model_path)
    except Exception:  # noqa: BLE001 - a missing or foreign file just means "unknown"
        return None
    return reasoning_controls_from_metadata(metadata)


def reasoning_controls_from_metadata(metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    template = metadata.get("tokenizer.chat_template")
    if not isinstance(template, str) or not template:
        return None
    return detect_reasoning_controls(template)


def _detect_effort(text: str) -> Dict[str, Any]:
    names = _effort_names(text)
    name_pattern = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))

    aliases: Dict[str, str] = {}
    alias_pattern = (
        rf"\{{%-?\s*if\s+({name_pattern})\s*==\s*{_STRING}\s*-?%\}}\s*"
        rf"\{{%-?\s*set\s+\1\s*=\s*{_STRING}\s*-?%\}}"
    )
    for match in re.finditer(alias_pattern, text):
        source = match.group(2) if match.group(2) is not None else match.group(3)
        target = match.group(4) if match.group(4) is not None else match.group(5)
        if source != target:
            aliases[source] = target

    values: List[str] = []
    for match in re.finditer(rf"\b(?:{name_pattern})\s+(?:not\s+)?in\s*[\(\[]([^\)\]]*)[\)\]]", text):
        values.extend(v for v in _strings(match.group(1)) if v not in values)
    if not values:
        for match in re.finditer(rf"\b(?:{name_pattern})\s*==\s*{_STRING}", text):
            value = match.group(1) if match.group(1) is not None else match.group(2)
            if value not in values and value not in aliases:
                values.append(value)
    if not values:
        values = list(FALLBACK_EFFORT_VALUES)

    default = None
    match = re.search(rf"\b{EFFORT_VARIABLE}\s*\|\s*default\(\s*{_STRING}", text)
    if match is None:
        match = re.search(rf"\{{%-?\s*set\s+{EFFORT_VARIABLE}\s*=\s*{_STRING}\s*-?%\}}", text)
    if match is not None:
        default = match.group(1) if match.group(1) is not None else match.group(2)
        default = aliases.get(default, default)
        if default not in values:
            default = None

    aliases = {source: target for source, target in aliases.items() if target in values and source not in values}
    order = {name: rank for rank, name in enumerate(_EFFORT_ORDER)}
    position = {value: index for index, value in enumerate(values)}
    values = sorted(values, key=lambda v: (order.get(v, len(order)), position[v]))
    return {"values": values, "default": default, "aliases": aliases}


def _effort_names(text: str) -> set:
    """reasoning_effort plus template variables assigned from it."""
    names = {EFFORT_VARIABLE}
    for _ in range(3):
        pattern = "|".join(re.escape(name) for name in names)
        found = set(re.findall(rf"\{{%-?\s*set\s+(\w+)\s*=\s*(?:{pattern})\b[^%]*-?%\}}", text))
        if found <= names:
            break
        names |= found
    return names


def _strings(fragment: str) -> List[str]:
    return [a if a else b for a, b in re.findall(_STRING, fragment)]
