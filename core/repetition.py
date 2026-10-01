"""Keeping a llama model from getting stuck repeating itself.

Greedy decoding can fall into a loop -- a watermark handle transcribed as
"@Inarototototo…", or the same sentence over and over -- that runs until
max_tokens, which can mean hours. Two layers deal with it:

* `DRY_SAMPLING` goes out with every request: llama.cpp's DRY ("don't repeat
  yourself") sampler stops most loops before they start.
* `LoopDetector` watches the streamed thinking and answer for the loops that
  get through, so the request can be abandoned and retried once with the
  model's recommended sampling (`rescue_sampling`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

# DRY penalizes a token by how long a repeated run it would extend:
# multiplier * base ** (run_length - allowed_length). llama.cpp's default
# sequence breakers (newline, ':', '"', '*') end a run, so JSON structure and
# quoted tags copied from the thinking never build one; a run inside a single
# string, like a looping handle, does. The penalty is applied before the
# temperature step, so greedy decoding stays deterministic.
DRY_SAMPLING: Dict[str, Any] = {
    "dry_multiplier": 0.8,
    "dry_base": 1.75,
    "dry_allowed_length": 4,
    "dry_penalty_last_n": 256,
}

# The single retry after a loop: a firmer DRY penalty, plus the publisher's
# recommended sampling (read from the GGUF), or these values without one.
RESCUE_DRY_SAMPLING: Dict[str, Any] = {
    "dry_multiplier": 1.2,
    "dry_base": 1.75,
    "dry_allowed_length": 3,
    "dry_penalty_last_n": 512,
}
DEFAULT_RESCUE_SAMPLING: Dict[str, Any] = {"temperature": 0.6, "top_k": 20, "top_p": 0.95}

MIN_LOOP_CHARS = 200
MAX_UNIT_CHARS = 400


def find_loop(text: str, min_chars: int = MIN_LOOP_CHARS, max_unit: int = MAX_UNIT_CHARS) -> Optional[str]:
    """The unit `text` ends by repeating, when it is stuck in a loop.

    The tail must repeat one unit back to back over at least `min_chars`
    characters, and at least four times (three for units of 100 characters or
    more). Repetition that is meant -- an ellipsis, a separator line, "BUY NOW!
    BUY NOW!" read off an image -- stays well short of that.
    """
    n = len(text)
    if n < min_chars:
        return None
    for unit in range(1, min(max_unit, n // 3) + 1):
        span = max(min_chars, (4 if unit < 100 else 3) * unit)
        if span > n:
            break
        # Periodic with this unit over the whole span: each character equals
        # the one `unit` places later.
        if text[n - span:n - unit] == text[n - span + unit:]:
            return text[n - unit:]
    return None


def loop_excerpt(unit: str, width: int = 32) -> str:
    """A one-line sample of a loop for messages: “totototo…”."""
    sample = unit if len(unit) >= width else unit * (width // len(unit) + 1)
    sample = " ".join(sample[:width].split())
    return f"“{sample}…”"


def rescue_sampling(recommended: Dict[str, Any], temperature: float) -> Tuple[float, Dict[str, Any]]:
    """Temperature and sampler fields for the retry after a loop.

    Never colder than the configured temperature: a loop under greedy decoding
    repeats exactly, so the retry needs the randomness the model was tuned for.
    """
    settings = dict(DEFAULT_RESCUE_SAMPLING)
    settings.update(recommended or {})
    rescue_temperature = max(float(temperature), float(settings.pop("temperature")))
    return rescue_temperature, {**settings, **RESCUE_DRY_SAMPLING}


class LoopDetector:
    """Feeds a stream's fragments to `find_loop`, a few dozen characters apart."""

    CHECK_EVERY_CHARS = 48
    TAIL_CHARS = 3 * MAX_UNIT_CHARS

    def __init__(self):
        self._tail = ""
        self._unchecked = 0

    def feed(self, fragment: str) -> Optional[str]:
        """The repeating unit once the stream is looping, else None."""
        if not fragment:
            return None
        self._tail = (self._tail + fragment)[-self.TAIL_CHARS:]
        self._unchecked += len(fragment)
        if self._unchecked < self.CHECK_EVERY_CHARS:
            return None
        self._unchecked = 0
        return find_loop(self._tail)
