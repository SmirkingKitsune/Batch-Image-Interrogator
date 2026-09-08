"""Context-size sizing from GGUF architecture metadata.

Picking a context size is three separate questions that are easy to conflate:

1. What was the model trained for? (`model_max_context`)
2. What will fit in memory? (`kv_cache_bytes` against a budget)
3. What does the workload actually need?

Only the third should drive the value. The first two are ceilings, and using a
ceiling as a target is how a 6,300-token job ends up configured for 131,072 —
which on a unified-memory machine reserves tens of gigabytes of KV cache that
nothing ever reads.

The KV figure here is computed from the architecture tensors rather than
estimated from the weight file's size. Weight size is a poor proxy: a model with
aggressive grouped-query attention (few KV heads against many attention heads)
carries far less cache per byte of weights than a multi-head model of the same
size, and the error runs to an order of magnitude in the GQA direction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from core.gguf_metadata import context_length_from_metadata

# Bytes per stored element, by llama.cpp KV cache type. The quantized entries
# are block formats: q8_0 packs 32 values into 34 bytes (32 quants + one f16
# scale), q4_0 into 18, so the per-element cost is not a whole number.
KV_TYPE_BYTES: Dict[str, float] = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34.0 / 32.0,
    "q5_1": 24.0 / 32.0,
    "q4_0": 18.0 / 32.0,
}

DEFAULT_KV_TYPE = "f16"

# Floor for a suggestion. Below this a multimodal turn cannot hold one image
# plus its prompt, so a "fits in memory" answer would be useless anyway.
MIN_SUGGESTED_CONTEXT = 2048

# Fraction of the memory budget the KV cache may claim. The rest has to cover
# weights, the compute buffers, the image encoder, and everything else on the
# machine; on unified memory that includes the desktop session.
KV_BUDGET_FRACTION = 0.35


@dataclass
class ContextSuggestion:
    """A suggested context size and the bounds that produced it."""

    suggested_ctx: int
    workload_ctx: int
    model_max: Optional[int] = None
    memory_max: Optional[int] = None
    bytes_per_token: Optional[int] = None
    kv_bytes_at_suggested: Optional[int] = None
    bound_by: str = "workload"
    notes: list = field(default_factory=list)


def _arch_prefix(metadata: Dict[str, Any]) -> Optional[str]:
    """GGUF namespaces architecture keys under the architecture's own name.

    Falls back to recovering the prefix from `<arch>.block_count`, since a
    conversion that omits `general.architecture` still has to namespace the
    layer count somewhere.
    """
    arch = metadata.get("general.architecture")
    if isinstance(arch, str) and arch:
        return str(arch)
    for key in metadata:
        if isinstance(key, str) and key.endswith(".block_count"):
            return key[: -len(".block_count")]
    return None


def _int_key(metadata: Dict[str, Any], arch: str, suffix: str) -> Optional[int]:
    value = metadata.get(f"{arch}.{suffix}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    return number if number > 0 else None


def model_max_context(metadata: Dict[str, Any]) -> Optional[int]:
    """Context length the model was trained for, if declared.

    Delegates so the known-architecture fallbacks live in one place.
    """
    return context_length_from_metadata(metadata)


def kv_bytes_per_token(
    metadata: Dict[str, Any],
    kv_type: str = DEFAULT_KV_TYPE,
) -> Optional[int]:
    """Dense-attention KV estimate per token, or None if undeterminable.

    Assumes every declared layer stores full-context KV. This is not an
    allocation measurement: hybrid/recurrent layers, sliding windows, padding,
    and runtime cache settings can change the actual allocation.

    Both halves of the cache are counted: `key_length` and `value_length` are
    per-head dimensions and are not always equal, so they are summed rather
    than one being doubled.
    """
    arch = _arch_prefix(metadata)
    if not arch:
        return None

    layers = _int_key(metadata, arch, "block_count")
    if not layers:
        return None

    heads = _int_key(metadata, arch, "attention.head_count")
    # Absent head_count_kv means multi-head attention, where every attention
    # head carries its own KV. Grouped-query models always declare it.
    kv_heads = _int_key(metadata, arch, "attention.head_count_kv") or heads
    if not kv_heads:
        return None

    key_len = _int_key(metadata, arch, "attention.key_length")
    value_len = _int_key(metadata, arch, "attention.value_length")
    if key_len is None or value_len is None:
        # Older conversions omit these. The conventional derivation is
        # embedding width split evenly across attention heads -- note this
        # divides by head_count, not head_count_kv, because it is describing
        # the head dimension rather than the number of cached heads.
        embedding = _int_key(metadata, arch, "embedding_length")
        if not embedding or not heads:
            return None
        derived = embedding // heads
        if derived <= 0 or embedding % heads:
            return None
        key_len = key_len or derived
        value_len = value_len or derived

    element_bytes = KV_TYPE_BYTES.get(kv_type.lower())
    if element_bytes is None:
        return None
    per_token = layers * kv_heads * (key_len + value_len) * element_bytes
    return int(round(per_token))


def kv_cache_bytes(
    metadata: Dict[str, Any],
    ctx_tokens: int,
    kv_type: str = DEFAULT_KV_TYPE,
) -> Optional[int]:
    """Total KV cache bytes for a context of `ctx_tokens`."""
    per_token = kv_bytes_per_token(metadata, kv_type=kv_type)
    if per_token is None or ctx_tokens <= 0:
        return None
    return per_token * int(ctx_tokens)


def available_memory_bytes() -> Optional[int]:
    """Host memory available right now, or None when it cannot be read.

    `MemAvailable` rather than `MemFree`: page cache is reclaimable, and on a
    machine that has just walked a large image set nearly all free memory shows
    up as cache. Reading MemFree would under-report by tens of gigabytes.

    Host memory is the right budget on unified-memory hardware. With a discrete
    GPU the real ceiling is VRAM, and this only bounds the host side.
    """
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _round_up_pow2(value: int) -> int:
    result = 1
    while result < value:
        result *= 2
    return result


def suggest_context_size(
    metadata: Dict[str, Any],
    prompt_tokens: int,
    max_tokens: int,
    available_bytes: Optional[int] = None,
    kv_type: str = DEFAULT_KV_TYPE,
) -> ContextSuggestion:
    """Suggest a context size from the workload, clamped by model and memory.

    `prompt_tokens` is the largest prompt the job is expected to send and
    `max_tokens` the reply budget; together they are what the context has to
    hold. Rounding up to a power of two supplies the headroom, so no separate
    fudge factor is applied.
    """
    notes: list = []
    workload = max(int(prompt_tokens), 0) + max(int(max_tokens), 0)
    workload_ctx = max(_round_up_pow2(max(workload, 1)), MIN_SUGGESTED_CONTEXT)

    suggested = workload_ctx
    bound_by = "workload"

    model_max = model_max_context(metadata)
    if model_max and suggested > model_max:
        suggested = model_max
        bound_by = "model"
        notes.append(
            f"Clamped to the model's trained context of {model_max:,}."
        )

    per_token = kv_bytes_per_token(metadata, kv_type=kv_type)
    memory_max: Optional[int] = None
    if per_token and available_bytes is not None:
        budget = int(max(available_bytes, 0) * KV_BUDGET_FRACTION)
        memory_max = max(budget // per_token, 0)
        if suggested > memory_max:
            suggested = _round_down_pow2(memory_max)
            bound_by = "memory"
            notes.append(
                "Clamped by available memory; the workload wanted "
                f"{workload_ctx:,}."
            )

    return ContextSuggestion(
        suggested_ctx=suggested,
        workload_ctx=workload_ctx,
        model_max=model_max,
        memory_max=memory_max,
        bytes_per_token=per_token,
        kv_bytes_at_suggested=(per_token * suggested) if per_token else None,
        bound_by=bound_by,
        notes=notes,
    )


def _round_down_pow2(value: int) -> int:
    if value < 1:
        return 0
    result = 1
    while result * 2 <= value:
        result *= 2
    return result


def format_bytes(value: Optional[int]) -> str:
    """Human-readable size for status text."""
    if value is None:
        return "unknown"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TiB"
