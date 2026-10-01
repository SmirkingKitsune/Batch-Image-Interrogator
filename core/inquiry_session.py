"""Persistence for single-image llama.cpp inquiries.

A single inquiry runs on a worker thread, but what happens to its result is
shared by both front ends: audit deletions are applied to the sidecar, the
image and model are registered, the result becomes the model's latest
interrogation, and the turn is appended to the image's session history. The
PyQt6 Inquiry tab and the Electron bridge both call `record_single_inquiry`.
"""

from pathlib import Path
from typing import Any, Dict

from core.database import InterrogationDatabase
from core.file_manager import FileManager
from core.hashing import get_image_metadata, hash_image_content


def apply_audit_result(image_path: str, results: Dict[str, Any]) -> None:
    """Delete the tags an audit rejected from the sidecar, in place on `results`."""
    removed_tags, remaining_tags = FileManager.delete_tags_from_file(
        Path(image_path),
        (results.get("multimodal_response") or {}).get("delete_tags", []),
    )
    response_json = results.get("multimodal_response", {}) or {}
    response_json["removed_tags"] = removed_tags
    response_json["remaining_tags"] = remaining_tags
    results["multimodal_response"] = response_json
    results["audit_removed_tags"] = removed_tags
    results["audit_remaining_tags"] = remaining_tags
    results["tags"] = remaining_tags


def record_single_inquiry(
    database: InterrogationDatabase,
    request: Dict[str, Any],
    results: Dict[str, Any],
) -> Dict[str, Any]:
    """Persist a finished single-image inquiry.

    `request` carries task, prompt_text, included_tables, included_transcripts,
    sidecar_tags, image_path, session_key, image_hash (optional), model_name,
    model_type and model_config. `results` is the interrogator's return value
    and is updated in place (audit tags, remaining tags).

    Returns {"file_hash": str, "turn": dict} where turn is the transcript entry
    for the completed request.
    """
    task = request.get("task", "describe")
    prompt_text = request.get("prompt_text", "")
    included_tables = request.get("included_tables", []) or []
    included_transcripts = request.get("included_transcripts", []) or []
    sidecar_tags = request.get("sidecar_tags", []) or []
    image_path = request.get("image_path")
    session_key = request.get("session_key")

    if task == "audit":
        apply_audit_result(image_path, results)

    file_hash = request.get("image_hash") or hash_image_content(image_path)
    metadata = get_image_metadata(image_path)
    image_id = database.register_image(
        image_path,
        file_hash,
        metadata["width"],
        metadata["height"],
        metadata["file_size"],
    )
    model_id = database.register_model(
        request["model_name"],
        request["model_type"],
        config=request["model_config"],
    )
    database.save_interrogation(
        image_id,
        model_id,
        results["tags"],
        results.get("confidence_scores"),
        results.get("raw_output"),
    )

    session_id = database.create_or_get_multimodal_session(
        image_id=image_id,
        model_id=model_id,
        mode="single",
        session_key=session_key,
    )
    response_json = results.get("multimodal_response", {})
    database.append_multimodal_turn(
        session_id=session_id,
        prompt_type=task,
        prompt_text=prompt_text,
        included_tables=included_tables,
        included_transcripts=included_transcripts,
        sidecar_tags=sidecar_tags,
        response_json=response_json,
        tags=results["tags"],
        reasoning_summary=response_json.get("reasoning_summary", ""),
    )

    return {
        "file_hash": file_hash,
        "turn": {
            "prompt_type": task,
            "prompt_text": prompt_text,
            "included_tables": included_tables,
            "included_transcripts": included_transcripts,
            "sidecar_tags": sidecar_tags,
            "response_json": response_json,
            "tags": results.get("tags", []) or [],
            "model_name": request["model_name"],
            "image_path": image_path,
        },
    }
