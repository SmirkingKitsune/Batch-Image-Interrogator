"""Database management for image interrogations with SQLite."""

import os
import sqlite3
import json
import functools
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple, Any

from core.atomic_write import write_json_atomic
from core.db_connection import SQLiteConnectionManager
from core.db_queue import DatabaseOperationQueue
from core.db_schema import ensure_schema

# Local-database mode keeps one of these in each selected image directory.
LOCAL_DB_NAME = ".interrogations.db"
# Remembers the last database mode, beside the global database, for both front ends.
SETTINGS_FILE_NAME = "database_settings.json"


class DatabaseBusyError(Exception):
    """Raised when database is locked and user chose to abort."""
    pass


class DatabaseQueuedError(Exception):
    """Raised when operation was queued instead of executed."""

    def __init__(self, operation_id: str, message: str = "Operation queued"):
        self.operation_id = operation_id
        super().__init__(message)


def _is_busy_error(error: sqlite3.OperationalError) -> bool:
    """Recognize SQLite BUSY/LOCKED errors without relying only on wording."""
    error_code = getattr(error, "sqlite_errorcode", None)
    if error_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
        return True
    message = str(error).casefold()
    return "locked" in message or "busy" in message


def _retry_on_busy(max_attempts: int = 5, base_delay: float = 0.1):
    """
    Decorator to retry database operations on SQLITE_BUSY errors.

    After max_attempts silent retries fail, if a busy_callback is set on the
    database instance, it will be called to let the UI show a dialog.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            operation_name = func.__name__
            silent_attempts = max_attempts

            for attempt in range(silent_attempts):
                try:
                    return func(self, *args, **kwargs)
                except sqlite3.OperationalError as e:
                    if _is_busy_error(e) and attempt < silent_attempts - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        time.sleep(delay)
                    elif _is_busy_error(e):
                        # Silent retries exhausted - invoke callback if available
                        if self._busy_callback is not None:
                            params = _build_params_dict(func, args, kwargs)
                            result = self._handle_busy_with_callback(
                                operation_name, params, silent_attempts, e, func
                            )
                            if result is not None:
                                return result
                            # If result is None, callback handled it (queued or aborted)
                            return None
                        else:
                            raise
                    else:
                        raise
        return wrapper
    return decorator


def _build_params_dict(func, args, kwargs) -> Dict[str, Any]:
    """Build a dictionary of parameters from function args/kwargs."""
    import inspect
    sig = inspect.signature(func)
    params = list(sig.parameters.keys())

    result = {}
    # Skip 'self' parameter
    for i, arg in enumerate(args):
        if i + 1 < len(params):  # +1 to skip self
            result[params[i + 1]] = arg

    result.update(kwargs)
    return result


class InterrogationDatabase:
    """Manages interrogation cache and results using SQLite."""

    def __init__(self, db_path: str = "interrogations.db"):
        self._global_db_path = Path(db_path)
        self._connections = SQLiteConnectionManager(self._global_db_path)
        self._settings_path = self._global_db_path.with_name(SETTINGS_FILE_NAME)
        # The mode last chosen with set_local_mode(); global until one is.
        self.use_local_db = self._load_settings().get("use_local_db") is True
        # The directory last selected for interrogation. In global mode, local
        # databases under it are consulted for results the global one lacks.
        self._directory_root: Optional[Path] = None
        self._local_db_presence: Dict[Path, bool] = {}

        # Queue for handling database-busy scenarios
        self._operation_queue: Optional[DatabaseOperationQueue] = None
        self._busy_callback: Optional[Callable[[str, Dict, int], str]] = None

        self._ensure_schema()
        self._init_operation_queue()
        self._process_pending_queue()

    @property
    def db_path(self) -> Path:
        """Return the active database path (retained for caller compatibility)."""
        return self._connections.path

    @db_path.setter
    def db_path(self, value: Path) -> None:
        """Switch database paths safely for compatibility with legacy callers."""
        self._switch_database_path(Path(value))

    @contextmanager
    def _get_connection(self, *, write: bool = False):
        """Context manager for thread-safe database connections.

        Opens a fresh connection for each operation with:
        - 30 second timeout for lock acquisition
        - WAL mode for improved concurrent read/write
        - Thread lock for intra-process synchronization
        """
        with self._connections.connection(write=write) as conn:
            yield conn

    def _ensure_schema(self):
        """Create database schema if it doesn't exist."""
        with self._get_connection(write=True) as conn:
            ensure_schema(conn)
    
    @_retry_on_busy()
    def register_image(self, file_path: str, file_hash: str,
                      width: int, height: int, file_size: int) -> int:
        """Register an image and return its ID.

        Uses INSERT OR IGNORE + UPDATE pattern to avoid race conditions.
        """
        with self._get_connection(write=True) as conn:
            cursor = conn.cursor()

            # Use INSERT OR IGNORE to atomically insert if not exists
            cursor.execute("""
                INSERT OR IGNORE INTO images (file_path, file_hash, width, height, file_size)
                VALUES (?, ?, ?, ?, ?)
            """, (file_path, file_hash, width, height, file_size))

            # Update path if image already existed (may have moved)
            cursor.execute("""
                UPDATE images
                SET file_path = ?, updated_at = CURRENT_TIMESTAMP
                WHERE file_hash = ?
            """, (file_path, file_hash))

            cursor.execute("SELECT id FROM images WHERE file_hash = ?", (file_hash,))
            return cursor.fetchone()['id']
    
    @_retry_on_busy()
    def register_model(self, model_name: str, model_type: str,
                      version: Optional[str] = None,
                      config: Optional[Dict] = None) -> int:
        """Register a model and return its ID.

        Uses INSERT OR IGNORE pattern to avoid race conditions.
        """
        with self._get_connection(write=True) as conn:
            cursor = conn.cursor()

            config_json = json.dumps(config) if config else None

            # Use INSERT OR IGNORE to atomically insert if not exists
            cursor.execute("""
                INSERT OR IGNORE INTO models (model_name, model_type, version, config)
                VALUES (?, ?, ?, ?)
            """, (model_name, model_type, version, config_json))

            cursor.execute("SELECT id FROM models WHERE model_name = ?", (model_name,))
            return cursor.fetchone()['id']
    
    @_retry_on_busy()
    def save_interrogation(self, image_id: int, model_id: int,
                          tags: List[str],
                          confidence_scores: Optional[Dict[str, float]] = None,
                          raw_output: Optional[str] = None,
                          result_version: Optional[int] = None,
                          config: Optional[Dict[str, Any]] = None):
        """Save or update interrogation results.

        `config` is the interrogator's result_signature for this result.
        """
        return self._save_interrogation_impl(
            image_id, model_id, tags, confidence_scores, raw_output, result_version, config
        )

    def _save_interrogation_impl(
        self,
        image_id: int,
        model_id: int,
        tags: List[str],
        confidence_scores: Optional[Dict[str, float]] = None,
        raw_output: Optional[str] = None,
        result_version: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Shared implementation used by calls, retries, and queue replay."""
        with self._get_connection(write=True) as conn:
            cursor = conn.cursor()

            tags_json = json.dumps(tags)
            scores_json = json.dumps(confidence_scores) if confidence_scores else None
            config_json = json.dumps(config) if config else None

            cursor.execute("""
                INSERT INTO interrogations (image_id, model_id, tags, confidence_scores, raw_output, config, result_version)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id, model_id)
                DO UPDATE SET
                    tags = excluded.tags,
                    confidence_scores = excluded.confidence_scores,
                    raw_output = excluded.raw_output,
                    config = excluded.config,
                    result_version = excluded.result_version,
                    interrogated_at = CURRENT_TIMESTAMP
            """, (image_id, model_id, tags_json, scores_json, raw_output, config_json, result_version))

    @_retry_on_busy()
    def get_interrogation(self, image_hash: str, model_name: str) -> Optional[Dict]:
        """Retrieve cached interrogation if it exists."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT i.tags, i.confidence_scores, i.raw_output, i.interrogated_at, i.config, i.result_version
                FROM interrogations i
                JOIN images img ON i.image_id = img.id
                JOIN models m ON i.model_id = m.id
                WHERE img.file_hash = ? AND m.model_name = ?
            """, (image_hash, model_name))

            result = cursor.fetchone()
            if result:
                return {
                    'tags': json.loads(result['tags']),
                    'confidence_scores': json.loads(result['confidence_scores']) if result['confidence_scores'] else None,
                    'raw_output': result['raw_output'],
                    'interrogated_at': result['interrogated_at'],
                    'result_version': result['result_version'] or 0,
                    'config': json.loads(result['config']) if result['config'] else None
                }
            return None

    @_retry_on_busy()
    def import_missing_interrogation(self, image_path: str, image_hash: str, model_name: str) -> bool:
        """Copy a result the active database lacks from the other database mode.

        Global mode reads the local databases in the image's folder and each
        parent up to the selected directory, nearest first. Local mode reads the
        global database. Sources are only read; the copy goes into the active
        database, and an entry it already has is left alone. Returns True when
        an entry was copied, so the caller can look it up again through the
        normal cache path.
        """
        for source in self._import_sources_for(image_path):
            entry = self._read_interrogation(source, image_hash, model_name)
            if entry is not None:
                return self._copy_interrogation(image_path, image_hash, model_name, entry)
        return False

    def _import_sources_for(self, image_path: str) -> List[Path]:
        """Databases to read a missing result from, best candidate first."""
        if not self.use_local_db:
            return self._local_databases_for(image_path)
        global_db = Path(os.path.abspath(self._global_db_path))
        if global_db == Path(os.path.abspath(self.db_path)) or not global_db.is_file():
            # No directory selected yet, so the global database is the active one.
            return []
        return [global_db]

    def _local_databases_for(self, image_path: str) -> List[Path]:
        """Local databases that may hold this image's results, nearest first.

        A recursive scan can reach folders that local mode gave their own
        database, so every folder from the image's up to the selected directory
        is checked, not just the selected directory itself.
        """
        root = self._directory_root
        if root is None:
            return []
        folder = Path(os.path.abspath(image_path)).parent
        try:
            folder.relative_to(root)
        except ValueError:
            return []

        active = Path(os.path.abspath(self.db_path))
        found = []
        while True:
            candidate = folder / LOCAL_DB_NAME
            present = self._local_db_presence.get(candidate)
            if present is None:
                present = candidate.is_file()
                self._local_db_presence[candidate] = present
            if present and candidate != active:
                found.append(candidate)
            if folder == root or folder.parent == folder:
                break
            folder = folder.parent
        return found

    @staticmethod
    def _read_interrogation(
        source_db: Path, image_hash: str, model_name: str
    ) -> Optional[Dict[str, Any]]:
        """Read one result from another database without writing beside it.

        A cleanly closed database has no -wal file, so it is opened immutable,
        which takes no locks and leaves no -wal/-shm files in an image folder.
        With a -wal present (open elsewhere, or not closed cleanly) a plain
        read-only open is needed to see its newest writes.
        """
        wal = source_db.with_name(source_db.name + "-wal")
        mode = "mode=ro" if wal.exists() else "mode=ro&immutable=1"
        try:
            conn = sqlite3.connect(f"{source_db.resolve().as_uri()}?{mode}", uri=True, timeout=5.0)
        except (sqlite3.Error, OSError):
            return None
        try:
            conn.row_factory = sqlite3.Row
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(interrogations)")}
            if not columns:
                return None
            # Databases written before these columns existed still hold usable
            # results; the caller's cache rules decide whether to reuse them.
            optional = ", ".join(
                f"i.{name}" if name in columns else f"NULL AS {name}"
                for name in ("result_version", "config")
            )
            row = conn.execute(f"""
                SELECT i.tags, i.confidence_scores, i.raw_output, i.interrogated_at, {optional},
                       img.width, img.height, img.file_size,
                       m.model_type, m.version, m.config AS model_config
                FROM interrogations i
                JOIN images img ON i.image_id = img.id
                JOIN models m ON i.model_id = m.id
                WHERE img.file_hash = ? AND m.model_name = ?
            """, (image_hash, model_name)).fetchone()
            return dict(row) if row else None
        except sqlite3.Error:
            return None
        finally:
            conn.close()

    def _copy_interrogation(
        self, image_path: str, image_hash: str, model_name: str, entry: Dict[str, Any]
    ) -> bool:
        """Insert a local result, in one transaction, unless one already exists."""
        with self._get_connection(write=True) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO images (file_path, file_hash, width, height, file_size)
                VALUES (?, ?, ?, ?, ?)
            """, (image_path, image_hash, entry["width"], entry["height"], entry["file_size"]))
            cursor.execute("SELECT id FROM images WHERE file_hash = ?", (image_hash,))
            image_id = cursor.fetchone()["id"]

            cursor.execute("""
                INSERT OR IGNORE INTO models (model_name, model_type, version, config)
                VALUES (?, ?, ?, ?)
            """, (model_name, entry["model_type"], entry["version"], entry["model_config"]))
            cursor.execute("SELECT id FROM models WHERE model_name = ?", (model_name,))
            model_id = cursor.fetchone()["id"]

            # Copied verbatim, original timestamp included.
            cursor.execute("""
                INSERT INTO interrogations (image_id, model_id, tags, confidence_scores, raw_output,
                                            result_version, config, interrogated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, COALESCE(?, CURRENT_TIMESTAMP))
                ON CONFLICT(image_id, model_id) DO NOTHING
            """, (
                image_id, model_id, entry["tags"], entry["confidence_scores"], entry["raw_output"],
                entry["result_version"], entry["config"], entry["interrogated_at"],
            ))
            return cursor.rowcount > 0

    @_retry_on_busy()
    def save_interrogation_cache_entry(
        self,
        image_id: int,
        model_id: int,
        cache_key: str,
        cache_metadata: Dict[str, Any],
        results: Dict[str, Any],
    ) -> None:
        """Save or update an exact-match interrogation cache entry."""
        if not cache_key:
            raise ValueError("cache_key is required")

        with self._get_connection(write=True) as conn:
            cursor = conn.cursor()

            tags = list(results.get("tags") or [])
            confidence_scores = results.get("confidence_scores")
            raw_output = results.get("raw_output")

            cursor.execute(
                """
                INSERT INTO interrogation_cache_entries (
                    image_id, model_id, cache_key, cache_metadata,
                    result_json, tags, confidence_scores, raw_output
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id, model_id, cache_key)
                DO UPDATE SET
                    cache_metadata = excluded.cache_metadata,
                    result_json = excluded.result_json,
                    tags = excluded.tags,
                    confidence_scores = excluded.confidence_scores,
                    raw_output = excluded.raw_output,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    image_id,
                    model_id,
                    cache_key,
                    json.dumps(cache_metadata or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(results or {}, ensure_ascii=False, sort_keys=True),
                    json.dumps(tags, ensure_ascii=False),
                    json.dumps(confidence_scores, ensure_ascii=False) if confidence_scores else None,
                    raw_output,
                ),
            )

    @_retry_on_busy()
    def get_interrogation_cache_entry(
        self,
        image_hash: str,
        model_name: str,
        cache_key: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve an exact-match interrogation cache entry if it exists."""
        if not cache_key:
            return None

        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT
                    c.cache_metadata,
                    c.result_json,
                    c.tags,
                    c.confidence_scores,
                    c.raw_output,
                    c.updated_at
                FROM interrogation_cache_entries c
                JOIN images img ON c.image_id = img.id
                JOIN models m ON c.model_id = m.id
                WHERE img.file_hash = ? AND m.model_name = ? AND c.cache_key = ?
                """,
                (image_hash, model_name, cache_key),
            )

            result = cursor.fetchone()
            if not result:
                return None

            try:
                payload = json.loads(result["result_json"]) if result["result_json"] else {}
            except json.JSONDecodeError:
                payload = {}
            if not isinstance(payload, dict):
                payload = {}

            payload.setdefault("tags", json.loads(result["tags"]) if result["tags"] else [])
            payload.setdefault(
                "confidence_scores",
                json.loads(result["confidence_scores"]) if result["confidence_scores"] else None,
            )
            payload.setdefault("raw_output", result["raw_output"])
            payload["cache_key"] = cache_key
            payload["cache_metadata"] = (
                json.loads(result["cache_metadata"]) if result["cache_metadata"] else {}
            )
            payload["cached_at"] = result["updated_at"]
            return payload
    
    @_retry_on_busy()
    def get_all_interrogations_for_image(self, file_hash: str) -> List[Dict]:
        """Get all interrogations for an image across all models."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT m.model_name, m.model_type, i.tags, i.confidence_scores, i.raw_output, i.interrogated_at
                FROM interrogations i
                JOIN images img ON i.image_id = img.id
                JOIN models m ON i.model_id = m.id
                WHERE img.file_hash = ?
                ORDER BY i.interrogated_at DESC
            """, (file_hash,))

            results = []
            for row in cursor.fetchall():
                results.append({
                    'model_name': row['model_name'],
                    'model_type': row['model_type'],
                    'tags': json.loads(row['tags']),
                    'confidence_scores': json.loads(row['confidence_scores']) if row['confidence_scores'] else None,
                    'raw_output': row['raw_output'],
                    'interrogated_at': row['interrogated_at']
                })
            return results

    @_retry_on_busy()
    def create_or_get_multimodal_session(
        self,
        image_id: int,
        model_id: int,
        mode: str,
        session_key: str,
    ) -> int:
        """Create or retrieve a multimodal session row."""
        if mode not in ("single", "batch"):
            raise ValueError("mode must be 'single' or 'batch'")
        if not session_key:
            raise ValueError("session_key is required")

        with self._get_connection(write=True) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT OR IGNORE INTO multimodal_sessions (image_id, model_id, mode, session_key)
                VALUES (?, ?, ?, ?)
                """,
                (image_id, model_id, mode, session_key),
            )
            cursor.execute(
                """
                UPDATE multimodal_sessions
                SET updated_at = CURRENT_TIMESTAMP
                WHERE image_id = ? AND model_id = ? AND mode = ? AND session_key = ?
                """,
                (image_id, model_id, mode, session_key),
            )
            cursor.execute(
                """
                SELECT id
                FROM multimodal_sessions
                WHERE image_id = ? AND model_id = ? AND mode = ? AND session_key = ?
                """,
                (image_id, model_id, mode, session_key),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("Failed to create multimodal session")
            return row["id"]

    @_retry_on_busy()
    def append_multimodal_turn(
        self,
        session_id: int,
        prompt_type: str,
        prompt_text: str,
        included_tables: Optional[List[Dict[str, Any]]],
        response_json: Dict[str, Any],
        tags: List[str],
        reasoning_summary: str,
        included_transcripts: Optional[List[Dict[str, Any]]] = None,
        sidecar_tags: Optional[List[str]] = None,
    ) -> int:
        """Append a turn to multimodal conversation history."""
        with self._get_connection(write=True) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO multimodal_turns (
                    session_id, turn_index, prompt_type, prompt_text,
                    included_tables_json, included_transcripts_json, sidecar_tags_json,
                    response_json, tags_json, reasoning_summary
                )
                SELECT ?, COALESCE(MAX(turn_index), -1) + 1, ?, ?, ?, ?, ?, ?, ?, ?
                FROM multimodal_turns
                WHERE session_id = ?
                """,
                (
                    session_id,
                    prompt_type,
                    prompt_text,
                    json.dumps(included_tables or [], ensure_ascii=False),
                    json.dumps(included_transcripts or [], ensure_ascii=False),
                    json.dumps(sidecar_tags or [], ensure_ascii=False),
                    json.dumps(response_json, ensure_ascii=False),
                    json.dumps(tags, ensure_ascii=False),
                    reasoning_summary,
                    session_id,
                ),
            )
            turn_id = cursor.lastrowid

            cursor.execute(
                """
                UPDATE multimodal_sessions
                SET updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (session_id,),
            )

            return turn_id

    @_retry_on_busy()
    def get_multimodal_history(
        self,
        session_id: Optional[int] = None,
        session_key: Optional[str] = None,
        model_name: Optional[str] = None,
        mode: Optional[str] = None,
        image_hash: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve multimodal turn history ordered by turn index."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            query = [
                """
                SELECT
                    s.id AS session_id,
                    s.session_key,
                    s.mode,
                    m.model_name,
                    img.file_hash,
                    t.turn_index,
                    t.prompt_type,
                    t.prompt_text,
                    t.included_tables_json,
                    t.included_transcripts_json,
                    t.sidecar_tags_json,
                    t.response_json,
                    t.tags_json,
                    t.reasoning_summary,
                    t.created_at
                FROM multimodal_sessions s
                JOIN models m ON s.model_id = m.id
                JOIN images img ON s.image_id = img.id
                LEFT JOIN multimodal_turns t ON t.session_id = s.id
                WHERE 1 = 1
                """
            ]
            params: List[Any] = []

            if session_id is not None:
                query.append("AND s.id = ?")
                params.append(session_id)
            if session_key:
                query.append("AND s.session_key = ?")
                params.append(session_key)
            if model_name:
                query.append("AND m.model_name = ?")
                params.append(model_name)
            if mode:
                query.append("AND s.mode = ?")
                params.append(mode)
            if image_hash:
                query.append("AND img.file_hash = ?")
                params.append(image_hash)

            query.append("ORDER BY s.id ASC, t.turn_index ASC")
            cursor.execute("\n".join(query), tuple(params))

            history: List[Dict[str, Any]] = []
            for row in cursor.fetchall():
                if row["turn_index"] is None:
                    continue
                history.append(
                    {
                        "session_id": row["session_id"],
                        "session_key": row["session_key"],
                        "mode": row["mode"],
                        "model_name": row["model_name"],
                        "file_hash": row["file_hash"],
                        "turn_index": row["turn_index"],
                        "prompt_type": row["prompt_type"],
                        "prompt_text": row["prompt_text"] or "",
                        "included_tables": (
                            json.loads(row["included_tables_json"])
                            if row["included_tables_json"]
                            else []
                        ),
                        "included_transcripts": (
                            json.loads(row["included_transcripts_json"])
                            if row["included_transcripts_json"]
                            else []
                        ),
                        "sidecar_tags": (
                            json.loads(row["sidecar_tags_json"])
                            if row["sidecar_tags_json"]
                            else []
                        ),
                        "response_json": json.loads(row["response_json"]),
                        "tags": json.loads(row["tags_json"]) if row["tags_json"] else [],
                        "reasoning_summary": row["reasoning_summary"] or "",
                        "created_at": row["created_at"],
                    }
                )
            return history

    @_retry_on_busy()
    def clear_multimodal_session(
        self,
        session_id: Optional[int] = None,
        session_key: Optional[str] = None,
        model_name: Optional[str] = None,
        mode: Optional[str] = None,
        image_hash: Optional[str] = None,
    ) -> int:
        """Delete multimodal session(s) and associated turns."""
        if session_id is None and not session_key:
            raise ValueError("Either session_id or session_key must be provided")

        with self._get_connection(write=True) as conn:
            cursor = conn.cursor()

            query = [
                """
                SELECT s.id
                FROM multimodal_sessions s
                JOIN models m ON s.model_id = m.id
                JOIN images img ON s.image_id = img.id
                WHERE 1 = 1
                """
            ]
            params: List[Any] = []

            if session_id is not None:
                query.append("AND s.id = ?")
                params.append(session_id)
            if session_key:
                query.append("AND s.session_key = ?")
                params.append(session_key)
            if model_name:
                query.append("AND m.model_name = ?")
                params.append(model_name)
            if mode:
                query.append("AND s.mode = ?")
                params.append(mode)
            if image_hash:
                query.append("AND img.file_hash = ?")
                params.append(image_hash)

            cursor.execute("\n".join(query), tuple(params))
            session_ids = [row["id"] for row in cursor.fetchall()]
            if not session_ids:
                return 0

            placeholders = ",".join("?" for _ in session_ids)
            cursor.execute(
                f"DELETE FROM multimodal_turns WHERE session_id IN ({placeholders})",
                tuple(session_ids),
            )
            cursor.execute(
                f"DELETE FROM multimodal_sessions WHERE id IN ({placeholders})",
                tuple(session_ids),
            )
            return len(session_ids)
    
    @_retry_on_busy()
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as count FROM images")
            total_images = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) as count FROM interrogations")
            total_interrogations = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(DISTINCT model_id) as count FROM interrogations")
            unique_models = cursor.fetchone()['count']

            return {
                'total_images': total_images,
                'total_interrogations': total_interrogations,
                'unique_models_used': unique_models
            }

    @_retry_on_busy(max_attempts=3)
    def vacuum(self):
        """
        Optimize the database by reclaiming unused space and reorganizing data.
        This rebuilds the database file, removing deleted data and defragmenting.
        """
        return self._vacuum_impl()

    def _vacuum_impl(self) -> None:
        """Shared implementation used by calls, retries, and queue replay."""
        with self._get_connection() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")

    def set_local_mode(self, enabled: bool):
        """
        Enable or disable local database mode.

        Args:
            enabled: If True, use local databases in image directories.
                    If False, use global database.

        The choice is saved and restored the next time the database opens.
        """
        self.use_local_db = bool(enabled)
        settings = self._load_settings()
        settings["use_local_db"] = self.use_local_db
        try:
            write_json_atomic(self._settings_path, settings)
        except OSError as exc:
            print(f"Error saving database settings: {exc}")

    def _load_settings(self) -> Dict[str, Any]:
        """Saved database settings, or {} when there are none yet."""
        try:
            data = json.loads(self._settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def switch_to_directory(self, directory: str):
        """
        Switch to using a database in the specified directory (local mode)
        or return to global database (global mode).

        Args:
            directory: Path to the image directory
        """
        self._directory_root = Path(os.path.abspath(directory))
        self._local_db_presence = {}

        if self.use_local_db:
            # Use local database in image directory
            local_db_path = Path(directory) / LOCAL_DB_NAME
        else:
            # Use global database
            local_db_path = self._global_db_path

        # Only switch if it's a different database
        self._switch_database_path(local_db_path)

    def _switch_database_path(self, target: Path) -> None:
        """Switch paths only after the target schema and queue are ready."""
        if target == self.db_path:
            return
        previous_queue = getattr(self, "_operation_queue", None)

        def initialize_target() -> None:
            self._ensure_schema()
            self._init_operation_queue()
            self._process_pending_queue()

        try:
            self._connections.switch_atomically(target, initialize_target)
        except Exception:
            self._operation_queue = previous_queue
            raise

    def get_db_location(self) -> str:
        """Get the current database file location."""
        return str(self.db_path.absolute())

    # === Queue Management Methods ===

    def _init_operation_queue(self) -> None:
        """Initialize the operation queue."""
        queue_path = self.db_path.parent / f"{self.db_path.name}.queue.json"
        self._operation_queue = DatabaseOperationQueue(
            queue_path=queue_path,
            database_path=self.db_path,
            legacy_queue_path=self.db_path.parent / "db_queue.json",
        )

    def can_queue_operation(self, operation: str) -> bool:
        """Return whether an operation is eligible for durable queueing."""
        return bool(self._operation_queue and self._operation_queue.is_queueable(operation))

    def set_busy_callback(self, callback: Optional[Callable[[str, Dict, int], str]]) -> None:
        """
        Set callback for when database is busy after silent retries fail.

        The callback receives:
            - operation: str - The method name being attempted
            - params: Dict - The parameters for the method
            - retry_count: int - Number of retries attempted

        The callback should return one of:
            - "retry": Continue retrying the operation
            - "queue": Add operation to queue (if queueable)
            - "abort": Raise DatabaseBusyError

        Args:
            callback: The callback function, or None to disable.
        """
        self._busy_callback = callback

    def _handle_busy_with_callback(
        self,
        operation: str,
        params: Dict,
        retry_count: int,
        error: Exception,
        func: Optional[Callable] = None
    ) -> Optional[Any]:
        """
        Handle database busy state by invoking the callback.

        Returns:
            The result if operation eventually succeeded, or None if queued/aborted.
        """
        max_extended_retries = 100  # ~15 minutes with exponential backoff capped
        base_delay = 0.5

        current_retry = retry_count

        while current_retry < max_extended_retries:
            # Call the UI callback
            response = self._busy_callback(operation, params, current_retry)

            if response == "retry":
                # Try the operation again
                current_retry += 1
                try:
                    # Re-execute the original (undecorated) operation directly.
                    if func is not None:
                        return func(self, **params)
                    # Legacy fallback: use the decorator-free *_direct variant.
                    method = getattr(self, f"_{operation}_direct", None)
                    if method:
                        return method(**params)
                    else:
                        # Fall through to retry
                        delay = min(base_delay * (1.5 ** (current_retry - retry_count)), 5.0)
                        time.sleep(delay)
                except sqlite3.OperationalError as e:
                    if not _is_busy_error(e):
                        raise
                    # Continue the retry loop
                    delay = min(base_delay * (1.5 ** (current_retry - retry_count)), 5.0)
                    time.sleep(delay)

            elif response == "queue":
                # Check if operation is queueable
                if self.can_queue_operation(operation):
                    op_id = self._operation_queue.add_operation(operation, params)
                    raise DatabaseQueuedError(op_id, f"Operation {operation} queued as {op_id}")
                else:
                    # Not queueable, keep retrying
                    continue

            elif response == "abort":
                raise DatabaseBusyError(f"Database operation {operation} aborted by user")

            else:
                # Unknown response, treat as abort
                raise DatabaseBusyError(f"Database operation {operation} aborted")

        # Max retries exceeded
        raise DatabaseBusyError(
            f"Database operation {operation} failed after {max_extended_retries} attempts"
        )

    def get_queue_status(self) -> Dict:
        """
        Get the status of the operation queue.

        Returns:
            Dictionary with:
                - pending_count: int
                - operations: List of operation details
                - near_capacity: bool
                - at_capacity: bool
        """
        if self._operation_queue:
            return self._operation_queue.get_status()
        return {
            'pending_count': 0,
            'operations': [],
            'near_capacity': False,
            'at_capacity': False
        }

    def _process_pending_queue(self) -> None:
        """
        Attempt to process any pending queued operations.

        Called automatically on startup. Silently skips if database is still busy.
        """
        if not self._operation_queue:
            return

        pending = self._operation_queue.get_pending_operations()
        if not pending:
            return

        # Try to process each operation (silently, without UI)
        for op in pending:
            try:
                self.process_queued_operation(op.id)
            except sqlite3.OperationalError as e:
                if _is_busy_error(e):
                    # Database still busy, stop trying
                    break
            except Exception:
                continue

    def process_queued_operations(self) -> Tuple[int, int]:
        """
        Manually process all queued operations.

        Returns:
            Tuple of (success_count, failed_count)
        """
        if not self._operation_queue:
            return (0, 0)

        pending = self._operation_queue.get_pending_operations()
        success_count = 0
        failed_count = 0

        for op in pending:
            try:
                if self.process_queued_operation(op.id):
                    success_count += 1
                else:
                    failed_count += 1
            except Exception:
                failed_count += 1

        return (success_count, failed_count)

    def process_queued_operation(self, operation_id: str) -> bool:
        """Execute and durably remove one queued operation by ID.

        Returns False if the ID is no longer pending. Execution and queue
        persistence errors are intentionally propagated to the caller.
        """
        if not self._operation_queue:
            return False
        operation = next(
            (
                item
                for item in self._operation_queue.get_pending_operations()
                if item.id == operation_id
            ),
            None,
        )
        if operation is None:
            return False
        try:
            self._execute_queued_operation(
                operation.id, operation.operation, operation.params
            )
            if not self._operation_queue.mark_completed(operation.id):
                raise RuntimeError(
                    f"Queued operation disappeared before completion: {operation.id}"
                )
        except Exception as error:
            try:
                self._operation_queue.mark_failed(operation.id, str(error))
            except Exception as persistence_error:
                raise persistence_error from error
            raise
        return True

    def _execute_queued_operation(
        self,
        op_id: str,
        operation: str,
        params: Dict
    ) -> Any:
        """Execute a single queued operation."""
        if operation == 'save_interrogation':
            return self._save_interrogation_impl(**params)
        elif operation == 'vacuum':
            return self._vacuum_impl()
        else:
            raise ValueError(f"Unknown queued operation: {operation}")

    def _save_interrogation_direct(
        self,
        image_id: int,
        model_id: int,
        tags: List[str],
        confidence_scores: Optional[Dict[str, float]] = None,
        raw_output: Optional[str] = None,
        result_version: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Direct save_interrogation without retry decorator (for queue processing)."""
        return self._save_interrogation_impl(
            image_id, model_id, tags, confidence_scores, raw_output, result_version, config
        )

    def _vacuum_direct(self) -> None:
        """Direct vacuum without retry decorator (for queue processing)."""
        return self._vacuum_impl()

    def clear_operation_queue(self) -> int:
        """
        Clear all pending operations from the queue.

        Returns:
            Number of operations that were cleared.
        """
        if self._operation_queue:
            return self._operation_queue.clear_queue()
        return 0

    def close(self):
        """Close database connection.

        Note: This is now a no-op since connections are managed per-operation.
        Kept for backwards compatibility with existing code that calls close().
        """
        pass
