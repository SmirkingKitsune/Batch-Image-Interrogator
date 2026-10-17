"""Idempotent SQLite schema setup for interrogation databases."""

import sqlite3


SCHEMA_VERSION = 1


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create or additively upgrade the database schema in one transaction."""
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            width INTEGER,
            height INTEGER,
            file_size INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL UNIQUE,
            model_type TEXT NOT NULL,
            version TEXT,
            config TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS interrogations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            model_id INTEGER NOT NULL,
            tags TEXT NOT NULL,
            confidence_scores TEXT,
            raw_output TEXT,
            result_version INTEGER,
            interrogated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            config TEXT,
            FOREIGN KEY (image_id) REFERENCES images(id),
            FOREIGN KEY (model_id) REFERENCES models(id),
            UNIQUE(image_id, model_id)
        )
        """
    )
    cursor.execute("PRAGMA table_info(interrogations)")
    interrogation_columns = {row["name"] for row in cursor.fetchall()}
    if "result_version" not in interrogation_columns:
        cursor.execute("ALTER TABLE interrogations ADD COLUMN result_version INTEGER")
    # Settings the row was produced with (the interrogator's result_signature),
    # which decide whether the cache may reuse it.
    if "config" not in interrogation_columns:
        cursor.execute("ALTER TABLE interrogations ADD COLUMN config TEXT")
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS interrogation_cache_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            model_id INTEGER NOT NULL,
            cache_key TEXT NOT NULL,
            cache_metadata TEXT NOT NULL,
            result_json TEXT NOT NULL,
            tags TEXT NOT NULL,
            confidence_scores TEXT,
            raw_output TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES images(id),
            FOREIGN KEY (model_id) REFERENCES models(id),
            UNIQUE(image_id, model_id, cache_key)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS multimodal_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            model_id INTEGER NOT NULL,
            mode TEXT NOT NULL CHECK(mode IN ('single', 'batch')),
            session_key TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (image_id) REFERENCES images(id),
            FOREIGN KEY (model_id) REFERENCES models(id),
            UNIQUE(image_id, model_id, mode, session_key)
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS multimodal_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            turn_index INTEGER NOT NULL,
            prompt_type TEXT NOT NULL,
            prompt_text TEXT,
            included_tables_json TEXT,
            included_transcripts_json TEXT,
            sidecar_tags_json TEXT,
            response_json TEXT NOT NULL,
            tags_json TEXT NOT NULL,
            reasoning_summary TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES multimodal_sessions(id),
            UNIQUE(session_id, turn_index)
        )
        """
    )

    cursor.execute("PRAGMA table_info(multimodal_turns)")
    turn_columns = {row["name"] for row in cursor.fetchall()}
    if "included_transcripts_json" not in turn_columns:
        cursor.execute("ALTER TABLE multimodal_turns ADD COLUMN included_transcripts_json TEXT")
    if "sidecar_tags_json" not in turn_columns:
        cursor.execute("ALTER TABLE multimodal_turns ADD COLUMN sidecar_tags_json TEXT")

    for statement in (
        "CREATE INDEX IF NOT EXISTS idx_file_hash ON images(file_hash)",
        "CREATE INDEX IF NOT EXISTS idx_image_interrogations ON interrogations(image_id)",
        "CREATE INDEX IF NOT EXISTS idx_model_interrogations ON interrogations(model_id)",
        "CREATE INDEX IF NOT EXISTS idx_cache_entries_image ON interrogation_cache_entries(image_id)",
        "CREATE INDEX IF NOT EXISTS idx_cache_entries_model ON interrogation_cache_entries(model_id)",
        "CREATE INDEX IF NOT EXISTS idx_cache_entries_key ON interrogation_cache_entries(cache_key)",
        "CREATE INDEX IF NOT EXISTS idx_mm_sessions_image ON multimodal_sessions(image_id)",
        "CREATE INDEX IF NOT EXISTS idx_mm_turns_session ON multimodal_turns(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_mm_turns_created_at ON multimodal_turns(created_at)",
    ):
        cursor.execute(statement)

    # No query filters sessions by model alone, and model_id is very
    # low-cardinality, so this index could only ever lure the planner into
    # scanning every session for a single-model database.
    cursor.execute("DROP INDEX IF EXISTS idx_mm_sessions_model")

    _ensure_planner_statistics(cursor)

    cursor.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def _ensure_planner_statistics(cursor: sqlite3.Cursor) -> None:
    """Populate query planner statistics when they are missing.

    Without sqlite_stat1 the planner has no cardinality information and picks
    indexes by declaration order, which makes multimodal history lookups scan
    every session instead of seeking by image. ANALYZE skips empty tables, so
    key off a table that matters rather than the mere existence of
    sqlite_stat1 -- otherwise a database analyzed while empty would never pick
    up statistics as it grows.
    """
    cursor.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sqlite_stat1'"
    )
    if cursor.fetchone() is not None:
        cursor.execute("SELECT 1 FROM sqlite_stat1 WHERE tbl = 'multimodal_sessions'")
        if cursor.fetchone() is not None:
            return

    cursor.execute("ANALYZE")
