"""Database management for image interrogations with SQLite."""

import sqlite3
import json
import threading
import functools
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Optional


def _retry_on_busy(max_attempts: int = 5, base_delay: float = 0.1):
    """Decorator to retry database operations on SQLITE_BUSY errors."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        time.sleep(delay)
                    else:
                        raise
        return wrapper
    return decorator


class InterrogationDatabase:
    """Manages interrogation cache and results using SQLite."""

    def __init__(self, db_path: str = "interrogations.db"):
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self.use_local_db = False  # Global by default
        self._ensure_schema()

    @contextmanager
    def _get_connection(self):
        """Context manager for thread-safe database connections.

        Opens a fresh connection for each operation with:
        - 30 second timeout for lock acquisition
        - WAL mode for improved concurrent read/write
        - Thread lock for intra-process synchronization
        """
        with self._lock:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,  # Wait up to 30s for locks
                isolation_level=None  # Autocommit mode
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            try:
                yield conn
            finally:
                conn.close()

    def _ensure_schema(self):
        """Create database schema if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Images table with hash-based deduplication
            cursor.execute("""
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
            """)

            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT NOT NULL UNIQUE,
                    model_type TEXT NOT NULL,
                    version TEXT,
                    config TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Interrogations table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interrogations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_id INTEGER NOT NULL,
                    model_id INTEGER NOT NULL,
                    tags TEXT NOT NULL,
                    confidence_scores TEXT,
                    raw_output TEXT,
                    interrogated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (image_id) REFERENCES images(id),
                    FOREIGN KEY (model_id) REFERENCES models(id),
                    UNIQUE(image_id, model_id)
                )
            """)

            # Create indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_file_hash ON images(file_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_interrogations ON interrogations(image_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_model_interrogations ON interrogations(model_id)")
    
    @_retry_on_busy()
    def register_image(self, file_path: str, file_hash: str,
                      width: int, height: int, file_size: int) -> int:
        """Register an image and return its ID.

        Uses INSERT OR IGNORE + UPDATE pattern to avoid race conditions.
        """
        with self._get_connection() as conn:
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
        with self._get_connection() as conn:
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
                          raw_output: Optional[str] = None):
        """Save or update interrogation results."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            tags_json = json.dumps(tags)
            scores_json = json.dumps(confidence_scores) if confidence_scores else None

            cursor.execute("""
                INSERT INTO interrogations (image_id, model_id, tags, confidence_scores, raw_output)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(image_id, model_id)
                DO UPDATE SET
                    tags = excluded.tags,
                    confidence_scores = excluded.confidence_scores,
                    raw_output = excluded.raw_output,
                    interrogated_at = CURRENT_TIMESTAMP
            """, (image_id, model_id, tags_json, scores_json, raw_output))
    
    @_retry_on_busy()
    def get_interrogation(self, image_hash: str, model_name: str) -> Optional[Dict]:
        """Retrieve cached interrogation if it exists."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT i.tags, i.confidence_scores, i.raw_output, i.interrogated_at
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
                    'interrogated_at': result['interrogated_at']
                }
            return None
    
    @_retry_on_busy()
    def get_all_interrogations_for_image(self, file_hash: str) -> List[Dict]:
        """Get all interrogations for an image across all models."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                SELECT m.model_name, m.model_type, i.tags, i.confidence_scores, i.interrogated_at
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
                    'interrogated_at': row['interrogated_at']
                })
            return results
    
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
        with self._get_connection() as conn:
            # WAL mode requires checkpoint before vacuum
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")

    def set_local_mode(self, enabled: bool):
        """
        Enable or disable local database mode.

        Args:
            enabled: If True, use local databases in image directories.
                    If False, use global database.
        """
        self.use_local_db = enabled

    def switch_to_directory(self, directory: str):
        """
        Switch to using a database in the specified directory (local mode)
        or return to global database (global mode).

        Args:
            directory: Path to the image directory
        """
        if self.use_local_db:
            # Use local database in image directory
            local_db_path = Path(directory) / ".interrogations.db"
        else:
            # Use global database
            local_db_path = Path("interrogations.db")

        # Only switch if it's a different database
        if local_db_path != self.db_path:
            # Update path and ensure schema exists
            self.db_path = local_db_path
            self._ensure_schema()

    def get_db_location(self) -> str:
        """Get the current database file location."""
        return str(self.db_path.absolute())

    def close(self):
        """Close database connection.

        Note: This is now a no-op since connections are managed per-operation.
        Kept for backwards compatibility with existing code that calls close().
        """
        pass
