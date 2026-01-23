"""Database management for image interrogations with SQLite."""

import sqlite3
import json
import threading
import functools
import time
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple, Any

from core.db_queue import DatabaseOperationQueue


class DatabaseBusyError(Exception):
    """Raised when database is locked and user chose to abort."""
    pass


class DatabaseQueuedError(Exception):
    """Raised when operation was queued instead of executed."""

    def __init__(self, operation_id: str, message: str = "Operation queued"):
        self.operation_id = operation_id
        super().__init__(message)


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
                    if "locked" in str(e).lower() and attempt < silent_attempts - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        time.sleep(delay)
                    elif "locked" in str(e).lower():
                        # Silent retries exhausted - invoke callback if available
                        if self._busy_callback is not None:
                            params = _build_params_dict(func, args, kwargs)
                            result = self._handle_busy_with_callback(
                                operation_name, params, silent_attempts, e
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
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self.use_local_db = False  # Global by default

        # Queue for handling database-busy scenarios
        self._operation_queue: Optional[DatabaseOperationQueue] = None
        self._busy_callback: Optional[Callable[[str, Dict, int], str]] = None

        self._ensure_schema()
        self._init_operation_queue()
        self._process_pending_queue()

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

    # === Queue Management Methods ===

    def _init_operation_queue(self) -> None:
        """Initialize the operation queue."""
        queue_path = self.db_path.parent / "db_queue.json"
        self._operation_queue = DatabaseOperationQueue(
            queue_path=queue_path,
            database_path=self.db_path
        )

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
        error: Exception
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
                    # Re-execute the operation
                    method = getattr(self, f"_{operation}_impl", None)
                    if method:
                        return method(**params)
                    else:
                        # Fall through to retry
                        delay = min(base_delay * (1.5 ** (current_retry - retry_count)), 5.0)
                        time.sleep(delay)
                except sqlite3.OperationalError as e:
                    if "locked" not in str(e).lower():
                        raise
                    # Continue the retry loop
                    delay = min(base_delay * (1.5 ** (current_retry - retry_count)), 5.0)
                    time.sleep(delay)

            elif response == "queue":
                # Check if operation is queueable
                if self._operation_queue and self._operation_queue.is_queueable(operation):
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
                self._execute_queued_operation(op.id, op.operation, op.params)
                self._operation_queue.mark_completed(op.id)
            except sqlite3.OperationalError as e:
                if "locked" in str(e).lower():
                    # Database still busy, stop trying
                    self._operation_queue.mark_failed(op.id, str(e))
                    break
                else:
                    self._operation_queue.mark_failed(op.id, str(e))
            except Exception as e:
                self._operation_queue.mark_failed(op.id, str(e))

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
                self._execute_queued_operation(op.id, op.operation, op.params)
                self._operation_queue.mark_completed(op.id)
                success_count += 1
            except Exception as e:
                self._operation_queue.mark_failed(op.id, str(e))
                failed_count += 1

        return (success_count, failed_count)

    def _execute_queued_operation(
        self,
        op_id: str,
        operation: str,
        params: Dict
    ) -> Any:
        """Execute a single queued operation."""
        if operation == 'save_interrogation':
            return self._save_interrogation_direct(**params)
        elif operation == 'vacuum':
            return self._vacuum_direct()
        else:
            raise ValueError(f"Unknown queued operation: {operation}")

    def _save_interrogation_direct(
        self,
        image_id: int,
        model_id: int,
        tags: List[str],
        confidence_scores: Optional[Dict[str, float]] = None,
        raw_output: Optional[str] = None
    ) -> None:
        """Direct save_interrogation without retry decorator (for queue processing)."""
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

    def _vacuum_direct(self) -> None:
        """Direct vacuum without retry decorator (for queue processing)."""
        with self._get_connection() as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")

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
