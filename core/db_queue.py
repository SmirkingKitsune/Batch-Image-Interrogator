"""Database operation queue for handling locked database scenarios."""

import json
import threading
import uuid
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from core.atomic_write import write_json_atomic


class QueuePersistenceError(RuntimeError):
    """Raised when an in-memory queue mutation cannot be persisted durably."""


@dataclass
class QueuedOperation:
    """Represents a database operation waiting to be executed."""
    id: str                          # UUID
    operation: str                   # Method name (e.g., "save_interrogation")
    params: Dict[str, Any]           # Serialized parameters
    timestamp: str                   # ISO format timestamp
    retry_count: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueuedOperation':
        """Create from dictionary."""
        return cls(**data)


class DatabaseOperationQueue:
    """
    Manages a persistent queue of failed database operations.

    Operations that fail due to database locks are queued here and can be
    retried later when the database becomes available. The queue is persisted
    to disk to survive application restarts.
    """

    QUEUE_VERSION = 1
    MAX_QUEUE_SIZE = 1000
    WARNING_THRESHOLD = 900

    # Operations that can be safely queued (no return value needed)
    QUEUEABLE_OPERATIONS = {
        'save_interrogation',
        'vacuum',
    }

    def __init__(
        self,
        queue_path: Optional[Path] = None,
        database_path: Optional[Path] = None,
        legacy_queue_path: Optional[Path] = None,
    ):
        """
        Initialize the operation queue.

        Args:
            queue_path: Path to the queue JSON file. Defaults to db_queue.json in CWD.
            database_path: Path to the associated database file.
        """
        self.queue_path = queue_path or Path("db_queue.json")
        self.database_path = database_path
        self.legacy_queue_path = legacy_queue_path
        self.queue: List[QueuedOperation] = []
        self._lock = threading.Lock()
        if not self._adopt_legacy_queue():
            self._load_queue()

    def is_queueable(self, operation: str) -> bool:
        """Check if an operation can be safely queued."""
        return operation in self.QUEUEABLE_OPERATIONS

    def add_operation(self, operation: str, params: Dict[str, Any]) -> str:
        """
        Add an operation to the queue.

        Args:
            operation: The database method name.
            params: The parameters for the method.

        Returns:
            The UUID of the queued operation.

        Raises:
            ValueError: If queue is at capacity.
        """
        with self._lock:
            if len(self.queue) >= self.MAX_QUEUE_SIZE:
                raise ValueError(f"Queue is at capacity ({self.MAX_QUEUE_SIZE} operations)")

            op_id = str(uuid.uuid4())
            queued_op = QueuedOperation(
                id=op_id,
                operation=operation,
                params=params,
                timestamp=datetime.now().isoformat(),
                retry_count=0,
                last_error=None
            )
            snapshot = self._snapshot()
            self.queue.append(queued_op)
            self._persist_or_rollback(snapshot)
            return op_id

    def remove_operation(self, op_id: str) -> bool:
        """
        Remove an operation from the queue.

        Args:
            op_id: The UUID of the operation to remove.

        Returns:
            True if the operation was found and removed, False otherwise.
        """
        with self._lock:
            for i, op in enumerate(self.queue):
                if op.id == op_id:
                    snapshot = self._snapshot()
                    del self.queue[i]
                    self._persist_or_rollback(snapshot)
                    return True
            return False

    def get_pending_operations(self) -> List[QueuedOperation]:
        """
        Get all pending operations in the queue.

        Returns:
            A copy of the list of queued operations.
        """
        with self._lock:
            return [QueuedOperation.from_dict(op.to_dict()) for op in self.queue]

    def get_pending_count(self) -> int:
        """Get the number of pending operations."""
        with self._lock:
            return len(self.queue)

    def mark_completed(self, op_id: str) -> bool:
        """
        Mark an operation as completed (removes it from queue).

        Args:
            op_id: The UUID of the completed operation.

        Returns:
            True if the operation was found and removed, False otherwise.
        """
        return self.remove_operation(op_id)

    def mark_failed(self, op_id: str, error: str) -> bool:
        """
        Mark an operation as failed, incrementing its retry count.

        Args:
            op_id: The UUID of the failed operation.
            error: The error message.

        Returns:
            True if the operation was found and updated, False otherwise.
        """
        with self._lock:
            for op in self.queue:
                if op.id == op_id:
                    snapshot = self._snapshot()
                    op.retry_count += 1
                    op.last_error = error
                    self._persist_or_rollback(snapshot)
                    return True
            return False

    def clear_queue(self) -> int:
        """
        Clear all operations from the queue.

        Returns:
            The number of operations that were cleared.
        """
        with self._lock:
            count = len(self.queue)
            snapshot = self._snapshot()
            self.queue.clear()
            self._persist_or_rollback(snapshot)
            return count

    def is_near_capacity(self) -> bool:
        """Check if queue is approaching capacity (90%)."""
        with self._lock:
            return len(self.queue) >= self.WARNING_THRESHOLD

    def is_at_capacity(self) -> bool:
        """Check if queue is at capacity."""
        with self._lock:
            return len(self.queue) >= self.MAX_QUEUE_SIZE

    def get_status(self) -> Dict[str, Any]:
        """
        Get queue status information.

        Returns:
            Dictionary with queue status info.
        """
        with self._lock:
            operations = []
            for op in self.queue:
                operations.append({
                    'id': op.id,
                    'operation': op.operation,
                    'timestamp': op.timestamp,
                    'retry_count': op.retry_count,
                    'last_error': op.last_error,
                    # Include a summary of params for display
                    'summary': self._get_operation_summary(op)
                })

            return {
                'pending_count': len(self.queue),
                'max_size': self.MAX_QUEUE_SIZE,
                'near_capacity': len(self.queue) >= self.WARNING_THRESHOLD,
                'at_capacity': len(self.queue) >= self.MAX_QUEUE_SIZE,
                'database_path': str(self.database_path) if self.database_path else None,
                'operations': operations
            }

    def _get_operation_summary(self, op: QueuedOperation) -> str:
        """Generate a human-readable summary of an operation."""
        if op.operation == 'save_interrogation':
            image_id = op.params.get('image_id', '?')
            model_id = op.params.get('model_id', '?')
            tag_count = len(op.params.get('tags', []))
            return f"Save {tag_count} tags for image #{image_id} with model #{model_id}"
        elif op.operation == 'vacuum':
            return "Optimize database"
        else:
            return f"{op.operation}({len(op.params)} params)"

    def _snapshot(self) -> List[QueuedOperation]:
        """Return a deep-enough snapshot for rolling back queue mutations."""
        return [QueuedOperation.from_dict(op.to_dict()) for op in self.queue]

    def _persist_or_rollback(self, snapshot: List[QueuedOperation]) -> None:
        if self._save_queue():
            return
        self.queue = snapshot
        raise QueuePersistenceError(
            f"Failed to persist queued operations to {self.queue_path}"
        )

    @staticmethod
    def _same_database_path(first: str, second: Path) -> bool:
        try:
            return Path(first).resolve(strict=False) == second.resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            return False

    def _adopt_legacy_queue(self) -> bool:
        """Migrate a matching legacy ``db_queue.json`` to the scoped name."""
        legacy_path = self.legacy_queue_path
        if self.queue_path.exists() or legacy_path is None or not legacy_path.exists():
            return False
        if self.database_path is None:
            return False

        try:
            with open(legacy_path, 'r', encoding='utf-8') as handle:
                data = json.load(handle)
            stored_path = data.get('database_path')
            if not stored_path or not self._same_database_path(stored_path, self.database_path):
                return False
            if data.get('version', 0) != self.QUEUE_VERSION:
                return False

            self.queue = [
                QueuedOperation.from_dict(op_data)
                for op_data in data.get('queue', [])
            ]
            if not self._save_queue():
                self.queue = []
                raise QueuePersistenceError(
                    f"Failed to migrate legacy queue to {self.queue_path}"
                )

            migrated_path = legacy_path.with_suffix(
                f'.migrated.{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            try:
                legacy_path.replace(migrated_path)
            except OSError:
                # The scoped queue now takes precedence, so leaving the legacy
                # file in place cannot replay it in this version.
                pass
            return True
        except QueuePersistenceError:
            raise
        except Exception:
            self.queue = []
            return False

    def _load_queue(self) -> None:
        """Load queue from disk."""
        if not self.queue_path.exists():
            self.queue = []
            return

        try:
            with open(self.queue_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate version
            version = data.get('version', 0)
            if version != self.QUEUE_VERSION:
                # Future: handle version migrations
                self._backup_and_reset("version mismatch")
                return

            # Validate database path matches
            stored_db_path = data.get('database_path')
            if self.database_path and stored_db_path:
                if Path(stored_db_path).resolve() != self.database_path.resolve():
                    # Database path mismatch - keep the queue but log warning
                    # The UI will prompt the user about this
                    pass

            # Load operations
            self.queue = [
                QueuedOperation.from_dict(op_data)
                for op_data in data.get('queue', [])
            ]

        except json.JSONDecodeError as e:
            self._backup_and_reset(f"JSON parse error: {e}")
        except Exception as e:
            self._backup_and_reset(f"Load error: {e}")

    def _save_queue(self) -> bool:
        """Save queue to disk atomically.

        Returns True on success, False if the write failed (so callers that
        depend on durability, e.g. add_operation, can react instead of
        silently assuming the operation was persisted).
        """
        data = {
            'version': self.QUEUE_VERSION,
            'database_path': str(self.database_path) if self.database_path else None,
            'saved_at': datetime.now().isoformat(),
            'queue': [op.to_dict() for op in self.queue]
        }

        try:
            # The shared helper writes a temp file beside the queue, fsyncs it,
            # renames it over the original, and cleans up the temp on failure.
            write_json_atomic(self.queue_path, data)
            return True
        except Exception:
            return False

    def _backup_and_reset(self, reason: str) -> None:
        """Backup corrupt queue file and start fresh."""
        if self.queue_path.exists():
            backup_path = self.queue_path.with_suffix(
                f'.backup.{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            )
            try:
                shutil.copy2(self.queue_path, backup_path)
            except Exception:
                pass  # Best effort backup

        self.queue = []
        self._save_queue()
