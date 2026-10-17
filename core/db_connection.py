"""SQLite connection lifecycle helpers for the interrogation database."""

from contextlib import contextmanager
from pathlib import Path
import sqlite3
import threading
from typing import Callable, Iterator


class SQLiteConnectionManager:
    """Own the active database path and serialize connections and path switches."""

    def __init__(self, db_path: Path):
        self._path = Path(db_path)
        self._lock = threading.RLock()
        self._wal_initialized_paths: set[str] = set()

    @property
    def path(self) -> Path:
        with self._lock:
            return self._path

    @staticmethod
    def _path_key(path: Path) -> str:
        return str(path.resolve(strict=False))

    @contextmanager
    def connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """Yield a configured connection, optionally in an immediate transaction."""
        with self._lock:
            conn = None
            try:
                path = self._path
                conn = sqlite3.connect(
                    path,
                    timeout=30.0,
                    isolation_level=None,
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys=ON")

                path_key = self._path_key(path)
                if path_key not in self._wal_initialized_paths:
                    conn.execute("PRAGMA journal_mode=WAL")
                    self._wal_initialized_paths.add(path_key)

                if write:
                    conn.execute("BEGIN IMMEDIATE")

                yield conn

                if write and conn.in_transaction:
                    conn.commit()
            except Exception:
                if conn is not None and conn.in_transaction:
                    conn.rollback()
                raise
            finally:
                if conn is not None:
                    conn.close()

    def switch_atomically(self, target: Path, initialize: Callable[[], None]) -> None:
        """Initialize ``target`` while all other operations are excluded.

        The previous path is restored if initialization fails, so callers never
        observe a half-initialized database switch.
        """
        target = Path(target)
        with self._lock:
            previous = self._path
            self._path = target
            try:
                initialize()
            except Exception:
                self._path = previous
                raise
