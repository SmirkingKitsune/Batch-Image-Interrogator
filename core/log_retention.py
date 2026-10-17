"""Shared size bounding for the llama.cpp log families.

Two families share `cache/llama_cpp/logs/`: per-run server logs written by
`LlamaCppRuntimeManager`, and per-attempt build logs written by
`LlamaProvisioner`. They have very different sizes and lifetimes, so each keeps
its own policy — but the pruning and truncation behaviour is identical, and one
implementation avoids the two drifting apart.

Budgets are applied per family rather than to the directory as a whole. A shared
pool would let a single multi-hundred-megabyte inference log evict every build
log, which is the opposite of useful: the build log is what a failed install
tells the user to go read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO, Iterable


@dataclass(frozen=True)
class RetentionPolicy:
    """Size limits for one family of log files."""

    keep_files: int
    keep_total_bytes: int
    max_bytes: int
    tail_bytes: int

    def historical_files(self) -> int:
        """Slots for existing logs, reserving one for the run about to start."""
        return max(0, self.keep_files - 1)

    def historical_bytes(self) -> int:
        """Bytes for existing logs, reserving a full allowance for the new one.

        Without this reservation, `keep_files - 1` logs at `max_bytes` each plus
        a fresh `max_bytes` log would exceed the advertised total.
        """
        return max(0, self.keep_total_bytes - self.max_bytes)


def open_append_log(log_path: Path, encoding: str = "utf-8", errors: str = "replace") -> IO[str]:
    """Open a log so that a child process inheriting it also appends.

    On POSIX, append mode sets O_APPEND on the file description, which the
    child inherits. On Windows, O_APPEND is emulated by the opening process's C
    runtime only: the OS handle a child inherits writes at its own file
    pointer, so after truncate_log its next write lands at the old offset and
    zero-fills the gap back to the original size. A handle granted
    FILE_APPEND_DATA without FILE_WRITE_DATA makes the OS itself append every
    write, including the child's.
    """
    if os.name != "nt":
        return open(log_path, "a", encoding=encoding, errors=errors)

    import ctypes
    import msvcrt
    from ctypes import wintypes

    FILE_READ_ATTRIBUTES = 0x0080
    FILE_APPEND_DATA = 0x0004
    SYNCHRONIZE = 0x00100000
    # Share everything, so truncate_log and log pruning can still reach the file.
    FILE_SHARE_ALL = 0x1 | 0x2 | 0x4
    OPEN_ALWAYS = 4
    FILE_ATTRIBUTE_NORMAL = 0x80

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE

    handle = create_file(
        str(log_path),
        FILE_APPEND_DATA | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
        FILE_SHARE_ALL,
        None,
        OPEN_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        None,
    )
    if handle is None or handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_APPEND)
    except OSError:
        kernel32.CloseHandle(wintypes.HANDLE(handle))
        raise
    return os.fdopen(fd, "a", encoding=encoding, errors=errors)


def truncate_log(log_path: Path, tail_bytes: int) -> None:
    """Drop the head of an oversized log, keeping its recent tail.

    Safe to call on a file a live process still holds open, provided that
    process opened it in append mode: every subsequent write then lands at the
    new end of file rather than at a stale offset that would leave a sparse
    hole. Lines written during the swap can be lost, which is the deliberate
    trade for bounding a runaway log.
    """
    try:
        with log_path.open("rb") as handle:
            handle.seek(-tail_bytes, os.SEEK_END)
            tail = handle.read()
    except OSError:
        return

    # The tail almost certainly starts mid-line; drop the partial one.
    newline = tail.find(b"\n")
    if newline != -1:
        tail = tail[newline + 1:]

    marker = (
        f"# [log truncated at {datetime.now().isoformat(timespec='seconds')}; "
        f"kept trailing {len(tail)} bytes]\n"
    ).encode("utf-8")
    try:
        os.truncate(log_path, 0)
        with log_path.open("ab") as handle:
            handle.write(marker)
            handle.write(tail)
    except OSError:
        # Windows can refuse truncation while another process holds the file
        # open. Leaving the log oversized beats interrupting the writer.
        return


def prune_logs(log_dir: Path, pattern: str, policy: RetentionPolicy) -> None:
    """Bring one log family within its budget, newest first.

    Call before opening a new log, so the budget leaves room for the run about
    to start. Failures are ignored throughout: losing a diagnostic log is never
    a reason to refuse to do the actual work.
    """
    try:
        logs = sorted(
            (
                path
                for path in log_dir.glob(pattern)
                if path.is_file() and not path.is_symlink()
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return

    keep_files = policy.historical_files()
    keep_bytes = policy.historical_bytes()
    kept_bytes = 0
    budget_exhausted = False
    for index, path in enumerate(logs):
        try:
            size = path.stat().st_size
        except OSError:
            continue

        # A previous unclean shutdown may have left an uncapped log. Keep its
        # useful tail before applying the historical-log budget.
        if size > policy.max_bytes:
            truncate_log(path, policy.tail_bytes)
            try:
                size = path.stat().st_size
            except OSError:
                continue

        if not budget_exhausted and index < keep_files and kept_bytes + size <= keep_bytes:
            kept_bytes += size
            continue
        budget_exhausted = True
        try:
            path.unlink()
        except OSError:
            pass
