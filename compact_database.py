#!/usr/bin/env python3
"""Reclaim space taken by debug payloads stored for cleanly-parsed responses.

`LlamaCppInterrogator` used to attach `_debug_raw_response` (up to 20 KB of raw
model output) to every response it parsed. The transcript view only ever
surfaces that text for parses that needed repair, retry, or a fallback, so on a
clean `primary_json` parse the payload is written but never read. It is stored
in three places -- `multimodal_turns.response_json`, `interrogations.raw_output`
and `interrogation_cache_entries` -- and inside cache rows it appears up to
three times per row.

The interrogator no longer writes it for clean parses. This script removes it
from rows already on disk. Payloads for non-clean parses are left untouched.

Cache keys are derived from the prompt, config, and context -- never from the
result -- so rewriting stored results does not invalidate any cache entry.

Usage:
    python compact_database.py                     # report only, no writes
    python compact_database.py --apply             # rewrite rows
    python compact_database.py --apply --vacuum    # rewrite, then reclaim file space
"""

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

DEBUG_KEY = "_debug_raw_response"
CLEAN_PARSE_MODE = "primary_json"

# (table, primary key column, JSON column) -- every column here holds either a
# JSON document or a JSON-encoded string.
TARGETS = [
    ("multimodal_turns", "id", "response_json"),
    ("interrogations", "id", "raw_output"),
    ("interrogation_cache_entries", "id", "result_json"),
    ("interrogation_cache_entries", "id", "raw_output"),
]


def _strip_debug(node: Any) -> bool:
    """Drop DEBUG_KEY from any object whose parse was clean. Returns True if changed.

    Recurses into nested containers and into strings that are themselves JSON
    documents, since `raw_output` is stored as an encoded string inside
    `result_json`.
    """
    changed = False

    if isinstance(node, dict):
        if node.get("_parse_mode") == CLEAN_PARSE_MODE and DEBUG_KEY in node:
            del node[DEBUG_KEY]
            changed = True

        for key, value in list(node.items()):
            if isinstance(value, (dict, list)):
                changed |= _strip_debug(value)
            elif isinstance(value, str):
                inner, inner_changed = _strip_nested_json(value)
                if inner_changed:
                    node[key] = inner
                    changed = True

    elif isinstance(node, list):
        for item in node:
            changed |= _strip_debug(item)

    return changed


def _strip_nested_json(text: str) -> Tuple[str, bool]:
    """Strip a JSON-encoded string in place, returning (text, changed)."""
    stripped = text.lstrip()
    if not stripped.startswith(("{", "[")) or DEBUG_KEY not in text:
        return text, False
    try:
        document = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return text, False
    if not _strip_debug(document):
        return text, False
    return json.dumps(document, ensure_ascii=False), True


def _rewrite(value: str) -> Optional[str]:
    """Return the rewritten column value, or None when nothing changed."""
    if not value or DEBUG_KEY not in value:
        return None
    stripped = value.lstrip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        document = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None
    if not _strip_debug(document):
        return None
    return json.dumps(document, ensure_ascii=False)


def compact(db_path: Path, apply: bool, vacuum: bool) -> int:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    total_rows = 0
    total_saved = 0

    for table, pk, column in TARGETS:
        rows_changed = 0
        saved = 0
        pending = []

        cursor = conn.execute(
            f"SELECT {pk} AS pk, {column} AS payload FROM {table} "
            f"WHERE {column} LIKE '%{DEBUG_KEY}%'"
        )
        for row in cursor:
            rewritten = _rewrite(row["payload"])
            if rewritten is None:
                continue
            rows_changed += 1
            saved += len(row["payload"]) - len(rewritten)
            if apply:
                pending.append((rewritten, row["pk"]))
                if len(pending) >= 2000:
                    conn.executemany(
                        f"UPDATE {table} SET {column} = ? WHERE {pk} = ?", pending
                    )
                    conn.commit()
                    pending.clear()

        if apply and pending:
            conn.executemany(f"UPDATE {table} SET {column} = ? WHERE {pk} = ?", pending)
            conn.commit()

        total_rows += rows_changed
        total_saved += saved
        print(f"  {table}.{column:14s} {rows_changed:8,d} rows  {saved / 1048576:8.1f} MB")

    verb = "removed" if apply else "would remove"
    print(f"\n{verb} {total_saved / 1048576:.1f} MB across {total_rows:,} rows")

    if apply and vacuum:
        free = shutil.disk_usage(db_path.parent).free
        size = db_path.stat().st_size
        if free < size * 1.2:
            print(
                f"\nSkipping VACUUM: needs ~{size / 1073741824:.1f} GB free "
                f"(a full rebuild), only {free / 1073741824:.1f} GB available."
            )
        else:
            print("\nVACUUM (rebuilding file)...")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.execute("VACUUM")
            conn.execute("ANALYZE")
            print(f"file is now {db_path.stat().st_size / 1073741824:.2f} GB")
    elif apply:
        print("\nSpace is freed inside the file but not returned to the OS.")
        print("Re-run with --vacuum to shrink the file itself.")

    conn.close()
    return total_saved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", nargs="?", default="interrogations.db")
    parser.add_argument("--apply", action="store_true", help="write changes (default: report only)")
    parser.add_argument("--vacuum", action="store_true", help="rebuild the file afterwards")
    args = parser.parse_args()

    db_path = Path(args.database)
    if not db_path.exists():
        print(f"No such database: {db_path}", file=sys.stderr)
        return 1

    print(f"{db_path} ({db_path.stat().st_size / 1073741824:.2f} GB)")
    print("Reporting only -- pass --apply to write.\n" if not args.apply else "")
    compact(db_path, apply=args.apply, vacuum=args.vacuum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
