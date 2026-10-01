"""Finding and launching the Electron front end.

`./setup.sh --electron` installs Electron into ui_electron/node_modules. The
npm package records its binary's location in path.txt, so no Node.js is needed
at launch time: the bridge runs that binary directly.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ELECTRON_APP_DIR = PROJECT_ROOT / "ui_electron"
ELECTRON_PACKAGE_DIR = ELECTRON_APP_DIR / "node_modules" / "electron"
NOT_INSTALLED_MESSAGE = (
    "[X] --electron requested but Electron is not installed.\n"
    "    Run ./setup.sh --electron once, or drop the flag\n"
    "    to use the default PyQt6 interface."
)


def electron_binary(package_dir: Path = ELECTRON_PACKAGE_DIR) -> Optional[Path]:
    """Path to the installed Electron executable, or None."""
    try:
        relative = (package_dir / "path.txt").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not relative:
        return None
    binary = package_dir / "dist" / relative
    return binary if binary.is_file() else None


def electron_version(package_dir: Path = ELECTRON_PACKAGE_DIR) -> str:
    try:
        data = json.loads((package_dir / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str(data.get("version", ""))


def sandbox_flags(binary: Path) -> List[str]:
    """Chromium flags needed for the renderer sandbox on this host.

    Chromium's Linux sandbox needs either unprivileged user namespaces or a
    setuid-root chrome-sandbox helper. npm installs the helper without setuid,
    and Ubuntu 24.04+ restricts user namespaces through AppArmor, so on such
    hosts Electron refuses to start unless the sandbox is disabled. The window
    only ever loads this app's own pages with Node integration off, which
    limits what the sandbox would be protecting against.
    """
    if os.environ.get("II_ELECTRON_NO_SANDBOX") == "1":
        return ["--no-sandbox"]
    if not sys.platform.startswith("linux"):
        return []
    helper = binary.parent / "chrome-sandbox"
    try:
        stat = helper.stat()
        if stat.st_uid == 0 and stat.st_mode & 0o4000:
            return []  # setuid helper present; the sandbox works.
    except OSError:
        pass
    if _userns_restricted():
        return ["--no-sandbox"]
    return []


def _userns_restricted() -> bool:
    checks = {
        "/proc/sys/kernel/apparmor_restrict_unprivileged_userns": "1",
        "/proc/sys/kernel/unprivileged_userns_clone": "0",
    }
    for path, restricted_value in checks.items():
        try:
            if Path(path).read_text(encoding="utf-8").strip() == restricted_value:
                return True
        except OSError:
            continue
    return False


def spawn_electron(
    binary: Path,
    bridge_url: str,
    token: str,
    extra_args: Optional[List[str]] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    """Start the Electron window pointed at the bridge.

    The token travels through the environment, never the command line, so it
    does not show up in process listings.
    """
    env = dict(os.environ)
    env.update(extra_env or {})
    env["II_BRIDGE_URL"] = bridge_url
    env["II_BRIDGE_TOKEN"] = token
    env["II_PROJECT_ROOT"] = str(PROJECT_ROOT)
    # Electron is a GUI process; keep a stray ELECTRON_RUN_AS_NODE from
    # turning it into a bare Node.js interpreter.
    env.pop("ELECTRON_RUN_AS_NODE", None)
    argv = [str(binary), *sandbox_flags(binary), *(extra_args or []), str(ELECTRON_APP_DIR)]
    return subprocess.Popen(argv, cwd=str(ELECTRON_APP_DIR), env=env)
