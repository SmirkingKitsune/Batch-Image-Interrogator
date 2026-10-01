"""Entry point for the opt-in Electron front end.

main.py detects devices first, exactly as for PyQt6, then hands over here
when `--electron` is passed. Qt is never imported on this path.
"""

from __future__ import annotations

import argparse
import secrets
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from bridge.electron import (
    ELECTRON_APP_DIR,
    NOT_INSTALLED_MESSAGE,
    electron_binary,
    spawn_electron,
)
from bridge.events import EventBus
from bridge.server import BridgeServer, StaticRoutes
from bridge.thumbnails import Thumbnailer, resolve_image_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NODE_MODULES = ELECTRON_APP_DIR / "node_modules"


def parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--electron", action="store_true")
    # Development aid: serve the UI without spawning Electron, so it can be
    # opened in a browser at the printed /auth URL.
    parser.add_argument("--bridge-only", action="store_true")
    parser.add_argument("--bridge-port", type=int, default=0)
    args, _unknown = parser.parse_known_args(argv or [])
    return args


def static_routes() -> StaticRoutes:
    routes = StaticRoutes()
    routes.add_dir("/app", ELECTRON_APP_DIR / "renderer")
    routes.add_file("/vendor/htm-preact.mjs", NODE_MODULES / "htm" / "preact" / "standalone.module.js")
    routes.add_dir("/fonts", NODE_MODULES / "@fontsource" / "jetbrains-mono" / "files")
    routes.add_file("/assets/icon.png", PROJECT_ROOT / "icon.png")
    return routes


def start_bridge(device_status: Dict[str, Any], argv: Optional[List[str]] = None) -> int:
    """Run the headless backend and the Electron window until it closes."""
    args = parse_args(argv)

    binary = None
    if not args.bridge_only:
        binary = electron_binary()
        if binary is None:
            print(NOT_INSTALLED_MESSAGE, file=sys.stderr)
            return 1

    # Imported after the Electron check so a missing install fails fast.
    from bridge.service import BridgeService

    token = secrets.token_urlsafe(32)
    bus = EventBus()
    service = BridgeService(device_status=device_status, bus=bus)
    server = BridgeServer(
        dispatch=service.dispatch,
        bus=bus,
        token=token,
        static=static_routes(),
        thumbnailer=Thumbnailer(),
        image_reader=resolve_image_path,
        on_client_connected=service.on_client_connected,
        port=args.bridge_port,
    )
    server.start()
    print(f"[bridge] listening 127.0.0.1:{server.port} (token via env)", flush=True)

    stop = threading.Event()

    def request_stop(*_args: Any) -> None:
        stop.set()

    previous_term = signal.signal(signal.SIGTERM, request_stop)
    process: Optional[subprocess.Popen] = None
    exit_code = 0
    try:
        if args.bridge_only:
            print(
                f"[bridge] --bridge-only: open {server.origin}/auth?token={token}\n"
                "[bridge] press Ctrl+C to stop",
                flush=True,
            )
            while not stop.wait(0.5):
                pass
        else:
            print("[bridge] spawning electron …", flush=True)
            service.spawned_at = time.monotonic()
            process = spawn_electron(binary, server.origin, token)
            while not stop.is_set():
                try:
                    exit_code = process.wait(timeout=0.5)
                    break
                except subprocess.TimeoutExpired:
                    continue
    except KeyboardInterrupt:
        pass
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        print("[bridge] shutting down", flush=True)
        service.shutdown()
        server.stop()
        signal.signal(signal.SIGTERM, previous_term)
    return exit_code or 0
