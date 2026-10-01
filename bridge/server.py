"""Local HTTP transport between the Electron window and the Python backend.

Serves on 127.0.0.1 only, with a per-launch token:

    GET  /auth?token=...   exchanges the token for an HttpOnly session cookie
    GET  /app/...          the renderer (index.html, scripts, styles)
    GET  /vendor/..., /fonts/..., /assets/...   bundled front-end assets
    GET  /events           server-sent events: the backend's signals
    POST /rpc              {"method": "group.name", "params": {...}}
    GET  /thumb?path=&size=   cached JPEG thumbnail of an image
    GET  /image?path=      the original image file

Every route except /auth needs the token, either as the `ii_token` cookie
(set by Electron before the page loads, or by /auth) or as an
`X-Bridge-Token` header. Requests must name the bridge in their Host header,
which blocks DNS-rebinding pages, and POSTs must carry `X-II-Request: 1`,
which a cross-origin page cannot add without a CORS preflight this server
never approves.
"""

from __future__ import annotations

import hmac
import json
import mimetypes
import queue
import sys
import threading
import time
from http import HTTPStatus
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlsplit

from bridge.events import EventBus, to_json

TOKEN_COOKIE = "ii_token"
TOKEN_HEADER = "X-Bridge-Token"
REQUEST_HEADER = "X-II-Request"
MAX_RPC_BODY = 16 * 1024 * 1024
SSE_HEARTBEAT_SECONDS = 15.0

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".png": "image/png",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
    ".woff": "font/woff",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

# Scripts only from the bridge itself. Inline styles are allowed because the
# renderer sizes bars and panes with style attributes.
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; font-src 'self'; connect-src 'self'; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
)


class RpcError(Exception):
    """An error the UI should show as-is, rather than as a crash."""

    def __init__(self, message: str, code: str = "error", data: Any = None):
        super().__init__(message)
        self.code = code
        self.data = data


class StaticRoutes:
    """Maps URL prefixes onto directories and individual files."""

    def __init__(self):
        self._dirs: Dict[str, Path] = {}
        self._files: Dict[str, Path] = {}

    def add_dir(self, prefix: str, root: Path) -> None:
        self._dirs[prefix.rstrip("/") + "/"] = Path(root).resolve()

    def add_file(self, url_path: str, file_path: Path) -> None:
        self._files[url_path] = Path(file_path)

    def resolve(self, url_path: str) -> Optional[Path]:
        if url_path in self._files:
            return self._files[url_path]
        for prefix, root in self._dirs.items():
            if url_path == prefix.rstrip("/"):
                url_path = prefix
            if not url_path.startswith(prefix):
                continue
            relative = url_path[len(prefix):] or "index.html"
            candidate = (root / relative).resolve()
            # Refuse anything that escapes the mapped directory.
            if candidate != root and root not in candidate.parents:
                return None
            if candidate.is_dir():
                candidate = candidate / "index.html"
            return candidate
        return None


class BridgeServer:
    """Owns the listening socket and routes requests to the backend."""

    def __init__(
        self,
        dispatch: Callable[[str, Dict[str, Any]], Any],
        bus: EventBus,
        token: str,
        static: StaticRoutes,
        thumbnailer: Optional[Callable[[str, int], Tuple[bytes, str]]] = None,
        image_reader: Optional[Callable[[str], Path]] = None,
        on_client_connected: Optional[Callable[[], None]] = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        self.dispatch = dispatch
        self.bus = bus
        self.token = token
        self.static = static
        self.thumbnailer = thumbnailer
        self.image_reader = image_reader
        self.on_client_connected = on_client_connected or (lambda: None)
        self.stopping = threading.Event()

        handler = _make_handler(self)
        self.httpd = _Server((host, port), handler)
        self.host, self.port = self.httpd.server_address[:2]
        self._thread: Optional[threading.Thread] = None

    @property
    def origin(self) -> str:
        return f"http://{self.host}:{self.port}"

    def allowed_hosts(self) -> set:
        return {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self.httpd.serve_forever, name="bridge-http", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self.stopping.set()
        self.bus.close()
        self.httpd.shutdown()
        self.httpd.server_close()

    def token_matches(self, candidate: Optional[str]) -> bool:
        if not candidate:
            return False
        return hmac.compare_digest(candidate.encode("utf-8"), self.token.encode("utf-8"))


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # Rebinding a fixed --bridge-port right after a restart would otherwise
    # fail while old connections sit in TIME_WAIT. On Windows SO_REUSEADDR
    # would let another process bind the same port, so it stays off there.
    allow_reuse_address = not sys.platform.startswith("win")


def _make_handler(bridge: BridgeServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ImageInterrogatorBridge"
        sys_version = ""

        # Request lines carry the auth query string; never print them.
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        # -- routing -------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802
            if not self._host_allowed():
                return self._send_text(HTTPStatus.FORBIDDEN, "Forbidden host")
            parts = urlsplit(self.path)
            path, query = parts.path, parse_qs(parts.query)

            if path == "/auth":
                return self._handle_auth(query)
            if path == "/favicon.ico":
                return self._send_bytes(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            if not self._authorized():
                return self._send_text(
                    HTTPStatus.UNAUTHORIZED,
                    "This window is served by ./run.sh --electron. Open it from there.",
                )
            if path == "/":
                return self._redirect("/app/")
            if path == "/events":
                return self._serve_events()
            if path == "/thumb":
                return self._serve_thumbnail(query)
            if path == "/image":
                return self._serve_image(query)
            return self._serve_static(path)

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_allowed():
                return self._send_text(HTTPStatus.FORBIDDEN, "Forbidden host")
            if urlsplit(self.path).path != "/rpc":
                return self._send_text(HTTPStatus.NOT_FOUND, "Not found")
            if not self._authorized():
                return self._send_json(HTTPStatus.UNAUTHORIZED, {"ok": False, "error": {"message": "Unauthorized"}})
            if self.headers.get(REQUEST_HEADER) != "1" or not self._origin_allowed():
                return self._send_json(HTTPStatus.FORBIDDEN, {"ok": False, "error": {"message": "Forbidden"}})

            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_RPC_BODY:
                return self._send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"ok": False, "error": {"message": "Bad length"}})
            try:
                request = json.loads(self.rfile.read(length) or b"{}")
                method = str(request.get("method", ""))
                params = request.get("params") or {}
                if not isinstance(params, dict):
                    raise ValueError("params must be an object")
            except (ValueError, AttributeError) as exc:
                return self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": {"message": f"Bad request: {exc}"}})

            try:
                result = bridge.dispatch(method, params)
            except RpcError as exc:
                return self._send_json(
                    HTTPStatus.OK,
                    {"ok": False, "error": {"message": str(exc), "code": exc.code, "data": exc.data}},
                )
            except Exception as exc:  # noqa: BLE001 - reported to the UI
                message = str(exc).strip() or exc.__class__.__name__
                return self._send_json(
                    HTTPStatus.OK,
                    {"ok": False, "error": {"message": message, "code": "exception"}},
                )
            return self._send_json(HTTPStatus.OK, {"ok": True, "result": result})

        # -- security checks ----------------------------------------------

        def _host_allowed(self) -> bool:
            return (self.headers.get("Host") or "") in bridge.allowed_hosts()

        def _origin_allowed(self) -> bool:
            origin = self.headers.get("Origin")
            if not origin:
                return True  # non-browser client using the header token
            return origin in {f"http://{host}" for host in bridge.allowed_hosts()}

        def _authorized(self) -> bool:
            if bridge.token_matches(self.headers.get(TOKEN_HEADER)):
                return True
            raw_cookie = self.headers.get("Cookie")
            if not raw_cookie:
                return False
            cookie = SimpleCookie()
            try:
                cookie.load(raw_cookie)
            except Exception:
                return False
            morsel = cookie.get(TOKEN_COOKIE)
            return bool(morsel) and bridge.token_matches(morsel.value)

        def _handle_auth(self, query: Dict[str, list]) -> None:
            candidate = (query.get("token") or [""])[0]
            if not bridge.token_matches(candidate):
                return self._send_text(HTTPStatus.UNAUTHORIZED, "Invalid token")
            self.send_response(HTTPStatus.FOUND)
            self.send_header(
                "Set-Cookie",
                f"{TOKEN_COOKIE}={candidate}; Path=/; HttpOnly; SameSite=Strict",
            )
            self.send_header("Location", "/app/")
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()

        # -- responses -----------------------------------------------------

        def _serve_static(self, url_path: str) -> None:
            target = bridge.static.resolve(url_path)
            if target is None or not target.is_file():
                return self._send_text(HTTPStatus.NOT_FOUND, "Not found")
            content_type = CONTENT_TYPES.get(target.suffix.lower()) or (
                mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            )
            extra = {"Cache-Control": "no-cache"}
            if target.suffix.lower() == ".html":
                extra["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
            return self._send_bytes(HTTPStatus.OK, target.read_bytes(), content_type, extra)

        def _serve_thumbnail(self, query: Dict[str, list]) -> None:
            if bridge.thumbnailer is None:
                return self._send_text(HTTPStatus.NOT_FOUND, "Not found")
            path = (query.get("path") or [""])[0]
            try:
                size = max(32, min(2048, int((query.get("size") or ["200"])[0])))
            except ValueError:
                size = 200
            # Revalidated on every use: the URL names a path, not a version,
            # and an image can be edited in place.
            headers = {"Cache-Control": "private, no-cache"}
            cache_key = getattr(bridge.thumbnailer, "cache_key", None)
            try:
                if cache_key is not None:
                    headers["ETag"] = f'"{cache_key(path, size)}"'
                    if self._etag_matches(headers["ETag"]):
                        return self._send_not_modified(headers)
                data, content_type = bridge.thumbnailer(path, size)
            except FileNotFoundError:
                return self._send_text(HTTPStatus.NOT_FOUND, "Not found")
            except Exception as exc:  # noqa: BLE001 - unreadable image
                return self._send_text(HTTPStatus.UNPROCESSABLE_ENTITY, f"Cannot decode image: {exc}")
            return self._send_bytes(HTTPStatus.OK, data, content_type, headers)

        def _serve_image(self, query: Dict[str, list]) -> None:
            if bridge.image_reader is None:
                return self._send_text(HTTPStatus.NOT_FOUND, "Not found")
            path = (query.get("path") or [""])[0]
            try:
                target = bridge.image_reader(path)
            except (FileNotFoundError, ValueError):
                return self._send_text(HTTPStatus.NOT_FOUND, "Not found")
            content_type = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
            stat = target.stat()
            headers = {"Cache-Control": "private, no-cache", "ETag": f'"{stat.st_mtime_ns:x}-{stat.st_size:x}"'}
            if self._etag_matches(headers["ETag"]):
                return self._send_not_modified(headers)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(stat.st_size))
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            with open(target, "rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)

        def _serve_events(self) -> None:
            subscriber = bridge.bus.subscribe()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                self.wfile.write(b"retry: 1500\n\n")
                self.wfile.flush()
                bridge.on_client_connected()
                last_write = time.monotonic()
                while not bridge.stopping.is_set():
                    try:
                        item = subscriber.get(timeout=1.0)
                    except queue.Empty:
                        if time.monotonic() - last_write >= SSE_HEARTBEAT_SECONDS:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                            last_write = time.monotonic()
                        continue
                    if item is None:
                        break
                    name, payload = item
                    self.wfile.write(f"event: {name}\ndata: {payload}\n\n".encode("utf-8"))
                    self.wfile.flush()
                    last_write = time.monotonic()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            finally:
                bridge.bus.unsubscribe(subscriber)

        def _redirect(self, location: str) -> None:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def _send_json(self, status: HTTPStatus, payload: Any) -> None:
            body = to_json(payload).encode("utf-8")
            self._send_bytes(status, body, "application/json; charset=utf-8", {"Cache-Control": "no-store"})

        def _send_text(self, status: HTTPStatus, text: str) -> None:
            self._send_bytes(status, text.encode("utf-8"), "text/plain; charset=utf-8")

        def _etag_matches(self, etag: str) -> bool:
            header = self.headers.get("If-None-Match")
            if not header:
                return False
            candidates = [candidate.strip() for candidate in header.split(",")]
            return "*" in candidates or etag in candidates or f"W/{etag}" in candidates

        def _send_not_modified(self, headers: Dict[str, str]) -> None:
            self.send_response(HTTPStatus.NOT_MODIFIED)
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()

        def _send_bytes(
            self,
            status: HTTPStatus,
            body: bytes,
            content_type: str,
            extra_headers: Optional[Dict[str, str]] = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            if body:
                self.wfile.write(body)

    return Handler
