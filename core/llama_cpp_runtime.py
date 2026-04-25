"""Managed llama.cpp server runtime for multimodal chat completions."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request, error
from urllib.parse import urlparse


class LlamaCppRuntimeError(RuntimeError):
    """Raised when llama.cpp runtime operations fail."""


def is_llama_timeout_error(value: Any) -> bool:
    """Return True when error content indicates a request timeout."""
    stack: List[Any] = [value]
    seen: set[int] = set()

    while stack:
        current = stack.pop()
        if current is None:
            continue

        marker = id(current)
        if marker in seen:
            continue
        seen.add(marker)

        if isinstance(current, (TimeoutError, socket.timeout)):
            return True

        if isinstance(current, str):
            lower = current.lower()
            if "timed out" in lower or "timeout" in lower:
                return True
            continue

        lower = str(current).lower()
        if "timed out" in lower or "timeout" in lower:
            return True

        for attr in ("reason", "__cause__", "__context__"):
            related = getattr(current, attr, None)
            if related is not None and related is not current:
                stack.append(related)

    return False


class LlamaCppRuntimeManager:
    """Singleton manager for a local llama.cpp server process."""

    _instance: Optional["LlamaCppRuntimeManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._log_handle = None
        self._log_file_path: Optional[str] = None
        self._config_key: Optional[Tuple[Any, ...]] = None
        self._base_url: Optional[str] = None
        self._ref_count = 0
        self._model_alias = "local-llama-cpp"

    @classmethod
    def get_instance(cls) -> "LlamaCppRuntimeManager":
        """Get singleton runtime manager."""
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def ensure_server(
        self,
        binary_path: str,
        model_path: str,
        mmproj_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8080,
        ctx_size: int = 4096,
        gpu_layers: int = -1,
        startup_timeout: float = 90.0,
    ) -> str:
        """
        Ensure a matching llama.cpp server is available and healthy.

        Returns:
            Base URL for OpenAI-compatible server endpoint.
        """
        binary = Path(binary_path).expanduser().resolve()
        model = Path(model_path).expanduser().resolve()
        mmproj = Path(mmproj_path).expanduser().resolve() if mmproj_path else None

        if not binary.exists():
            raise LlamaCppRuntimeError(f"llama-server binary not found: {binary}")
        if not model.exists():
            raise LlamaCppRuntimeError(f"Multimodal model not found: {model}")
        if mmproj and not mmproj.exists():
            raise LlamaCppRuntimeError(f"Multimodal projector not found: {mmproj}")

        resolved_port = self.resolve_server_port(host=host, requested_port=int(port))

        config_key = (
            str(binary),
            str(model),
            str(mmproj) if mmproj else "",
            host,
            int(resolved_port),
            int(ctx_size),
            int(gpu_layers),
        )
        base_url = f"http://{host}:{resolved_port}"

        with self._lock:
            if self._is_process_running() and self._config_key == config_key:
                if self._check_health(base_url):
                    self._ref_count += 1
                    return base_url
                self._stop_locked()

            if self._is_process_running():
                self._stop_locked()

            cmd = [
                str(binary),
                "--host",
                host,
                "--port",
                str(resolved_port),
                "--model",
                str(model),
                "--ctx-size",
                str(int(ctx_size)),
                "--n-gpu-layers",
                str(int(gpu_layers)),
            ]
            if mmproj:
                cmd.extend(["--mmproj", str(mmproj)])

            try:
                log_dir = Path(__file__).resolve().parents[1] / "cache" / "llama_cpp" / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                log_path = log_dir / f"llama-server-{resolved_port}-{timestamp}.log"
                process_env = self._build_process_env(binary.parent)
                self._log_handle = log_path.open("w", encoding="utf-8", errors="replace")
                self._log_handle.write(f"# Started at {datetime.now().isoformat()}\n")
                self._log_handle.write("# Command:\n")
                self._log_handle.write(" ".join(cmd) + "\n\n")
                if "LD_LIBRARY_PATH" in process_env:
                    self._log_handle.write("# LD_LIBRARY_PATH:\n")
                    self._log_handle.write(f"{process_env['LD_LIBRARY_PATH']}\n\n")
                self._log_handle.flush()
                self._log_file_path = str(log_path)

                self._process = subprocess.Popen(
                    cmd,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=process_env,
                )
            except OSError as exc:
                raise LlamaCppRuntimeError(f"Failed to start llama-server: {exc}") from exc

            self._config_key = config_key
            self._base_url = base_url

            deadline = time.time() + startup_timeout
            while time.time() < deadline:
                if not self._is_process_running():
                    logs = self._read_recent_logs_from_path(self._log_file_path, max_lines=80)
                    details = f"\nRecent logs:\n{logs}" if logs else ""
                    raise LlamaCppRuntimeError(f"llama-server exited during startup{details}")
                if self._check_health(base_url):
                    self._ref_count = 1
                    return base_url
                time.sleep(0.5)

            self._stop_locked()
            logs = self._read_recent_logs_from_path(self._log_file_path, max_lines=80)
            details = f"\nRecent logs:\n{logs}" if logs else ""
            raise LlamaCppRuntimeError(f"Timed out waiting for llama-server to become healthy{details}")

    def release_server(self) -> None:
        """Release one runtime reference and stop server when no clients remain."""
        with self._lock:
            if self._ref_count > 0:
                self._ref_count -= 1
            if self._ref_count <= 0:
                self._stop_locked()

    def stop(self) -> None:
        """Force-stop the runtime immediately."""
        with self._lock:
            self._ref_count = 0
            self._stop_locked()

    def get_runtime_metadata(self) -> Dict[str, Any]:
        """Return current runtime metadata for diagnostics."""
        with self._lock:
            return {
                "base_url": self._base_url,
                "pid": self._process.pid if self._is_process_running() else None,
                "log_path": self._log_file_path,
                "is_running": self._is_process_running(),
            }

    def get_recent_logs(self, max_lines: int = 120) -> str:
        """Return tail of runtime logs for troubleshooting."""
        with self._lock:
            log_path = self._log_file_path

        return self._read_recent_logs_from_path(log_path, max_lines=max_lines)

    def resolve_server_port(
        self,
        host: str = "127.0.0.1",
        requested_port: int = 8080,
        scan_window: int = 200,
    ) -> int:
        """
        Resolve a usable server port for llama.cpp.

        Rules:
        - Reuse the currently managed healthy runtime port when available.
        - Otherwise prefer requested_port when free.
        - Otherwise scan upward for a free port to avoid overlap.
        """
        requested = int(requested_port)
        if requested < 1 or requested > 65535:
            raise LlamaCppRuntimeError(f"Invalid server port: {requested}")

        with self._lock:
            active_port = self._get_active_managed_port_locked(host=host)
            if active_port is not None:
                return active_port

        if self._is_port_available(host=host, port=requested):
            return requested

        for candidate in range(requested + 1, min(65536, requested + 1 + max(1, int(scan_window)))):
            if self._is_port_available(host=host, port=candidate):
                return candidate

        fallback_port = self._bind_ephemeral_port(host=host)
        if fallback_port is not None:
            return fallback_port

        raise LlamaCppRuntimeError(
            f"No available llama-server port found near {requested} on host {host}"
        )

    @staticmethod
    def _read_recent_logs_from_path(log_path: Optional[str], max_lines: int = 120) -> str:
        """Read recent runtime logs from a specific path without runtime lock access."""
        if not log_path:
            return ""

        try:
            path = Path(log_path)
            if not path.exists():
                return ""
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if max_lines <= 0:
                return "\n".join(lines)
            return "\n".join(lines[-max_lines:])
        except Exception:
            return ""

    @staticmethod
    def _build_process_env(binary_dir: Path) -> Dict[str, str]:
        """Build process environment with runtime library search path hints."""
        env = dict(os.environ)
        dir_text = str(binary_dir)
        path_sep = os.pathsep or ":"

        if os.name == "nt":
            current_path = env.get("PATH", "")
            env["PATH"] = f"{dir_text}{path_sep}{current_path}" if current_path else dir_text
            return env

        current_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{dir_text}{path_sep}{current_ld}" if current_ld else dir_text

        current_dyld = env.get("DYLD_LIBRARY_PATH", "")
        if current_dyld:
            env["DYLD_LIBRARY_PATH"] = f"{dir_text}{path_sep}{current_dyld}"
        else:
            env["DYLD_LIBRARY_PATH"] = dir_text

        return env

    @staticmethod
    def _is_port_available(host: str, port: int) -> bool:
        """Return True when a TCP port can be bound on the requested host."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, int(port)))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    @staticmethod
    def _bind_ephemeral_port(host: str) -> Optional[int]:
        """Reserve and return an ephemeral free port, or None when unavailable."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])
        except OSError:
            return None
        finally:
            sock.close()

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Dict[str, Any]] = None,
        timeout: float = 120.0,
    ) -> Dict[str, Any]:
        """Send an OpenAI-compatible chat completion request to llama-server."""
        with self._lock:
            base_url = self._base_url
            if not base_url or not self._is_process_running():
                raise LlamaCppRuntimeError("llama-server is not running")

        payload: Dict[str, Any] = {
            "model": self._model_alias,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice

        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{base_url}/v1/chat/completions",
            data=data,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LlamaCppRuntimeError(f"llama-server HTTP error {exc.code}: {detail}") from exc
        except (error.URLError, TimeoutError) as exc:
            raise LlamaCppRuntimeError(f"llama-server request failed: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise LlamaCppRuntimeError(f"Invalid JSON response from llama-server: {exc}") from exc

    def _is_process_running(self) -> bool:
        """Return True if managed server process is alive."""
        return self._process is not None and self._process.poll() is None

    def _get_active_managed_port_locked(self, host: str) -> Optional[int]:
        """
        Return active managed runtime port for host when healthy.
        Must be called under self._lock.
        """
        if not self._is_process_running() or not self._base_url:
            return None
        parsed = urlparse(self._base_url)
        active_host = parsed.hostname
        active_port = parsed.port
        if not active_host or active_port is None:
            return None
        if active_host != host:
            return None
        if not self._check_health(self._base_url):
            return None
        return int(active_port)

    def _stop_locked(self) -> None:
        """Stop process and clear runtime state. Must be called under lock."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)

        if self._log_handle:
            try:
                self._log_handle.flush()
                self._log_handle.close()
            except Exception:
                pass

        self._process = None
        self._log_handle = None
        self._config_key = None
        self._base_url = None

    def _check_health(self, base_url: str) -> bool:
        """Check server health endpoint(s)."""
        for endpoint in ("/health", "/v1/models"):
            req = request.Request(url=f"{base_url}{endpoint}", method="GET")
            try:
                with request.urlopen(req, timeout=2.0) as resp:
                    if 200 <= resp.status < 300:
                        return True
            except Exception:
                continue
        return False
