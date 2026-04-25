"""Unit tests for llama.cpp runtime port resolution."""

import importlib.util
import socket
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[1] / "core" / "llama_cpp_runtime.py"
SPEC = importlib.util.spec_from_file_location("llama_cpp_runtime", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)
LlamaCppRuntimeError = MODULE.LlamaCppRuntimeError
LlamaCppRuntimeManager = MODULE.LlamaCppRuntimeManager


class TestLlamaCppRuntimePortResolution(unittest.TestCase):
    """Tests for llama.cpp runtime port selection and overlap avoidance."""

    @staticmethod
    def _get_free_local_port() -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
        finally:
            sock.close()

    def setUp(self):
        self.manager = LlamaCppRuntimeManager()

    def test_resolve_server_port_returns_requested_when_available(self):
        preferred = self._get_free_local_port()
        resolved = self.manager.resolve_server_port("127.0.0.1", preferred)
        self.assertEqual(resolved, preferred)

    def test_resolve_server_port_avoids_busy_requested_port(self):
        busy_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            busy_sock.bind(("127.0.0.1", 0))
            busy_sock.listen(1)
            busy_port = int(busy_sock.getsockname()[1])

            resolved = self.manager.resolve_server_port("127.0.0.1", busy_port)
            self.assertNotEqual(resolved, busy_port)
            self.assertTrue(self.manager._is_port_available("127.0.0.1", resolved))
        finally:
            busy_sock.close()

    def test_resolve_server_port_reuses_active_managed_runtime_port(self):
        active_port = self._get_free_local_port()
        self.manager._base_url = f"http://127.0.0.1:{active_port}"
        with mock.patch.object(self.manager, "_is_process_running", return_value=True):
            with mock.patch.object(self.manager, "_check_health", return_value=True):
                resolved = self.manager.resolve_server_port("127.0.0.1", requested_port=8080)
        self.assertEqual(resolved, active_port)

    def test_resolve_server_port_rejects_invalid_port(self):
        with self.assertRaises(LlamaCppRuntimeError):
            self.manager.resolve_server_port("127.0.0.1", requested_port=70000)


if __name__ == "__main__":
    unittest.main()
