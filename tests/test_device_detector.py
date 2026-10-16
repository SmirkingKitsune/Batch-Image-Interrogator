"""Tests for keeping PyTorch off Triton kernels that cannot be built."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from core.device_detector import DeviceDetector, _triton_build_blocker


def fake_torch_native(calls, triton_ops=("bmm",)):
    """torch._native stand-ins that record deregistration."""
    native = ModuleType("torch._native")
    native.registry = SimpleNamespace(get_dsl_operations=lambda _dsl: list(triton_ops))
    native.triton_utils = SimpleNamespace(deregister_op_overrides=lambda: calls.append("deregistered"))
    return {"torch._native": native}


class TritonBuildBlockerTests(unittest.TestCase):
    def test_reports_missing_python_headers_then_a_missing_compiler(self):
        with tempfile.TemporaryDirectory() as include_dir, \
                mock.patch("core.device_detector.sysconfig.get_paths", return_value={"include": include_dir}), \
                mock.patch("core.device_detector.shutil.which", return_value=None), \
                mock.patch.dict(os.environ, {}, clear=True):
            self.assertIn("Python.h", _triton_build_blocker())

            Path(include_dir, "Python.h").touch()
            self.assertIn("C compiler", _triton_build_blocker())

            os.environ["CC"] = "cc"
            self.assertIsNone(_triton_build_blocker())


class AvoidUnbuildableTritonTests(unittest.TestCase):
    def test_triton_ops_are_deregistered_when_triton_cannot_build(self):
        calls = []
        with mock.patch("core.device_detector._triton_build_blocker", return_value="Python.h is missing"), \
                mock.patch.dict(sys.modules, fake_torch_native(calls)), \
                self.assertLogs("core.device_detector", level="WARNING") as logs:
            DeviceDetector()._avoid_unbuildable_triton()

        self.assertEqual(calls, ["deregistered"])
        self.assertIn("Python.h is missing", logs.output[0])

    def test_nothing_changes_when_triton_can_build(self):
        calls = []
        with mock.patch("core.device_detector._triton_build_blocker", return_value=None), \
                mock.patch.dict(sys.modules, fake_torch_native(calls)):
            DeviceDetector()._avoid_unbuildable_triton()

        self.assertEqual(calls, [])

    def test_a_failed_deregistration_is_only_logged(self):
        native = fake_torch_native([])

        def broken():
            raise RuntimeError("library already destroyed")

        native["torch._native"].triton_utils.deregister_op_overrides = broken
        with mock.patch("core.device_detector._triton_build_blocker", return_value="Python.h is missing"), \
                mock.patch.dict(sys.modules, native), \
                self.assertLogs("core.device_detector", level="WARNING") as logs:
            DeviceDetector()._avoid_unbuildable_triton()

        self.assertIn("library already destroyed", logs.output[0])


if __name__ == "__main__":
    unittest.main()
