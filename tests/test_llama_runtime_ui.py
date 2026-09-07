"""Unit tests for the managed-runtime Inquiry UI.

Covers the resolution rules that decide which llama-server actually launches,
and the card states that tell the user what they are running.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtWidgets import QApplication  # noqa: E402

from core.llama_provisioner import ProvisionError, UpdateStatus  # noqa: E402
from ui.llama_runtime import (  # noqa: E402
    RUNTIME_MODE_CUSTOM,
    RUNTIME_MODE_MANAGED,
    LlamaRuntimeCard,
    resolve_runtime_binary,
    runtime_mode,
    runtime_summary,
)

_app = QApplication.instance() or QApplication([])


def make_runtime(root: Path, accelerator: str = "cuda", **extra) -> Path:
    """Create a fake installed runtime with provisioner state."""
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    executable = bin_dir / "llama-server"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)

    record = {
        "version": "b10828 (source, cuda)",
        "method": "source",
        "accelerator": accelerator,
        "executable": str(executable),
        "installed_at": "2026-09-06T15:50:12",
    }
    record.update(extra)
    (root / "active-runtime.json").write_text(json.dumps(record), encoding="utf-8")
    return executable


class TestRuntimeMode(unittest.TestCase):
    def test_empty_config_defaults_to_managed(self):
        self.assertEqual(runtime_mode({}), RUNTIME_MODE_MANAGED)

    def test_explicit_mode_is_respected(self):
        self.assertEqual(runtime_mode({"llama_runtime_mode": "custom"}), RUNTIME_MODE_CUSTOM)
        self.assertEqual(runtime_mode({"llama_runtime_mode": "managed"}), RUNTIME_MODE_MANAGED)

    def test_unknown_mode_falls_back_to_inference(self):
        config = {"llama_runtime_mode": "nonsense", "llama_binary_path": "/opt/llama-server"}
        self.assertEqual(runtime_mode(config), RUNTIME_MODE_CUSTOM)

    def test_legacy_config_with_a_path_is_treated_as_custom(self):
        # Settings saved before this option existed carried only a path. It must
        # keep launching that binary rather than being silently ignored.
        self.assertEqual(
            runtime_mode({"llama_binary_path": "/somewhere/else/llama-server"}),
            RUNTIME_MODE_CUSTOM,
        )

    def test_legacy_config_with_no_path_is_managed(self):
        self.assertEqual(runtime_mode({"llama_binary_path": ""}), RUNTIME_MODE_MANAGED)


class TestRuntimeResolution(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_custom_mode_uses_the_configured_path(self):
        config = {"llama_runtime_mode": "custom", "llama_binary_path": "/opt/custom/llama-server"}
        self.assertEqual(resolve_runtime_binary(config, self.root), "/opt/custom/llama-server")

    def test_managed_mode_ignores_a_stale_configured_path(self):
        """The point of managed mode: reprovisioning takes effect by itself.

        A stored absolute path goes stale the moment the backend is rebuilt, so
        managed mode resolves from disk instead of trusting what was saved.
        """
        executable = make_runtime(self.root)
        config = {
            "llama_runtime_mode": "managed",
            "llama_binary_path": "/an/old/path/from/last/year/llama-server",
        }
        self.assertEqual(resolve_runtime_binary(config, self.root), str(executable))

    def test_managed_mode_with_nothing_installed_resolves_empty(self):
        config = {"llama_runtime_mode": "managed"}
        self.assertEqual(resolve_runtime_binary(config, self.root), "")

    def test_custom_mode_with_no_path_resolves_empty(self):
        config = {"llama_runtime_mode": "custom", "llama_binary_path": "   "}
        self.assertEqual(resolve_runtime_binary(config, self.root), "")


class TestRuntimeSummary(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_absent_runtime_reports_not_installed(self):
        summary = runtime_summary(self.root)
        self.assertFalse(summary["installed"])
        self.assertEqual(summary["executable"], "")

    def test_gpu_runtime_is_flagged_as_gpu(self):
        make_runtime(self.root, accelerator="cuda", cuda_architectures="121a-real")
        summary = runtime_summary(self.root)
        self.assertTrue(summary["installed"])
        self.assertTrue(summary["is_gpu"])
        self.assertEqual(summary["accelerator"], "cuda")
        self.assertEqual(summary["cuda_architectures"], "121a-real")
        self.assertEqual(summary["fallback_from"], "", "a CUDA build is not a fallback")

    def test_metal_counts_as_gpu(self):
        make_runtime(self.root, accelerator="metal")
        self.assertTrue(runtime_summary(self.root)["is_gpu"])

    def test_cpu_runtime_is_not_flagged_as_gpu(self):
        make_runtime(self.root, accelerator="cpu")
        summary = runtime_summary(self.root)
        self.assertTrue(summary["installed"])
        self.assertFalse(summary["is_gpu"])

    def test_cpu_runtime_on_gpu_hardware_is_reported_as_a_fallback(self):
        # The ladder's last rung. The provisioning dialog warns once and closes;
        # this is what still says so while a slow batch is running.
        make_runtime(self.root, accelerator="cpu")
        with patch("ui.llama_runtime.detect_accelerator", return_value="cuda"):
            summary = runtime_summary(self.root)
        self.assertEqual(summary["fallback_from"], "cuda")

    def test_cpu_runtime_on_cpu_only_hardware_is_not_a_fallback(self):
        make_runtime(self.root, accelerator="cpu")
        with patch("ui.llama_runtime.detect_accelerator", return_value="cpu"):
            summary = runtime_summary(self.root)
        self.assertEqual(summary["fallback_from"], "", "CPU is the correct target here")


class TestRuntimeCardStates(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_absent_runtime_prompts_to_install(self):
        card = LlamaRuntimeCard(provision_dir=self.root)
        self.assertIn("No runtime installed", card.headline_label.text())
        self.assertEqual(card.provision_button.text(), "Install Runtime")
        self.assertFalse(card.warning_label.isVisibleTo(card))

    def test_installed_runtime_shows_backend_and_build_details(self):
        make_runtime(self.root, accelerator="cuda", cuda_architectures="121a-real")
        card = LlamaRuntimeCard(provision_dir=self.root)
        self.assertIn("cuda", card.headline_label.text())
        self.assertIn("Ready", card.headline_label.text())
        detail = card.detail_label.text()
        self.assertIn("121a-real", detail)
        self.assertIn("source", detail)
        self.assertEqual(card.provision_button.text(), "Reinstall / Update")

    def test_refresh_picks_up_a_newly_installed_runtime(self):
        card = LlamaRuntimeCard(provision_dir=self.root)
        self.assertIn("No runtime installed", card.headline_label.text())
        make_runtime(self.root, accelerator="cuda")
        card.refresh()
        self.assertIn("Ready", card.headline_label.text())

    def test_health_check_reports_a_missing_runtime(self):
        card = LlamaRuntimeCard(provision_dir=self.root)
        self.assertIn("No runtime installed", card.check_health())

    def test_cpu_fallback_shows_a_persistent_warning(self):
        make_runtime(self.root, accelerator="cpu")
        with patch("ui.llama_runtime.detect_accelerator", return_value="cuda"):
            card = LlamaRuntimeCard(provision_dir=self.root)
        self.assertTrue(card.warning_label.isVisibleTo(card))
        self.assertIn("cuda", card.warning_label.text())
        self.assertIn("slower", card.warning_label.text())

    def test_correct_backend_shows_no_warning(self):
        make_runtime(self.root, accelerator="cuda")
        card = LlamaRuntimeCard(provision_dir=self.root)
        self.assertFalse(card.warning_label.isVisibleTo(card))


class TestUpdateDisplay(unittest.TestCase):
    """The card reports what an update costs, not just that one exists."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        make_runtime(self.root, accelerator="cuda")
        self.card = LlamaRuntimeCard(provision_dir=self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_up_to_date_is_reported(self):
        self.card._on_update_checked(UpdateStatus(latest_version="b10850", behind=0))
        self.assertIn("Up to date", self.card.update_label.text())
        self.assertTrue(self.card.update_label.isVisibleTo(self.card))

    def test_slightly_behind_is_neither_current_nor_an_update(self):
        # Claiming "up to date" while two builds behind would be false; calling
        # it an update would make the signal meaningless at ~12 builds a day.
        self.card._on_update_checked(
            UpdateStatus(latest_version="b10850", behind=2, update_available=False)
        )
        text = self.card.update_label.text()
        self.assertIn("Close to current", text)
        self.assertIn("2 builds ahead", text)
        self.assertNotIn("Up to date", text)
        self.assertNotIn("Update available", text)

    def test_single_build_behind_is_not_pluralised(self):
        self.card._on_update_checked(
            UpdateStatus(latest_version="b10850", behind=1, update_available=False)
        )
        self.assertIn("1 build ahead", self.card.update_label.text())

    def test_reported_update_states_the_distance(self):
        self.card._on_update_checked(
            UpdateStatus(
                latest_version="b10850", behind=142, update_available=True, action="compile"
            )
        )
        self.assertIn("142 builds behind", self.card.update_label.text())

    def test_release_update_names_the_cheap_path(self):
        self.card._on_update_checked(
            UpdateStatus(latest_version="b10850", update_available=True, action="release")
        )
        text = self.card.update_label.text()
        self.assertIn("b10850", text)
        self.assertIn("downloads a matched release", text)

    def test_compile_update_warns_about_the_rebuild(self):
        self.card._on_update_checked(
            UpdateStatus(
                latest_version="b10850",
                update_available=True,
                action="compile",
                warning="No official asset exists.",
            )
        )
        text = self.card.update_label.text()
        self.assertIn("requires a source rebuild", text)
        self.assertIn("No official asset exists.", text)

    def test_check_failure_is_shown_rather_than_swallowed(self):
        self.card._on_update_checked(UpdateStatus(error="no route to host"))
        self.assertIn("no route to host", self.card.update_label.text())

    def test_refresh_discards_a_stale_update_result(self):
        # After provisioning, "update available: bNNNN" may describe the
        # version that was just installed.
        self.card._on_update_checked(
            UpdateStatus(latest_version="b10850", update_available=True, action="release")
        )
        self.assertTrue(self.card.update_label.isVisibleTo(self.card))
        self.card.refresh()
        self.assertFalse(self.card.update_label.isVisibleTo(self.card))
        self.assertIsNone(self.card.update_status)

    def test_update_check_is_disabled_without_a_runtime(self):
        with tempfile.TemporaryDirectory() as empty:
            card = LlamaRuntimeCard(provision_dir=Path(empty))
            self.assertFalse(card.update_button.isEnabled())

    def test_update_check_preserves_a_source_install(self):
        worker = MagicMock()
        worker.isRunning.return_value = False
        with patch("ui.workers.LlamaUpdateCheckWorker", return_value=worker) as worker_type:
            self.card.check_for_updates()

        config_arg, version_arg = worker_type.call_args.args
        self.assertEqual(config_arg.install_method, "source")
        self.assertEqual(version_arg, "b10828 (source, cuda)")
        worker.start.assert_called_once_with()


class TestHealthCheckWorker(unittest.TestCase):
    def test_validation_failure_is_emitted_for_the_ui(self):
        from ui.workers import LlamaHealthCheckWorker

        failures = []
        worker = LlamaHealthCheckWorker("/missing/llama-server")
        worker.completed.connect(failures.append)
        with patch(
            "core.llama_provisioner.validate_server",
            side_effect=ProvisionError("runtime did not start"),
        ):
            worker.run()

        self.assertEqual(failures, ["runtime did not start"])


class TestConfigWidgetContract(unittest.TestCase):
    """The Inquiry tab indexes these keys; renaming one breaks model loading."""

    REQUIRED_KEYS = {
        "runtime_card",
        "custom_binary_check",
        "binary_path_edit",
        "model_path_edit",
        "mmproj_path_edit",
        "ctx_size_spin",
        "gpu_layers_spin",
        "temperature_spin",
        "max_tokens_spin",
        "server_port_spin",
    }

    def test_every_expected_reference_is_returned(self):
        from ui.dialogs import create_llama_config_widget

        _widget, refs = create_llama_config_widget({})
        self.assertTrue(self.REQUIRED_KEYS.issubset(refs), self.REQUIRED_KEYS - set(refs))

    def test_sections_are_grouped(self):
        from PyQt6.QtWidgets import QGroupBox

        from ui.dialogs import create_llama_config_widget

        widget, _refs = create_llama_config_widget({})
        titles = [group.title() for group in widget.findChildren(QGroupBox)]
        self.assertEqual(titles, ["Runtime", "Model", "Inference"])

    def test_custom_toggle_reflects_saved_mode(self):
        from ui.dialogs import create_llama_config_widget

        _w, managed = create_llama_config_widget({"llama_runtime_mode": "managed"})
        self.assertFalse(managed["custom_binary_check"].isChecked())

        _w2, custom = create_llama_config_widget(
            {"llama_runtime_mode": "custom", "llama_binary_path": "/opt/llama-server"}
        )
        self.assertTrue(custom["custom_binary_check"].isChecked())
        self.assertEqual(custom["binary_path_edit"].text(), "/opt/llama-server")

    def test_custom_mode_disables_the_managed_card(self):
        # The card describes the managed runtime, which is not what launches
        # while a custom binary is selected.
        from ui.dialogs import create_llama_config_widget

        _w, refs = create_llama_config_widget({"llama_runtime_mode": "custom"})
        self.assertFalse(refs["runtime_card"].isEnabled())


if __name__ == "__main__":
    unittest.main()
