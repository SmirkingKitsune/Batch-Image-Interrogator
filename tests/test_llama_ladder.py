"""The acquisition ladder recorded beside an installed llama.cpp runtime."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import llama_provisioner as lp
from core.llama_runtime_info import summarize_runtime


def cfg_for(root: Path, **overrides) -> lp.ProvisionConfig:
    values = dict(
        provision_dir=root, accelerator="cuda", cuda_arch="86", platform="linux", arch="x64",
    )
    values.update(overrides)
    return lp.ProvisionConfig(**values)


class LadderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_record_is_attached_only_to_an_existing_runtime(self):
        lp._record_ladder(self.root, [{"method": "source"}])
        self.assertFalse((self.root / "active-runtime.json").exists())

        (self.root / "active-runtime.json").write_text(json.dumps({"version": "b1"}), encoding="utf-8")
        lp._record_ladder(self.root, [{"method": "source", "ok": True}])
        record = json.loads((self.root / "active-runtime.json").read_text(encoding="utf-8"))
        self.assertEqual(record["version"], "b1")
        self.assertEqual(record["ladder"], [{"method": "source", "ok": True}])

    def test_unpublished_release_is_recorded_as_skipped(self):
        with patch.object(lp, "release_asset_patterns", return_value=[]):
            rungs = lp.LlamaProvisioner(cfg_for(self.root))._skipped_rungs()
        self.assertEqual(len(rungs), 1)
        self.assertEqual((rungs[0]["method"], rungs[0]["accelerator"]), ("release", "cuda"))
        self.assertTrue(rungs[0]["skipped"])
        self.assertIn("no published build for linux-x64 + cuda", rungs[0]["detail"])

        with patch.object(lp, "release_asset_patterns", return_value=[]):
            explicit = lp.LlamaProvisioner(cfg_for(self.root, install_method="source"))._skipped_rungs()
        self.assertEqual(explicit, [], "an explicit method is not a ladder")

    def test_full_ladder_is_recorded_when_the_cpu_rung_sticks(self):
        root = self.root

        def fake_run_plan(self, plan):
            cfg = plan[0]
            if cfg.install_method == "source":
                raise lp.ProvisionError("nvcc not found on PATH\nmore detail")
            binary = root / "bin" / "llama-server"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            lp._write_version(cfg, binary, "b10828", cfg.install_method)

        patterns = lambda cfg: [] if cfg.accelerator == "cuda" else ["llama-*-bin-ubuntu-x64.zip"]
        with patch.object(lp, "release_asset_patterns", side_effect=patterns), \
                patch.object(lp, "build_install_plan", side_effect=lambda cfg: [cfg]), \
                patch.object(lp.LlamaProvisioner, "_run_plan", fake_run_plan), \
                patch.object(lp, "detect_accelerator", return_value="cuda"):
            provisioner = lp.LlamaProvisioner(cfg_for(root))
            binary = provisioner.ensure_runtime()
            summary = summarize_runtime(root, detect=lambda: "cuda")

        self.assertTrue(binary.name.startswith("llama-server"))
        self.assertIn("Installed a cpu runtime", provisioner.target_mismatch)
        ladder = [(r["method"], r["accelerator"], r["ok"], r["skipped"]) for r in summary["ladder"]]
        self.assertEqual(ladder, [
            ("release", "cuda", False, True),
            ("source", "cuda", False, False),
            ("release", "cpu", True, False),
        ])
        self.assertEqual(summary["ladder"][1]["detail"], "nvcc not found on PATH")
        self.assertEqual(summary["ladder"][2]["detail"], "installed — last rung")
        self.assertEqual(summary["fallback_from"], "cuda")


if __name__ == "__main__":
    unittest.main()
