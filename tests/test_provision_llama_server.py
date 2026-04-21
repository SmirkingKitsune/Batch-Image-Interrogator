"""Unit tests for llama.cpp release asset selection."""

import importlib.util
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "provision_llama_server.py"
SPEC = importlib.util.spec_from_file_location("provision_llama_server", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class TestProvisionLlamaServerSelection(unittest.TestCase):
    """Covers CUDA/CPU fallback selection policy."""

    @staticmethod
    def _assets() -> list[dict[str, str]]:
        return [
            {
                "name": "llama-b5124-bin-win-cuda-12.4-x64.zip",
                "browser_download_url": "https://example.invalid/cuda-12.4.zip",
            },
            {
                "name": "llama-b5131-bin-win-cuda-13.1-x64.zip",
                "browser_download_url": "https://example.invalid/cuda-13.1.zip",
            },
            {
                "name": "llama-b5000-bin-win-cpu-x64.zip",
                "browser_download_url": "https://example.invalid/cpu.zip",
            },
        ]

    def test_cuda_prefers_highest_exact_match(self):
        selected = MODULE._select_asset(
            assets=self._assets(),
            os_id="windows",
            arch_id="x64",
            prefer_cuda=True,
            prefer_rocm=False,
            target_cuda_version=13.1,
            cuda_versions=[13.1, 12.4],
            rocm_versions=[],
        )
        assert selected is not None
        self.assertIn("cuda-13.1", str(selected["name"]).lower())
        self.assertIn("exact toolkit match", str(selected.get("_selection_note", "")))

    def test_cuda_uses_highest_lower_compatible_when_exact_missing(self):
        selected = MODULE._select_asset(
            assets=self._assets(),
            os_id="windows",
            arch_id="x64",
            prefer_cuda=True,
            prefer_rocm=False,
            target_cuda_version=12.8,
            cuda_versions=[12.8],
            rocm_versions=[],
        )
        assert selected is not None
        self.assertIn("cuda-12.4", str(selected["name"]).lower())
        self.assertIn("highest <= detected toolkit 12.8", str(selected.get("_selection_note", "")))

    def test_cuda_12_0_falls_back_to_cpu_with_warning(self):
        selected = MODULE._select_asset(
            assets=self._assets(),
            os_id="windows",
            arch_id="x64",
            prefer_cuda=True,
            prefer_rocm=False,
            target_cuda_version=12.0,
            cuda_versions=[12.0],
            rocm_versions=[],
        )
        assert selected is not None
        self.assertIn("cpu", str(selected["name"]).lower())
        self.assertEqual(selected.get("_selection_warning"), "cuda_too_old_for_available_builds")


if __name__ == "__main__":
    unittest.main()
