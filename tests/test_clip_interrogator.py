"""Tests for CLIPInterrogator's model loading and caption handling, against stub clip-interrogator and open_clip packages."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
import weakref
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from PIL import Image

from interrogators.clip_interrogator import CLIPInterrogator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeTable:
    def __init__(self, labels):
        self.labels = list(labels)

    def rank(self, image_features, top_count=1, reverse=False):
        return self.labels[:top_count]


class FakeConfig:
    clip_model = None
    clip_preprocess = None

    def __init__(self, **kwargs):
        self.flavor_intermediate_count = 2048
        self.clip_model_path = None
        self.quiet = True
        self.__dict__.update(kwargs)


class FakeClipModel:
    def __init__(self, name, kwargs):
        self.name = name
        self.kwargs = kwargs
        self.in_eval = False

    def eval(self):
        self.in_eval = True
        return self


built_models = []


def fake_create_model_and_transforms(name, **kwargs):
    model = FakeClipModel(name, kwargs)
    built_models.append(weakref.ref(model))
    return model, None, "preprocess"


class FakeInterrogator:
    instances = []
    # Makes the error raised while "moving the model" to a CUDA device,
    # when set. A fresh one each time, as a stored exception would keep its
    # traceback, and with it the model.
    cuda_error = None

    def __init__(self, config):
        self.config = config
        if config.device == "cuda" and FakeInterrogator.cuda_error is not None:
            raise FakeInterrogator.cuda_error()
        self.caption_calls = 0
        self.mediums = FakeTable(["a photograph"])
        self.artists = FakeTable(["by some artist"])
        self.trendings = FakeTable(["trending on artstation"])
        self.movements = FakeTable(["realism"])
        self.flavors = FakeTable(["sharp focus", "highly detailed", "4k"])
        FakeInterrogator.instances.append(self)

    def generate_caption(self, image):
        self.caption_calls += 1
        assert self.config.caption_model_name, "No caption model loaded."
        return "a cat on a sofa"

    def tokenize(self, texts):
        return texts

    def image_to_features(self, image):
        return "features"

    def chain(self, image_features, phrases, best_prompt="", best_sim=0,
              min_count=8, max_count=32, desc="", reverse=False):
        return ", ".join(sorted(phrases)[:2])

    def similarities(self, image_features, candidates):
        return [len(candidate) for candidate in candidates]

    def interrogate(self, image):
        return self.generate_caption(image) + ", sharp focus"

    def interrogate_fast(self, image):
        return self.generate_caption(image) + ", sharp focus"

    def interrogate_classic(self, image):
        return self.generate_caption(image) + ", a photograph"

    def interrogate_negative(self, image):
        return "blurry, lowres"


def _merge_tables(tables, ci):
    return FakeTable([label for table in tables for label in table.labels])


def _truncate_to_fit(text, tokenize):
    return text


class CLIPInterrogatorCaptionTests(unittest.TestCase):
    def setUp(self):
        package = ModuleType("clip_interrogator")
        package.Config = FakeConfig
        package.Interrogator = FakeInterrogator
        module = ModuleType("clip_interrogator.clip_interrogator")
        module._merge_tables = _merge_tables
        module._truncate_to_fit = _truncate_to_fit
        package.clip_interrogator = module
        open_clip = ModuleType("open_clip")
        open_clip.create_model_and_transforms = fake_create_model_and_transforms

        names = ("clip_interrogator", "clip_interrogator.clip_interrogator", "open_clip")
        self._original_modules = {name: sys.modules.get(name) for name in names}
        sys.modules.update({names[0]: package, names[1]: module, names[2]: open_clip})
        FakeInterrogator.instances.clear()
        FakeInterrogator.cuda_error = None
        built_models.clear()

        self._tmp = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self._tmp.name) / "image.png")
        Image.new("RGB", (8, 8), (20, 40, 60)).save(self.image_path)

    def tearDown(self):
        for name, module in self._original_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        self._tmp.cleanup()

    def _load(self, mode, caption_model=None):
        interrogator = CLIPInterrogator()
        interrogator.load_model(mode=mode, device="cpu", caption_model=caption_model)
        return interrogator, FakeInterrogator.instances[-1]

    def test_no_caption_model_is_passed_explicitly(self):
        _interrogator, ci = self._load("best")
        # Leaving the name out would fall back to clip-interrogator's blip-large.
        self.assertIn("caption_model_name", ci.config.__dict__)
        self.assertIsNone(ci.config.caption_model_name)

    def test_no_caption_model_never_generates_a_caption(self):
        for mode in CLIPInterrogator.MODES:
            with self.subTest(mode=mode):
                interrogator, ci = self._load(mode)
                result = interrogator.interrogate(self.image_path)

                self.assertEqual(ci.caption_calls, 0)
                self.assertTrue(result["tags"])
                self.assertNotIn("a cat on a sofa", result["raw_output"])
                self.assertEqual(interrogator.get_model_type(), "CLIP")

    def test_caption_free_classic_leads_with_clip_terms(self):
        interrogator, _ci = self._load("classic")
        result = interrogator.interrogate(self.image_path)
        self.assertEqual(
            result["raw_output"],
            "a photograph by some artist, trending on artstation, realism, "
            "sharp focus, highly detailed, 4k",
        )

    def _cuda_available(self):
        detector = type("Detector", (), {"is_pytorch_cuda_available": lambda self: True})()
        return patch("core.device_detector.get_device_detector", return_value=detector)

    def test_running_out_of_gpu_memory_is_a_load_error_not_a_cpu_fallback(self):
        import torch

        FakeInterrogator.cuda_error = lambda: torch.OutOfMemoryError(
            "CUDA out of memory. Tried to allocate 1.12 GiB."
        )
        interrogator = CLIPInterrogator("ViT-bigG-14/laion2b_s39b_b160k")
        with self._cuda_available(), self.assertRaises(RuntimeError) as raised:
            interrogator.load_model(mode="best", device="cuda", caption_model="blip-base")

        self.assertTrue(str(raised.exception).startswith(
            "Not enough GPU memory to load ViT-bigG-14/laion2b_s39b_b160k with blip-base"
        ))
        self.assertFalse(interrogator.is_loaded)
        self.assertEqual(FakeInterrogator.instances, [], "no quiet retry on the CPU")
        # The half-loaded model is let go, so its VRAM can be handed back.
        self.assertIsNone(built_models[-1]())

    def test_a_cuda_startup_failure_still_falls_back_to_the_cpu(self):
        FakeInterrogator.cuda_error = lambda: RuntimeError(
            "CUDA error: no kernel image is available for execution on the device"
        )
        interrogator = CLIPInterrogator()
        with self._cuda_available():
            interrogator.load_model(mode="fast", device="cuda")

        self.assertTrue(interrogator.is_loaded)
        self.assertEqual(FakeInterrogator.instances[-1].config.device, "cpu")

    def test_clip_model_is_built_on_the_cpu_in_its_final_precision(self):
        for device, precision in (("cuda", "fp16"), ("cpu", "fp32")):
            with self.subTest(device=device), self._cuda_available():
                interrogator = CLIPInterrogator("ViT-bigG-14/laion2b_s39b_b160k")
                interrogator.load_model(mode="fast", device=device)
                config = FakeInterrogator.instances[-1].config

                # Built off the GPU, so only the cast weights ever reach it;
                # clip-interrogator moves the model to the device itself.
                model = config.clip_model
                self.assertEqual(model.name, "ViT-bigG-14")
                self.assertEqual(model.kwargs["pretrained"], "laion2b_s39b_b160k")
                self.assertEqual(model.kwargs["device"], "cpu")
                self.assertEqual(model.kwargs["precision"], precision)
                self.assertTrue(model.in_eval)
                self.assertEqual(config.clip_preprocess, "preprocess")
                self.assertEqual(config.device, device)

    def test_cached_vram_is_handed_back_after_load_and_on_unload(self):
        import torch

        for cuda_in_use in (True, False):
            with self.subTest(cuda_in_use=cuda_in_use), \
                    patch.object(torch.cuda, "is_available", return_value=cuda_in_use), \
                    patch.object(torch.cuda, "is_initialized", return_value=cuda_in_use), \
                    patch.object(torch.cuda, "empty_cache") as empty_cache:
                interrogator, _ci = self._load("best")
                after_load = empty_cache.call_count
                interrogator.unload_model()

                # Without CUDA in use there is nothing to empty, and emptying
                # would create a CUDA context.
                expected = (1, 2) if cuda_in_use else (0, 0)
                self.assertEqual((after_load, empty_cache.call_count), expected)

    def test_selected_caption_model_is_used(self):
        interrogator, ci = self._load("fast", caption_model="blip-base")
        result = interrogator.interrogate(self.image_path)

        self.assertEqual(ci.config.caption_model_name, "blip-base")
        self.assertEqual(ci.caption_calls, 1)
        self.assertTrue(result["raw_output"].startswith("a cat on a sofa"))
        self.assertEqual(interrogator.get_model_type(), "CLIP+blip-base")


class LibraryWarningTests(unittest.TestCase):
    def test_only_the_librarys_autocast_deprecation_is_silenced(self):
        # A fresh interpreter, as the app starts: unittest's runner puts its
        # own warning filter in front of the one under test.
        script = textwrap.dedent("""
            import json, warnings
            import torch
            import interrogators.clip_interrogator

            def autocast_called_from(module_name):
                # Warnings are attributed to the calling module.
                code = compile("torch.cuda.amp.autocast(enabled=False)", "caller.py", "exec")
                exec(code, {"__name__": module_name, "torch": torch})

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", append=True)
                autocast_called_from("clip_interrogator.clip_interrogator")
                autocast_called_from("some_other_module")
                warnings.warn("unrelated", FutureWarning)
            print(json.dumps([str(w.message) for w in caught]))
        """)
        env = {key: value for key, value in os.environ.items() if key != "PYTHONWARNINGS"}
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=PROJECT_ROOT, env=env,
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        messages = json.loads(result.stdout.strip().splitlines()[-1])

        self.assertEqual(len(messages), 2, messages)
        self.assertIn("torch.cuda.amp.autocast", messages[0])  # from some_other_module
        self.assertEqual(messages[1], "unrelated")


if __name__ == "__main__":
    unittest.main()
