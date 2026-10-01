"""Images reach llama-server in a form its decoder accepts."""

import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from core.llama_cpp_runtime import LlamaCppRuntimeError
from interrogators import LlamaCppInterrogator

IMAGE_REJECTED = (
    'llama-server HTTP error 400: {"error":{"code":400,"message":"Failed to load image or audio file",'
    '"type":"invalid_request_error"}}'
)


def decode(data_url: str):
    header, payload = data_url.split(",", 1)
    return header[len("data:"):-len(";base64")], base64.b64decode(payload)


class ImagePayloadTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def save(self, name, image, fmt, **params):
        path = self.root / name
        image.save(path, fmt, **params)
        return path

    def test_formats_llama_reads_are_sent_byte_for_byte(self):
        rgb = Image.new("RGB", (24, 16), (200, 40, 40))
        for name, fmt, mime in (
            ("a.png", "PNG", "image/png"),
            ("a.jpg", "JPEG", "image/jpeg"),
            ("a.gif", "GIF", "image/gif"),
            ("a.bmp", "BMP", "image/bmp"),
        ):
            with self.subTest(fmt=fmt):
                path = self.save(name, rgb, fmt)
                sent_mime, sent = decode(LlamaCppInterrogator._encode_image_as_data_url(str(path)))
                self.assertEqual(sent_mime, mime)
                self.assertEqual(sent, path.read_bytes())

    def test_webp_is_converted(self):
        path = self.save("photo.webp", Image.new("RGB", (40, 30), (10, 120, 200)), "WEBP")
        mime, data = decode(LlamaCppInterrogator._encode_image_as_data_url(str(path)))
        self.assertEqual(mime, "image/jpeg")
        with Image.open(io.BytesIO(data)) as sent:
            self.assertEqual((sent.format, sent.size), ("JPEG", (40, 30)))

    def test_transparency_survives_conversion(self):
        path = self.save("sticker.webp", Image.new("RGBA", (20, 20), (0, 255, 0, 90)), "WEBP", lossless=True)
        mime, data = decode(LlamaCppInterrogator._encode_image_as_data_url(str(path)))
        self.assertEqual(mime, "image/png")
        with Image.open(io.BytesIO(data)) as sent:
            self.assertEqual(sent.mode, "RGBA")

    def test_format_is_judged_by_content_not_extension(self):
        path = self.save("misnamed.png", Image.new("RGB", (16, 16), (90, 90, 90)), "WEBP")
        mime, data = decode(LlamaCppInterrogator._encode_image_as_data_url(str(path)))
        self.assertEqual(mime, "image/jpeg")
        self.assertNotEqual(data, path.read_bytes())

    def test_exif_orientation_is_applied(self):
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90° clockwise to display
        path = self.save("phone.jpg", Image.new("RGB", (40, 20), (1, 2, 3)), "JPEG", exif=exif.tobytes())
        _, data = decode(LlamaCppInterrogator._encode_image_as_data_url(str(path)))
        with Image.open(io.BytesIO(data)) as sent:
            self.assertEqual(sent.size, (20, 40), "sent upright, as the gallery shows it")

    def test_unreadable_files_get_a_clear_error(self):
        path = self.root / "broken.png"
        path.write_bytes(b"this is not an image")
        with self.assertRaises(ValueError) as caught:
            LlamaCppInterrogator._encode_image_as_data_url(str(path))
        self.assertIn("broken.png", str(caught.exception))

    def test_forced_reencode(self):
        path = self.save("a.png", Image.new("RGB", (24, 16), (5, 5, 5)), "PNG")
        mime, data = decode(LlamaCppInterrogator._encode_image_as_data_url(str(path), reencode=True))
        self.assertEqual(mime, "image/jpeg")
        self.assertNotEqual(data, path.read_bytes())


class RejectedImageRetryTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.image = Path(self._tmp.name) / "a.png"
        Image.new("RGB", (24, 16), (200, 40, 40)).save(self.image)
        self.interrogator = LlamaCppInterrogator()
        self.interrogator.is_loaded = True
        self.interrogator.runtime = mock.Mock()

    def tearDown(self):
        self._tmp.cleanup()

    @staticmethod
    def sent_image(call):
        return call.kwargs["messages"][-1]["content"][1]["image_url"]["url"]

    def test_a_rejected_image_is_resent_converted_once(self):
        answer = {"choices": [{"message": {"content": json.dumps({"tags": ["red"], "comment": "Red."})}}]}
        self.interrogator.runtime.chat_completion.side_effect = [LlamaCppRuntimeError(IMAGE_REJECTED), answer]
        result = self.interrogator.interrogate(str(self.image), task="describe")
        self.assertEqual(result["tags"], ["red"])
        first, second = self.interrogator.runtime.chat_completion.call_args_list
        self.assertTrue(self.sent_image(first).startswith("data:image/png;"))
        self.assertTrue(self.sent_image(second).startswith("data:image/jpeg;"))

    def test_the_retry_happens_only_once(self):
        self.interrogator.runtime.chat_completion.side_effect = LlamaCppRuntimeError(IMAGE_REJECTED)
        with self.assertRaises(RuntimeError) as caught:
            self.interrogator.interrogate(str(self.image), task="describe")
        self.assertIn("Failed to load image", str(caught.exception))
        self.assertEqual(self.interrogator.runtime.chat_completion.call_count, 2)

    def test_other_bad_requests_are_not_image_errors(self):
        self.interrogator.runtime.chat_completion.side_effect = LlamaCppRuntimeError(
            "llama-server HTTP error 400: the request exceeds the available context size"
        )
        with self.assertRaises(RuntimeError):
            self.interrogator.interrogate(str(self.image), task="describe")
        self.assertEqual(self.interrogator.runtime.chat_completion.call_count, 1)


if __name__ == "__main__":
    unittest.main()
