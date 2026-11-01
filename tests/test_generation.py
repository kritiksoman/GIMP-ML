import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import requests
from PIL import Image

from gimpml import providers, worker
from gimpml.settings import GenerationError, GenerationRequest, api_key, preferences


def encoded_image(mode="RGB", format="PNG"):
    output = io.BytesIO()
    Image.new(mode, (8, 4)).save(output, format=format)
    return base64.b64encode(output.getvalue()).decode()


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.saved_key = patch("gimpml.credentials.read_key", return_value=None)
        self.saved_key.start()
        self.addCleanup(self.saved_key.stop)

    def test_valid_defaults(self):
        GenerationRequest("a landscape").validate()

    def test_invalid_requests(self):
        for fields in ({"prompt": ""}, {"prompt": "x" * 16001},
                       {"provider": "openai"}, {"model": "gemini-chat"},
                       {"resolution": "8K"}, {"aspect_ratio": "banana"},
                       {"resolution": "4K"}):
            with self.subTest(fields=fields), self.assertRaises(GenerationError):
                GenerationRequest(**{"prompt": "a landscape", **fields}).validate()

    def test_preferences(self):
        self.assertEqual(preferences()["provider"], "openrouter")

    def test_bad_preferences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gimpml").mkdir()
            for contents in ("invalid", "[]", "{}", json.dumps({
                **preferences(), "timeout_seconds": True
            })):
                (root / "gimpml/config.json").write_text(contents)
                with self.subTest(contents=contents), self.assertRaises(GenerationError):
                    preferences(root)

    def test_dotenv_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            (root / ".env").write_text(
                "OPENROUTER_API_KEY=local-key\nGOOGLE_API_KEY=google-key\n")
            self.assertEqual(api_key("openrouter", root), "local-key")
            self.assertEqual(api_key("google", root), "google-key")
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "environment-key"}):
                self.assertEqual(api_key("openrouter", root), "environment-key")

    def test_missing_key(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(GenerationError, "OPENROUTER_API_KEY"):
                api_key("openrouter", Path(directory))

    def test_keychain_precedence_and_google_environment_alias(self):
        with (tempfile.TemporaryDirectory() as directory,
              patch.dict(os.environ, {}, clear=True),
              patch("gimpml.credentials.read_key", return_value="saved-key") as read):
            root = Path(directory)
            (root / ".env").write_text("GEMINI_API_KEY=file-key\nOPENROUTER_API_KEY=file-key\n")
            self.assertEqual(api_key("google", root), "saved-key")
            self.assertEqual(api_key("openrouter", root), "saved-key")
            read.reset_mock()
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "environment-key"}):
                self.assertEqual(api_key("google", root), "environment-key")
            read.assert_not_called()

    def test_keychain_failure_does_not_fall_back_to_file(self):
        with (tempfile.TemporaryDirectory() as directory,
              patch.dict(os.environ, {}, clear=True),
              patch("gimpml.credentials.read_key", side_effect=GenerationError("Unlock Keychain."))):
            root = Path(directory)
            (root / ".env").write_text("OPENROUTER_API_KEY=file-key\n")
            with self.assertRaisesRegex(GenerationError, "Unlock Keychain"):
                api_key("openrouter", root)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.key = patch("gimpml.providers.api_key", return_value="secret-test-key")
        self.key.start()
        self.addCleanup(self.key.stop)

    @patch("gimpml.providers.post_json")
    def test_openrouter_request_and_result(self, post):
        post.return_value = {"data": [{"b64_json": encoded_image(), "media_type": "image/png"}]}
        result = providers.generate(GenerationRequest("a landscape"))
        self.assertEqual((result.width, result.height), (8, 4))
        self.assertTrue(result.png.startswith(b"\x89PNG"))
        url, headers, payload, timeout = post.call_args.args
        self.assertEqual(url, "https://openrouter.ai/api/v1/images")
        self.assertEqual(headers["Authorization"], "Bearer secret-test-key")
        self.assertEqual(payload, {"model": "google/gemini-2.5-flash-image",
                                  "prompt": "a landscape", "aspect_ratio": "1:1",
                                  "n": 1})
        self.assertEqual(timeout, 180)

    @patch("gimpml.providers.post_json")
    def test_google_request_and_skips_thought_image(self, post):
        post.return_value = {"candidates": [{"content": {"parts": [
            {"thought": True, "inlineData": {"data": "invalid", "mimeType": "image/png"}},
            {"text": "Here is your image"},
            {"inlineData": {"data": encoded_image(), "mimeType": "image/png"}}
        ]}}]}
        result = providers.generate(GenerationRequest("a landscape", provider="google"))
        self.assertEqual(result.width, 8)
        url, headers, payload, _ = post.call_args.args
        self.assertTrue(url.endswith("/gemini-2.5-flash-image:generateContent"))
        self.assertEqual(headers["x-goog-api-key"], "secret-test-key")
        self.assertEqual(payload["generationConfig"]["imageConfig"], {"aspectRatio": "1:1"})

    @patch("gimpml.providers.post_json")
    def test_google_high_resolution(self, post):
        post.return_value = {"candidates": [{"content": {"parts": [
            {"inlineData": {"data": encoded_image(), "mimeType": "image/png"}}
        ]}}]}
        providers.generate(GenerationRequest("test", provider="google",
                                            model="gemini-3.1-flash-image", resolution="4K"))
        self.assertEqual(post.call_args.args[2]["generationConfig"]["imageConfig"]["imageSize"], "4K")

    @patch("gimpml.providers.post_json")
    def test_editing_reference_on_both_providers(self, post):
        reference = base64.b64decode(encoded_image("RGBA"))
        for provider in ("openrouter", "google"):
            with self.subTest(provider=provider):
                post.return_value = (
                    {"data": [{"b64_json": encoded_image(), "media_type": "image/png"}]}
                    if provider == "openrouter" else
                    {"candidates": [{"content": {"parts": [
                        {"inlineData": {"data": encoded_image(), "mimeType": "image/png"}}
                    ]}}]}
                )
                result = providers.generate(
                    GenerationRequest("make it blue", provider=provider), reference_png=reference)
                self.assertEqual((result.width, result.height), (8, 4))
                payload = post.call_args.args[2]
                if provider == "openrouter":
                    inputs = payload["input_references"]
                    self.assertEqual(len(inputs), 1)
                    self.assertEqual(inputs[0]["type"], "image_url")
                    data_url = inputs[0]["image_url"]["url"]
                    self.assertTrue(data_url.startswith("data:image/png;base64,"))
                    encoded = data_url.split(",", 1)[1]
                else:
                    parts = payload["contents"][0]["parts"]
                    self.assertEqual(parts[0], {"text": "make it blue"})
                    self.assertEqual(parts[1]["inlineData"]["mimeType"], "image/png")
                    encoded = parts[1]["inlineData"]["data"]
                with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
                    self.assertEqual(image.mode, "RGBA")
                    self.assertEqual(image.size, (8, 4))
                    self.assertEqual(image.info, {})

    @patch("gimpml.providers.post_json")
    def test_invalid_editing_inputs_never_contact_provider(self, post):
        for reference in (b"", b"not png", b"\x89PNG\r\n\x1a\ncorrupt",
                          base64.b64decode(encoded_image(format="JPEG")), "not bytes"):
            with self.subTest(reference=reference), self.assertRaises(GenerationError):
                providers.generate(GenerationRequest("edit"), reference_png=reference)
        reference = base64.b64decode(encoded_image())
        with patch("gimpml.providers.MAX_REFERENCE_BYTES", 8), self.assertRaises(GenerationError):
            providers.generate(GenerationRequest("edit"), reference_png=reference)
        with patch("gimpml.providers.MAX_PIXELS", 8), self.assertRaises(GenerationError):
            providers.generate(GenerationRequest("edit"), reference_png=reference)
        post.assert_not_called()

    def test_editing_input_limits_include_exact_boundary(self):
        reference = base64.b64decode(encoded_image("RGBA"))
        with patch("gimpml.providers.MAX_REFERENCE_BYTES", len(reference)):
            self.assertTrue(providers.reference_data(reference))
        with (patch("gimpml.providers.MAX_REFERENCE_BYTES", len(reference) - 1),
              self.assertRaises(GenerationError)):
            providers.reference_data(reference)
        with patch("gimpml.providers.MAX_PIXELS", 32):
            self.assertTrue(providers.reference_data(reference))
        with patch("gimpml.providers.MAX_PIXELS", 31), self.assertRaises(GenerationError):
            providers.reference_data(reference)

    @patch("gimpml.providers.post_json")
    def test_text_only_google_does_not_upload_image(self, post):
        post.return_value = {"candidates": [{"content": {"parts": [
            {"inlineData": {"data": encoded_image(), "mimeType": "image/png"}}
        ]}}]}
        providers.generate(GenerationRequest("test", provider="google"))
        self.assertEqual(post.call_args.args[2]["contents"], [{"parts": [{"text": "test"}]}])

    @patch("gimpml.providers.post_json")
    def test_no_images_and_blocked_prompts(self, post):
        for provider, response in (
            ("openrouter", {}), ("openrouter", {"data": []}),
            ("openrouter", {"data": [{"url": "https://example.invalid"}]}),
            ("google", {"promptFeedback": {"blockReason": "SAFETY"}}),
            ("google", {"candidates": [{"content": {"parts": [{"text": "Sorry"}]}}]}),
            ("google", {"candidates": [{"content": None}]}),
        ):
            with self.subTest(provider=provider, response=response):
                post.return_value = response
                with self.assertRaises(GenerationError):
                    providers.generate(GenerationRequest("test", provider=provider))

    def test_invalid_image_data(self):
        for encoded, mime in (("not-base64", "image/png"),
                              (base64.b64encode(b"not an image").decode(), "image/png"),
                              (encoded_image(), "image/svg+xml")):
            with self.subTest(mime=mime), self.assertRaises(GenerationError):
                providers.decode_image(encoded, mime)

    def test_jpeg_normalized_to_rgba_png(self):
        result = providers.decode_image(encoded_image(format="JPEG"), "image/jpeg")
        with Image.open(io.BytesIO(result.png)) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.format, "PNG")

    def test_pixel_limit(self):
        with patch("gimpml.providers.MAX_PIXELS", 8), self.assertRaises(GenerationError):
            providers.decode_image(encoded_image(), "image/png")

    @patch("gimpml.providers.requests.post")
    def test_http_errors_are_sanitized_and_not_retried(self, post):
        for status in (400, 401, 402, 403, 404, 429, 500, 302):
            with self.subTest(status=status):
                post.reset_mock()
                response = post.return_value.__enter__.return_value
                response.status_code = status
                response.text = "secret-test-key sensitive-provider-body"
                with self.assertRaises(GenerationError) as error:
                    providers.post_json("https://example.invalid", {}, {}, 180)
                self.assertNotIn("secret", str(error.exception))
                post.assert_called_once()
                self.assertFalse(post.call_args.kwargs["allow_redirects"])

    @patch("gimpml.providers.requests.post")
    def test_network_errors_sanitized(self, post):
        for exception in (requests.Timeout("secret-key"), requests.ConnectionError("secret-key")):
            post.side_effect = exception
            with self.assertRaises(GenerationError) as error:
                providers.post_json("https://example.invalid", {}, {}, 180)
            self.assertNotIn("secret-key", str(error.exception))

    @patch("gimpml.providers.requests.post")
    def test_json_and_response_limits(self, post):
        response = post.return_value.__enter__.return_value
        response.status_code = 200
        for body in (b"not-json", b"[]"):
            response.iter_content.return_value = [body]
            with self.assertRaises(GenerationError):
                providers.post_json("https://example.invalid", {}, {}, 180)
        response.iter_content.return_value = [b"12345"]
        with patch("gimpml.providers.MAX_RESPONSE_BYTES", 4), self.assertRaises(GenerationError):
            providers.post_json("https://example.invalid", {}, {}, 180)
        response.iter_content.return_value = [b'{"data": []}']
        self.assertEqual(providers.post_json("https://example.invalid", {}, {}, 180), {"data": []})


class WorkerTests(unittest.TestCase):
    @patch("gimpml.worker.generate")
    def test_worker_editing_input_transport(self, generate):
        generate.return_value = providers.decode_image(encoded_image(), "image/png")
        reference = base64.b64decode(encoded_image("RGBA"))
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.png"
            source.write_bytes(reference)
            output = Path(directory) / "generated.png"
            payload = json.dumps({"request": {"prompt": "edit"},
                                  "input": str(source), "output": str(output)})
            with patch("sys.stdin", io.StringIO(payload)), patch("sys.stdout", io.StringIO()):
                self.assertEqual(worker.main(), 0)
            self.assertEqual(generate.call_args.kwargs["reference_png"], reference)
            self.assertEqual(source.read_bytes(), reference)
            self.assertTrue(output.is_file())

    @patch("gimpml.worker.generate")
    def test_worker_invalid_editing_path_and_size(self, generate):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.png"
            source.write_bytes(b"too large")
            for input_path in ("relative.png", str(source.with_suffix(".jpg")),
                               str(source.with_name("missing.png")), str(source), None):
                payload = json.dumps({"request": {"prompt": "edit"}, "input": input_path,
                                      "output": str(Path(directory) / "generated.png")})
                with (self.subTest(input_path=input_path),
                      patch("gimpml.worker.MAX_REFERENCE_BYTES", 4),
                      patch("sys.stdin", io.StringIO(payload)),
                      patch("sys.stdout", io.StringIO())):
                    self.assertEqual(worker.main(), 1)
            generate.assert_not_called()

    @patch("gimpml.worker.generate")
    def test_worker_output(self, generate):
        generate.return_value = providers.decode_image(encoded_image(), "image/png")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated.png"
            payload = json.dumps({"request": {"prompt": "test"}, "output": str(output)})
            stdout = io.StringIO()
            with patch("sys.stdin", io.StringIO(payload)), patch("sys.stdout", stdout):
                self.assertEqual(worker.main(), 0)
            self.assertEqual(json.loads(stdout.getvalue()),
                             {"status": "success", "width": 8, "height": 4})
            self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))

    @patch("gimpml.worker.generate")
    def test_worker_does_not_overwrite(self, generate):
        generate.return_value = providers.decode_image(encoded_image(), "image/png")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "generated.png"
            output.write_bytes(b"existing")
            payload = json.dumps({"request": {"prompt": "test"}, "output": str(output)})
            with patch("sys.stdin", io.StringIO(payload)), patch("sys.stdout", io.StringIO()):
                self.assertEqual(worker.main(), 1)
            self.assertEqual(output.read_bytes(), b"existing")
            generate.assert_not_called()

    def test_worker_invalid_input(self):
        for payload in ("bad-json", "[]", "{}", json.dumps({
            "request": {"prompt": "test"}, "output": "relative.png"
        })):
            stdout = io.StringIO()
            with patch("sys.stdin", io.StringIO(payload)), patch("sys.stdout", stdout):
                self.assertEqual(worker.main(), 1)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "error")

    @patch("gimpml.worker.generate", side_effect=GenerationError("Insufficient provider credits."))
    def test_worker_provider_failure(self, _generate):
        with tempfile.TemporaryDirectory() as directory:
            payload = json.dumps({"request": {"prompt": "test"},
                                  "output": str(Path(directory) / "generated.png")})
            stdout = io.StringIO()
            with patch("sys.stdin", io.StringIO(payload)), patch("sys.stdout", stdout):
                self.assertEqual(worker.main(), 1)
            self.assertEqual(json.loads(stdout.getvalue())["message"], "Insufficient provider credits.")


if __name__ == "__main__":
    unittest.main()
