import io
import json
import sys
import unittest
from unittest.mock import MagicMock, patch

from gimpml import credentials
from gimpml.settings import GenerationError


@unittest.skipUnless(sys.platform == "darwin", "macOS Keychain integration")
class CredentialTests(unittest.TestCase):
    def setUp(self):
        self.store = MagicMock()
        self.values = {}
        self.store.get_password.side_effect = lambda service, provider: self.values.get(provider)
        self.store.set_password.side_effect = lambda service, provider, key: self.values.update({provider: key})
        self.store.delete_password.side_effect = lambda service, provider: self.values.pop(provider)
        backend = patch("gimpml.credentials.backend", return_value=self.store)
        backend.start()
        self.addCleanup(backend.stop)
        platform = patch("gimpml.credentials.sys.platform", "darwin")
        platform.start()
        self.addCleanup(platform.stop)

    def test_save_read_keep_and_remove(self):
        self.assertEqual(credentials.saved_status(), {"openrouter": False, "google": False})
        self.assertEqual(credentials.save_keys({"openrouter": "  test-key  "}, []),
                         {"openrouter": True, "google": False})
        self.assertEqual(credentials.read_key("openrouter"), "test-key")
        credentials.save_keys({"google": "google-key"}, [])
        credentials.save_keys({}, [])
        self.assertEqual(self.values, {"openrouter": "test-key", "google": "google-key"})
        self.assertEqual(credentials.save_keys({}, ["openrouter"]),
                         {"openrouter": False, "google": True})
        credentials.save_keys({}, ["openrouter"])
        self.store.set_password.assert_any_call(credentials.SERVICE, "openrouter", "test-key")

    def test_validation_before_any_write(self):
        for keys, clear in (({"other": "key"}, []), ({"google": ""}, []),
                            ({"google": "x" * 4097}, []), ({"google": 12}, []),
                            ({"google": "key"}, ["google"]), ({}, ["other"]),
                            ([], []), ({}, "google")):
            with self.subTest(keys=keys, clear=clear), self.assertRaises(GenerationError):
                credentials.save_keys(keys, clear)
        self.store.set_password.assert_not_called()
        self.store.delete_password.assert_not_called()
        credentials.save_keys({"google": "x" * 4096}, [])

    def test_errors_do_not_expose_keys(self):
        from keyring.errors import KeyringError

        self.store.set_password.side_effect = KeyringError("private-secret")
        with self.assertRaises(GenerationError) as error:
            credentials.save_keys({"google": "private-secret"}, [])
        self.assertNotIn("private-secret", str(error.exception))
        self.assertIn("Some changes may have been saved", str(error.exception))
        self.store.get_password.side_effect = KeyringError("private-secret")
        with self.assertRaises(GenerationError) as error:
            credentials.read_key("google")
        self.assertNotIn("private-secret", str(error.exception))

    def test_worker_never_returns_secrets(self):
        payload = {"operation": "save", "keys": {"google": "private-secret"}, "clear": []}
        output = io.StringIO()
        with patch("sys.stdin", io.StringIO(json.dumps(payload))), patch("sys.stdout", output):
            self.assertEqual(credentials.main(), 0)
        self.assertNotIn("private-secret", output.getvalue())
        self.assertEqual(json.loads(output.getvalue()),
                         {"status": "success", "saved": {"openrouter": False, "google": True}})

    def test_worker_invalid_input(self):
        for payload in ("not json", "[]", "{}", '{"operation":"status","extra":true}',
                        '{"operation":"save","keys":null,"clear":[]}'):
            output = io.StringIO()
            with (self.subTest(payload=payload), patch("sys.stdin", io.StringIO(payload)),
                  patch("sys.stdout", output)):
                self.assertEqual(credentials.main(), 1)
            self.assertEqual(json.loads(output.getvalue())["status"], "error")

    def test_non_macos_read_does_not_use_keychain(self):
        with patch("gimpml.credentials.sys.platform", "linux"):
            self.assertIsNone(credentials.read_key("google"))


if __name__ == "__main__":
    unittest.main()
