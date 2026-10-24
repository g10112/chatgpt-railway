import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from security import (  # noqa: E402
    MAX_PROMPT_CHARS,
    MAX_PROMPT_TOKENS,
    RequestError,
    configured_token,
    is_authorized,
    validate_payload,
    validate_token_count,
)

TOKEN = "a" * 32


class SecurityPolicyTests(unittest.TestCase):
    def test_missing_secret_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                configured_token()

    def test_weak_secrets_rejected(self):
        for value in ["", "short", " " * 32, "a" * 31, "a" * 32 + "\n", "a" * 32 + "é"]:
            with self.subTest(value=repr(value)):
                with self.assertRaises(RuntimeError):
                    configured_token(value)

    def test_valid_secret(self):
        self.assertEqual(configured_token(TOKEN), TOKEN)

    def test_authentication(self):
        self.assertTrue(is_authorized("Bearer " + TOKEN, TOKEN))
        for value in [None, "", TOKEN, "bearer " + TOKEN, "Basic " + TOKEN, "Bearer " + TOKEN + " ", "Bearer " + "b" * 32, "Bearer " + TOKEN + ",Bearer b"]:
            with self.subTest(value=value):
                self.assertFalse(is_authorized(value, TOKEN))

    def test_payload_accepts_unicode_and_preserves_whitespace(self):
        self.assertEqual(validate_payload({"prompt": "  café\n"}), "  café\n")
        self.assertEqual(validate_payload({"prompt": "x" * MAX_PROMPT_CHARS}), "x" * MAX_PROMPT_CHARS)

    def test_payload_rejects_bad_shapes(self):
        for value in [None, [], "hello", {}, {"prompt": None}, {"prompt": 1}, {"prompt": False}, {"prompt": "\n  "}, {"prompt": "x", "max_new_tokens": 100000}]:
            with self.subTest(value=value), self.assertRaises(RequestError):
                validate_payload(value)

    def test_payload_rejects_oversize(self):
        with self.assertRaises(RequestError) as cm:
            validate_payload({"prompt": "x" * (MAX_PROMPT_CHARS + 1)})
        self.assertEqual((cm.exception.status, cm.exception.code), (413, "prompt_too_long"))

    def test_token_limits(self):
        for count in [1, MAX_PROMPT_TOKENS]:
            validate_token_count(count)
        for count in [0, -1, MAX_PROMPT_TOKENS + 1, True, 1.5]:
            with self.subTest(count=count), self.assertRaises(RequestError):
                validate_token_count(count)


if __name__ == "__main__":
    unittest.main()
