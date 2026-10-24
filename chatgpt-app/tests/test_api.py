"""HTTP-level tests use Flask and fake model objects, never download GPT-2."""

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import flask  # noqa: F401
except ImportError:
    create_app = None
else:
    from app import create_app

TOKEN = "z" * 40
HEADERS = {"Authorization": "Bearer " + TOKEN}


class FakeTokens:
    def __init__(self, length):
        self.shape = (1, length)


class FakeTokenizer:
    eos_token_id = 50256

    def encode(self, prompt, return_tensors):
        assert return_tensors == "pt"
        return FakeTokens(len(prompt.split()))

    def decode(self, tokens, skip_special_tokens):
        assert skip_special_tokens
        return "generated result"


class FakeModel:
    def __init__(self):
        self.calls = 0
        self.fail = False
        self.wait = None
        self.entered = None

    def generate(self, inputs, **kwargs):
        self.calls += 1
        assert kwargs["max_new_tokens"] == 96
        assert kwargs["do_sample"] is False
        assert kwargs["pad_token_id"] == 50256
        if self.entered:
            self.entered.set()
        if self.wait:
            self.wait.wait(timeout=5)
        if self.fail:
            raise RuntimeError("sensitive input must not leak")
        return [[1, 2, 3]]


@unittest.skipUnless(create_app is not None, "Flask not installed; run CI with Flask dependency")
class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.app = create_app(model=self.model, tokenizer=FakeTokenizer(), api_token=TOKEN)
        self.client = self.app.test_client()

    def test_health_and_authenticated_generation(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        response = self.client.post("/generate", headers=HEADERS, json={"prompt": "hello world"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"text": "generated result"})
        self.assertEqual(self.model.calls, 1)

    def test_unauthorized_never_invokes_model(self):
        for headers in [{}, {"Authorization": "Bearer wrong"}]:
            response = self.client.post("/generate", headers=headers, json={"prompt": "hello"})
            self.assertEqual(response.status_code, 401)
        self.assertEqual(self.model.calls, 0)

    def test_bad_json_and_media_type(self):
        self.assertEqual(self.client.post("/generate", headers=HEADERS, data="abc", content_type="application/json").status_code, 400)
        self.assertEqual(self.client.post("/generate", headers=HEADERS, data="abc", content_type="text/plain").status_code, 415)
        self.assertEqual(self.model.calls, 0)

    def test_body_limit(self):
        response = self.client.post("/generate", headers=HEADERS, data=b"x" * 9000, content_type="application/json")
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.model.calls, 0)

    def test_token_limit(self):
        response = self.client.post("/generate", headers=HEADERS, json={"prompt": "word " * 257})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json["error"], "prompt_tokens_exceeded")
        self.assertEqual(self.model.calls, 0)

    def test_model_failure_does_not_expose_exception(self):
        self.model.fail = True
        with patch.object(self.app.logger, "error") as log:
            response = self.client.post("/generate", headers=HEADERS, json={"prompt": "private"})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("sensitive", str(response.json))
        self.assertNotIn("private", str(log.call_args))

    def test_second_concurrent_request_is_rejected(self):
        self.model.entered = threading.Event()
        self.model.wait = threading.Event()
        results = []

        def first_request():
            with self.app.test_client() as client:
                results.append(client.post("/generate", headers=HEADERS, json={"prompt": "one"}).status_code)

        thread = threading.Thread(target=first_request)
        thread.start()
        try:
            self.assertTrue(self.model.entered.wait(timeout=3))
            response = self.client.post("/generate", headers=HEADERS, json={"prompt": "two"})
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers.get("Retry-After"), "2")
        finally:
            self.model.wait.set()
            thread.join(timeout=5)
        self.assertEqual(results, [200])
        self.assertEqual(self.model.calls, 1)


if __name__ == "__main__":
    unittest.main()
