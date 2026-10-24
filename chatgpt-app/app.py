"""Authenticated, bounded local GPT-2 text-generation API.

A single Gunicorn worker owns one model and one non-blocking generation slot.
Tests inject fake model/tokenizer objects and never download model weights.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from threading import BoundedSemaphore

from flask import Flask, jsonify, request

from security import (
    MAX_NEW_TOKENS,
    MAX_REQUEST_BYTES,
    RequestError,
    configured_token,
    is_authorized,
    validate_payload,
    validate_token_count,
)


def create_app(*, model=None, tokenizer=None, api_token: str | None = None) -> Flask:
    token = configured_token(api_token)
    if (model is None) != (tokenizer is None):
        raise ValueError("model and tokenizer must both be supplied or both omitted")

    injected_model = model is not None
    if model is None:
        # Load once at worker startup, not once per request. No model download in tests.
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        model = GPT2LMHeadModel.from_pretrained("gpt2")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model.eval()

    if injected_model:
        inference_context = nullcontext
    else:
        import torch
        inference_context = torch.inference_mode

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    generation_slot = BoundedSemaphore(value=1)

    @app.get("/health")
    def health():
        return jsonify({"status": "ready"})

    @app.post("/generate")
    def generate_text():
        # Authenticate before parsing, tokenization, or model execution.
        if not is_authorized(request.headers.get("Authorization"), token):
            return jsonify({"error": "unauthorized"}), 401
        if request.mimetype != "application/json":
            return jsonify({"error": "unsupported_media_type"}), 415
        try:
            prompt = validate_payload(request.get_json(silent=True))
        except RequestError as exc:
            return jsonify({"error": exc.code}), exc.status

        if not generation_slot.acquire(blocking=False):
            return jsonify({"error": "busy"}), 429, {"Retry-After": "2"}
        try:
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            validate_token_count(int(inputs.shape[-1]))
            # Inference mode avoids autograd memory growth. Injected fake models
            # bypass Torch entirely so the HTTP suite stays lightweight.
            with inference_context():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return jsonify({"text": result})
        except RequestError as exc:
            return jsonify({"error": exc.code}), exc.status
        except Exception as exc:
            # Never log prompt text, model output, or exception messages.
            app.logger.error("generation failed (%s)", type(exc).__name__)
            return jsonify({"error": "generation_unavailable"}), 503
        finally:
            generation_slot.release()

    @app.errorhandler(413)
    def too_large(_exc):
        return jsonify({"error": "request_too_large"}), 413

    return app


if __name__ == "__main__":
    # Local development only; container execution uses Gunicorn.
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")))
