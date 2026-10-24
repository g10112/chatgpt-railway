"""Pure request policy for the local GPT-2 generation service.

This module deliberately has no Flask, Torch, or network dependency, so security
boundaries can be tested without downloading a language model.
"""

from __future__ import annotations

import hmac
import os
from dataclasses import dataclass


MAX_REQUEST_BYTES = 8 * 1024
MAX_PROMPT_CHARS = 2_000
MAX_PROMPT_TOKENS = 256
MAX_NEW_TOKENS = 96
MIN_API_TOKEN_CHARS = 32


@dataclass(frozen=True)
class RequestError(Exception):
    status: int
    code: str


def configured_token(explicit: str | None = None) -> str:
    """Fail startup closed if the operator has not provisioned a strong token."""
    token = explicit if explicit is not None else os.environ.get("APP_API_TOKEN", "")
    if len(token) < MIN_API_TOKEN_CHARS or token != token.strip() or any(
        ord(character) < 33 or ord(character) > 126 for character in token
    ):
        raise RuntimeError(
            "APP_API_TOKEN must be a non-whitespace ASCII secret of at least 32 characters"
        )
    return token


def is_authorized(authorization: str | None, token: str) -> bool:
    """Accept only one exact Bearer token; compare secrets in constant time."""
    if not isinstance(authorization, str) or not authorization.startswith("Bearer "):
        return False
    supplied = authorization[len("Bearer ") :]
    if not supplied or supplied != supplied.strip() or " " in supplied:
        return False
    return hmac.compare_digest(supplied, token)


def validate_payload(value: object) -> str:
    """Reject malformed or excessive prompts before tokenizer/model execution."""
    if not isinstance(value, dict) or set(value) != {"prompt"}:
        raise RequestError(400, "invalid_payload")
    prompt = value["prompt"]
    if not isinstance(prompt, str) or not prompt.strip():
        raise RequestError(400, "invalid_prompt")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise RequestError(413, "prompt_too_long")
    return prompt


def validate_token_count(count: int) -> None:
    if not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= MAX_PROMPT_TOKENS:
        raise RequestError(413, "prompt_tokens_exceeded")
