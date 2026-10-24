# GPT-2 generation API (Railway / Docker)

This repository contains a small Flask API wrapping a local Hugging Face GPT-2
model. The original version exposed unauthenticated, unbounded generation and
used Flask's development server. The proposed hardening adds fail-closed API
authentication, input/token limits, bounded concurrency and a production WSGI
server without changing the `/generate` response shape.

## Deployment configuration (required before starting)

Set a **new, private** `APP_API_TOKEN` environment variable in your hosting
provider. It must be a printable non-whitespace ASCII secret of at least 32
characters. Generate one locally with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
Never commit it, paste it into issues or log it. **Without this variable the
service deliberately refuses to start.** Existing callers must be updated to
send an `Authorization: Bearer <token>` header. Treat the token as a secret;
this is suitable for trusted server-to-server clients, **not** a browser app
where the token would be visible to visitors.

The Docker image runs Gunicorn with one worker and two threads. The worker
loads `gpt2` at startup and serves `PORT` (default `5000`). Model weights are
downloaded from Hugging Face on first deployment unless already cached; allow
startup time, network access and memory for the model. The container runs as a
non-root user. Deploy only after testing in an isolated staging environment.

## API

- `GET /health`: readiness response once the model has loaded; no credentials.
- `POST /generate`: `Content-Type: application/json`, bearer authentication,
  body `{"prompt":"Your text"}`. Returns `{"text":"..."}` on success. The
  returned text includes the input prompt, as in the original implementation.
- Input body <= 8 KiB, prompt <= 2,000 characters and <= 256 model tokens;
  generated continuation <= 96 tokens. Extra request fields are rejected.
- Errors are JSON: `401 unauthorized`, `400 invalid_payload/invalid_prompt`,
  `413 request_too_large/prompt_too_long/prompt_tokens_exceeded`, `415
  unsupported_media_type`, `429 busy` with `Retry-After: 2`, or `503
  generation_unavailable`. No prompt or model output is included in error logs.

Example from a trusted machine (replace the placeholders; do not expose the
secret in shell history on a shared host):

```bash
curl -sS -X POST "http://127.0.0.1:5000/generate" \
  -H "Authorization: Bearer ${APP_API_TOKEN}" \
  -H 'Content-Type: application/json' \
  --data '{"prompt":"Once upon a time"}'
```

## Tests and limitations

```bash
python -m pip install 'Flask>=3.0,<4'
python -m compileall -q chatgpt-app
python -m unittest discover -s chatgpt-app/tests -v
```

Tests use a fake model and tokenizer; they do **not** download GPT-2. Run an
additional staging smoke test with actual model weights and representative
prompts before deploying. One worker means only one generation executes at a
time; excess concurrent requests receive `429` rather than accumulating model
work. This is a process-local guard, not distributed rate limiting. It does
not replace reverse-proxy rate limiting, TLS, request timeouts or observability.
The Gunicorn worker timeout is 120 seconds; generation may be interrupted if
model inference exceeds it. Never place a bearer token in frontend JavaScript.

### Rollback

Revert the PR or redeploy the previous image to restore the old API contract,
**but be aware that rollback re-exposes the unauthenticated/unbounded endpoint**.
Prefer keeping the hardened version and updating clients to use a bearer token.
No database or persistent data migration is performed.
