# Dwar

Stateless inference gateway for dadi. It takes a request, calls a model provider, returns a normalized response, and keeps nothing. It lives on the private mesh and must never be published to a host interface.

## Dependencies

- Provider APIs: Anthropic, Gemini, OpenAI, Deepgram (keys in `.env`)
- Nas for mesh DNS (`dwar.dadi`), compose/prod networking, and the shared logging contract

Dwar is unauthenticated by design. Device auth is not this module’s job.

## Layout

```
dwar/
  app.py              ASGI app, middleware, exception handlers
  config.py           config.toml + env
  logutil.py          JSON logging (nas contract)
  errors.py           DwarError / TransportError
  lanes.py            lane prompt loaders
  inference/          provider adapters
  routers/v1/         HTTP routes + schemas
  prompts/            lane prompt text
  tests/              pytest behavior suite
  config.toml
```

## Config vs env

`config.toml` (checked in) holds model IDs, token limits, thinking budget, size caps, retry, and timeout.

`.env` holds provider keys only: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`. Host/port come from Nas.

Dwar boots with empty keys allowed. Each route requires only its own key and returns `503` with `error.type` = `provider_unconfigured` if that key is unset. `config.toml` must be present and valid — fail at startup if missing or invalid.

## Local run

```sh
cp .env.example .env
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app:app --reload --host 127.0.0.1 --port 8080
pytest
```

Container listens on `8080` (internal). Do not publish that port to the host.

## CI / CD

| Workflow | When | What |
| --- | --- | --- |
| `ci.yml` → `ci` | PR + push to `main` | `pytest`, build `dev` + `production` images |
| `ci.yml` → `publish` | `main` after `ci` | Push `ghcr.io/<owner>/dwar:{latest,sha}` |

Concurrency cancels superseded runs on the same ref.

## Logging / error codes

Logs follow the nas JSON contract (`time`, `level`, `service=dwar`, `msg`, plus `code`, `request_id`, request summary fields). Uvicorn access logs are disabled; one structured request line per call is emitted instead.

HTTP errors: `{ "error": { "type": "<code>", "message": "..." } }`.

| Code | When |
| --- | --- |
| `invalid_request` | Validation / bad input (422) |
| `provider_unconfigured` | Required provider key missing (503) |
| `upstream_timeout` | Provider timeout after retries |
| `upstream_unreachable` | Provider connection failure after retries |
| `rate_limit` | Provider 429 after retries |
| `provider` | Non-retryable provider error |

See nas README for the shared infra catalog.

## Routes

| Method | Path | Model |
| --- | --- | --- |
| `GET` | `/health` | |
| `POST` | `/chat/reasoning` | Anthropic (config.toml) |
| `POST` | `/chat/conversation` | Gemini |
| `POST` | `/chat/complete` | Gemini |
| `POST` | `/embed` | OpenAI embeddings |
| `POST` | `/image/describe` | Gemini |
| `POST` | `/image/create` | Gemini image |
| `POST` | `/speech/transcribe` | Deepgram |

Unknown top-level fields are a 422. There is no model, temperature, or provider field on any request.

### Chat

`/chat/reasoning` and `/chat/conversation` prepend a Dwar lane prompt to `system`. `/chat/complete` does not — the caller's `system` is the whole instruction.

Chat endpoints share this body:

```json
{
  "system": "opaque agent system prompt",
  "messages": [
    { "role": "user", "content": "string or an array of blocks" }
  ],
  "tools": [
    {
      "name": "example",
      "description": "what it does",
      "input_schema": { "type": "object", "properties": {}, "required": [] }
    }
  ]
}
```

`tools` may be omitted or empty. Message `content` is a string or a list of `text` / `tool_use` / `tool_result` blocks. `tool_use` only on `assistant`; `tool_result` only on `user`.

Response:

```json
{
  "content": [
    { "type": "text", "text": "..." },
    { "type": "tool_use", "id": "...", "name": "...", "input": {} }
  ],
  "stop_reason": "end_turn",
  "usage": { "input_tokens": 0, "output_tokens": 0 }
}
```

`stop_reason`: `end_turn` | `tool_use` | `max_tokens` | `error`. Thinking blocks are stripped. Non-empty `tools` forces at least one tool call.

### Embed

```json
{ "texts": ["string", "..."] }
```

```json
{
  "embeddings": [[0.0]],
  "dimensions": 1536,
  "usage": { "input_tokens": 0 }
}
```

Empty `texts`, oversized batches, or overlong strings are 422. Nothing is truncated.

### Image describe / create / speech

Describe: `{ "image": { "media_type", "data" }, "prompt"? }` → `{ "description", "usage" }`.

Create: `{ "prompt" }` → `{ "image": { "media_type", "data" }, "usage" }`.

Transcribe: `{ "audio": { "media_type", "data" } }` → `{ "text", "duration_seconds" }`.

Invalid media, empty/base64 failures, or size caps are 422.

Transport failures are retried with bounded backoff, then returned as HTTP errors — never a partial 200.
