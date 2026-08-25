# Dwar

Stateless inference gateway for Dadi. It takes a request, calls a model provider, returns a normalized response, and keeps nothing.

Magaj uses the chat endpoints. Yaad uses `/v1/embed`. Dwar does not know what an agent is. The `system` string from Magaj is opaque: it is sent as the first system block, followed by Dwar's lane block.

Dwar is unauthenticated by design. It lives on a private mesh and must never be published to a host interface. Device auth will live in a Dadi-wide module later, not here.

## Routes

| Method | Path | Status |
| --- | --- | --- |
| `GET` | `/health` | live |
| `POST` | `/v1/chat/reasoning` | live |
| `POST` | `/v1/chat/conversation` | live |
| `POST` | `/v1/embed` | live |
| `POST` | `/v1/image/describe` | not built |
| `POST` | `/v1/image/create` | not built |
| `POST` | `/v1/stt/transcribe` | not built |

Unknown top-level fields are a 422. There is no model, temperature, or provider field on any request.

### Chat

Both chat endpoints share this body:

```json
{
  "system": "opaque agent system prompt",
  "messages": [
    {
      "role": "user",
      "content": "string or an array of blocks"
    }
  ],
  "tools": [
    {
      "name": "example",
      "description": "what it does",
      "input_schema": {
        "type": "object",
        "properties": {},
        "required": []
      }
    }
  ]
}
```

`tools` may be omitted or empty.

Message `content` is either a string or a list of blocks:

```json
{ "type": "text", "text": "..." }
{ "type": "tool_use", "id": "...", "name": "...", "input": {} }
{ "type": "tool_result", "tool_use_id": "...", "content": "...", "is_error": false }
```

`tool_use` is only valid on `assistant` messages. `tool_result` is only valid on `user` messages.

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

`stop_reason` is one of `end_turn`, `tool_use`, `max_tokens`, `error`. Thinking blocks are stripped. Magaj never sees them.

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

`embeddings[i]` matches `texts[i]`. `dimensions` is the width of the returned vectors so Yaad can check it against the pgvector column. An empty `texts` array is a 422. Batches larger than `embed.max_batch_size` or strings longer than `embed.max_text_length` (characters) are also 422. Nothing is truncated.

Configured model: OpenAI `text-embedding-3-small`, 1536 dimensions.

### Errors

```json
{ "error": { "type": "invalid_request", "message": "texts must not be empty" } }
```

Transport failures (429, 5xx, disconnects, timeouts) are retried with bounded backoff, then returned as HTTP errors. Never a partial 200.

## Config vs env

`config.toml` is checked in. It holds model IDs, token limits, thinking budget, retry, timeout, and embed caps. Change those in review, not per machine.

`.env` holds provider keys: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`. Host, port, and networks belong in `dadi/docker-compose.yml`, not here.

Reasoning uses Anthropic Claude Sonnet 5 with adaptive thinking. Conversation uses Gemini 3.6 Flash with thinking held to a minimum. Embed uses OpenAI as above.

## Run locally

Copy `.env.example` to `.env` and fill the three API keys. The process will not start if config.toml or those keys are missing.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app:app --reload --host 127.0.0.1 --port 8080
```

The image listens on container port 8080. Do not publish that port to the host. Compose will put Dwar on the internal network.

```sh
docker build -t dwar .
docker run --env-file .env dwar
```

`GET /health` returns `{"status":"ok"}`.
