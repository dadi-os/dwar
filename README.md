# Dwar

Stateless inference gateway for dadi. It takes a request, calls a model provider, returns a normalized response, and keeps nothing. Hath calls `/chat/agent` for an agent's tool loop and `/chat/complete` for context-free calls (its router, one-shots); Nas helpers and Yaad call the rest. It lives on the private mesh and must never be published to a host interface.

## Dependencies

- Provider APIs: Anthropic, Gemini, OpenAI, Deepgram (keys in `.env`)
- Nas for mesh DNS (`dwar.dadi`), compose/prod networking, and the shared logging contract

Dwar is unauthenticated by design. Device auth is not this module’s job.

## Layout

```
dwar/
  app.py              ASGI app, middleware, exception handlers
  config.py           models, budgets and limits in code; keys from env
  logutil.py          JSON logging (nas contract)
  errors.py           DwarError / TransportError
  prompts.py          prompt loaders (agent prompt, image describe instruction)
  inference/          provider adapters
  routers/v1/         HTTP routes + schemas
  prompts/            prompt text (agent.txt, describe.txt)
  tests/              pytest behavior suite
```

The tool names in `prompts/agent.txt` are those of hath's agent tool set (`hath/src/runtime/tool-set.ts`); hath's own tests check its prompts, so keep this file in step by hand when a tool is renamed.

## Config vs env

`config.py` holds every model ID (chat endpoints `chat.agent` and `chat.complete`), token limit, Anthropic effort level, size cap, retry and timeout as constants. Changing a model or an effort is a push to Dwar; nothing on the box overrides them.

`.env` holds provider keys only: `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`. Host/port come from Nas.

Dwar boots with empty keys allowed. Each route requires only its own key and returns `503` with `error.type` = `provider_unconfigured` if that key is unset.

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

Logs follow the nas JSON contract (`time`, `level`, `service=dwar`, `msg`, plus `code`, `request_id`, request summary fields). Uvicorn access logs and the provider SDKs' per-call info lines (`httpx`, `google_genai`) are off; one structured `request` line per call is emitted instead (healthy `/health` probes are not logged).

Every served inference call also emits one `inference` line: `caller`, `route` (`chat.agent`, `chat.complete`, `embed`, `image.describe`, `image.create`, `speech.transcribe`), `provider`, `model`, `duration_ms`, `request_id`, and the call's cost fields (`input_tokens`, `output_tokens`, `cache_read_input_tokens`, `cache_creation_input_tokens`, `stop_reason`; `audio_seconds` for speech). Prompt and response content are never logged. Dwar stays stateless — cost reporting is a LogQL query over these lines in Loki.

Every route except `/health` requires `X-Dadi-Caller` naming the service and actor that pays for the call (`hath/<agent id>`, `hath/router`, `yaad/ingest`, …); missing or blank is `invalid_request` (422).

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
| `POST` | `/chat/agent` | Claude Haiku 5.5, effort `xhigh` |
| `POST` | `/chat/complete` | Claude Haiku 5.5, effort `high` |
| `POST` | `/embed` | OpenAI text-embedding-3-small |
| `POST` | `/image/describe` | Gemini 3.6 Flash |
| `POST` | `/image/create` | Gemini 3.1 Flash Image |
| `POST` | `/speech/transcribe` | Deepgram |
| `POST` | `/web/search` | Claude Haiku 5.5 with web search (Hath's web tools) |
| `POST` | `/web/fetch` | Fetches public pages as text (Hath's web tools) |

Unknown top-level fields are a 422. There is no model, temperature, or provider field on any request.

### Chat

`/chat/agent` prepends Dwar's agent prompt (`prompts/agent.txt`) to `system`: the agent's loop does the job with its tools and sends its own messages. `/chat/complete` does not — the caller's `system` is the whole instruction.

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
  ],
  "tool_choice": "auto"
}
```

`tools` may be omitted or empty. `tool_choice` is required: `auto` lets the model think and write before, alongside, or instead of a tool call; `any` forces a tool call every turn, which leaves no room to think (use it for one-shot structured extraction, not agent loops).

Message `content` is a string or a list of `text` / `thinking` / `redacted_thinking` / `tool_use` / `tool_result` blocks. `thinking`, `redacted_thinking` and `tool_use` only on `assistant`; `tool_result` only on `user`. An assistant message may carry `provider` (`anthropic` | `gemini`) naming who produced it; send back exactly what a response returned, with its `provider`. At most one message may set `cache_breakpoint: true` to end a prefix the caller resends unchanged across loops (Hath marks the transcript a wake starts from); a merged turn carries the mark.

Each adapter replays its own provider's turns verbatim — thinking and signatures included, as Anthropic requires inside a tool loop — and translates the other provider's turns: Anthropic sees them as labelled text (`[other model thinking]`, `[other model called <tool>] {…}`, and the matching results), Gemini sees the other provider's thinking as labelled text and its tool calls as function calls carrying Google's `skip_thought_signature_validator` placeholder signature. Consecutive same-role turns are merged (tool results first); an assistant turn that opens with thinking stays whole behind a `[continue]` user turn.

Response:

```json
{
  "provider": "anthropic",
  "content": [
    { "type": "thinking", "thinking": "...", "signature": "..." },
    { "type": "text", "text": "..." },
    { "type": "tool_use", "id": "...", "name": "...", "input": {} }
  ],
  "stop_reason": "tool_use",
  "usage": {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_read_input_tokens": 0,
    "cache_creation_input_tokens": 0
  }
}
```

`stop_reason`: `end_turn` | `tool_use` | `max_tokens` | `error`. `content` keeps the provider's block order: thinking (Anthropic adaptive thinking requested with `display: "summarized"`, since current Claude models return empty thinking text by default; or Gemini thought summaries), text and tool calls, in one turn. On Anthropic, the last message block is a five-minute prompt-cache breakpoint, so an agent loop resending its history pays cache-read price for everything up to the previous step; the system prompt (its last block: the agent prompt on `/chat/agent`, `system` alone on `/chat/complete`) and the `cache_breakpoint` turn are one-hour breakpoints, so the next loop reads the tools, the system and its shared transcript instead of rewriting them; a request has at most three breakpoints, under Anthropic's limit of four. `input_tokens` counts only the uncached remainder.

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
