"""Behavior tests for Dwar HTTP contract and config."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from config import get_config
from lanes import (
    conversation_lane_block,
    describe_instruction,
    reasoning_lane_block,
)


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "")
    get_config.cache_clear()
    from app import create_app

    return TestClient(create_app(), headers={"X-Dadi-Caller": "tests/contract"})


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_embed_empty_texts_invalid_request(client: TestClient) -> None:
    response = client.post("/embed", json={"texts": []})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["type"] == "invalid_request"


def test_reasoning_without_key_provider_unconfigured(client: TestClient) -> None:
    response = client.post(
        "/chat/reasoning",
        json={
            "system": "sys",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "provider_unconfigured"
    assert "ANTHROPIC_API_KEY" in body["error"]["message"]


def test_conversation_without_key_provider_unconfigured(client: TestClient) -> None:
    response = client.post(
        "/chat/conversation",
        json={
            "system": "sys",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "provider_unconfigured"
    assert "GEMINI_API_KEY" in body["error"]["message"]


def test_complete_without_key_provider_unconfigured(client: TestClient) -> None:
    response = client.post(
        "/chat/complete",
        json={
            "system": "sys",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "provider_unconfigured"
    assert "GEMINI_API_KEY" in body["error"]["message"]


def test_embed_without_key_provider_unconfigured(client: TestClient) -> None:
    response = client.post("/embed", json={"texts": ["hello"]})
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "provider_unconfigured"
    assert "OPENAI_API_KEY" in body["error"]["message"]


def test_lane_prompts_nonempty() -> None:
    assert reasoning_lane_block()
    assert conversation_lane_block()
    assert describe_instruction()


def test_conversation_prompt_asks_for_markdown() -> None:
    text = conversation_lane_block()
    assert "Markdown" in text
    assert "dispatch_message" in text
    assert "steer_reasoning" in text
    assert "terminate" in text
    assert "route_message" not in text
    assert "kebab-case" in text
    assert "parent" in text
    assert "point of contact" in text


def test_reasoning_prompt_has_no_router_branch() -> None:
    text = reasoning_lane_block()
    assert "route_message" not in text
    assert "router" not in text
    assert "send_message" in text
    assert "Do not invent tool names" in text or "do not invent tool names" in text.lower()


def _fake_chat_adapter(seen: dict[str, object]):
    """Chat adapter stub that records the lane block and returns fixed usage including cache tokens."""
    from inference.types import ChatRequest, ChatResponse, TextBlock, Usage

    class _Adapter:
        def complete(self, request: ChatRequest, lane_block: str | None) -> ChatResponse:
            seen["lane_block"] = lane_block
            return ChatResponse(
                content=[TextBlock(type="text", text="ok")],
                stop_reason="end_turn",
                usage=Usage(
                    input_tokens=11,
                    output_tokens=7,
                    cache_read_input_tokens=900,
                    cache_creation_input_tokens=40,
                ),
            )

    return _Adapter()


def test_complete_text_passes_none_lane_block(monkeypatch: pytest.MonkeyPatch) -> None:
    from inference import complete_text
    from inference.types import ChatRequest
    from logutil import caller_var, request_id_var

    seen: dict[str, object] = {}
    monkeypatch.setattr("inference._chat_adapter", lambda endpoint: _fake_chat_adapter(seen))
    monkeypatch.setattr("inference._with_retry", lambda call: call())
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_config.cache_clear()
    caller_var.set("tests/direct")
    request_id_var.set("req-direct")

    response = complete_text(
        ChatRequest(
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    assert response.stop_reason == "end_turn"
    assert "lane_block" in seen
    assert seen["lane_block"] is None


def test_config_loads() -> None:
    get_config.cache_clear()
    cfg = get_config()
    assert cfg.chat.reasoning.provider == "anthropic"
    assert cfg.chat.complete.provider == "gemini"
    assert cfg.embed.dimensions > 0
    assert cfg.retry.attempts >= 1


def test_request_id_header(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-Id": "abc123"})
    assert response.status_code == 200
    assert response.headers.get("X-Request-Id") == "abc123"


def test_image_describe_without_key_provider_unconfigured(client: TestClient) -> None:
    response = client.post(
        "/image/describe",
        json={
            "image": {
                "media_type": "image/jpeg",
                "data": "AAAA",
            }
        },
    )
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "provider_unconfigured"


def test_speech_without_key_provider_unconfigured(client: TestClient) -> None:
    response = client.post(
        "/speech/transcribe",
        json={
            "audio": {
                "media_type": "audio/wav",
                "data": "AAAA",
            }
        },
    )
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "provider_unconfigured"


def test_json_formatter_includes_code() -> None:
    from logutil import JsonFormatter
    import logging

    record = logging.LogRecord(
        name="dwar",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="boom",
        args=(),
        exc_info=None,
    )
    record.code = "provider_unconfigured"
    record.request_id = "req1"
    line = JsonFormatter().format(record)
    payload = json.loads(line)
    assert payload["service"] == "dwar"
    assert payload["code"] == "provider_unconfigured"
    assert payload["request_id"] == "req1"
    assert payload["level"] == "error"


def test_inference_requires_caller_header(client: TestClient) -> None:
    response = client.post(
        "/chat/complete",
        json={"system": "sys", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Dadi-Caller": ""},
    )
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request"

    bare = TestClient(client.app)
    response = bare.post(
        "/chat/complete",
        json={"system": "sys", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 422
    assert response.json()["error"]["type"] == "invalid_request"


def test_inference_log_line_attributes_cost(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr("inference._chat_adapter", lambda endpoint: _fake_chat_adapter(seen))
    caplog.set_level("INFO", logger="dwar")

    response = client.post(
        "/chat/complete",
        json={"system": "sys", "messages": [{"role": "user", "content": "hi"}]},
        headers={"X-Dadi-Caller": "dimaag/browser-manager", "X-Request-Id": "req-42"},
    )

    assert response.status_code == 200
    assert response.json()["usage"]["cache_read_input_tokens"] == 900
    lines = [r for r in caplog.records if r.getMessage() == "inference"]
    assert len(lines) == 1
    line = lines[0]
    assert line.caller == "dimaag/browser-manager"
    assert line.request_id == "req-42"
    assert line.route == "chat.complete"
    assert (line.input_tokens, line.output_tokens) == (11, 7)
    assert (line.cache_read_input_tokens, line.cache_creation_input_tokens) == (900, 40)


def test_healthy_health_checks_are_not_logged(
    client: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="dwar")
    client.get("/health")
    client.post("/embed", json={"texts": []})
    paths = [getattr(r, "path", None) for r in caplog.records if r.getMessage() == "request"]
    assert "/health" not in paths
    assert "/embed" in paths


def test_anthropic_caches_history_and_reports_cache_usage() -> None:
    from types import SimpleNamespace

    from inference.anthropic import AnthropicAdapter
    from inference.types import ChatRequest

    sent: dict[str, object] = {}

    def create(**kwargs: object) -> SimpleNamespace:
        sent.update(kwargs)
        return SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", id="t1", name="yield", input={})],
            stop_reason="tool_use",
            usage=SimpleNamespace(
                input_tokens=12,
                output_tokens=30,
                cache_read_input_tokens=5000,
                cache_creation_input_tokens=None,
            ),
        )

    adapter = AnthropicAdapter(api_key="k", model="m", max_tokens=10, timeout_seconds=1)
    adapter._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    request = ChatRequest.model_validate(
        {
            "system": "sys",
            "messages": [
                {"role": "user", "content": "[From: Ankur]\\nfind Oliver"},
                {
                    "role": "assistant",
                    "content": [{"type": "tool_use", "id": "t0", "name": "wait", "input": {}}],
                },
                {
                    "role": "user",
                    "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "done"}],
                },
            ],
        }
    )

    response = adapter.complete(request, "lane doctrine")

    messages = sent["messages"]
    assert messages[-1]["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert all("cache_control" not in block for m in messages[:-1] for block in (m["content"] if isinstance(m["content"], list) else []))
    assert messages[0]["content"] == "[From: Ankur]\\nfind Oliver"
    assert response.usage.cache_read_input_tokens == 5000
    assert response.usage.cache_creation_input_tokens == 0
