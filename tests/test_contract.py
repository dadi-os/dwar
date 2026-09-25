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

    return TestClient(create_app())


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


def test_complete_text_passes_none_lane_block(monkeypatch: pytest.MonkeyPatch) -> None:
    from inference import complete_text
    from inference.types import ChatRequest, ChatResponse, TextBlock, Usage

    seen: dict[str, object] = {}

    class _Adapter:
        def complete(self, request: ChatRequest, lane_block: str | None) -> ChatResponse:
            seen["lane_block"] = lane_block
            return ChatResponse(
                content=[TextBlock(type="text", text="ok")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=1, output_tokens=1),
            )

    monkeypatch.setattr("inference._chat_adapter", lambda endpoint: _Adapter())
    monkeypatch.setattr("inference._with_retry", lambda call: call())
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    get_config.cache_clear()

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
