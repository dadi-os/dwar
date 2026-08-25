"""Provider-agnostic inference entry. Routers call these functions; they never import an SDK."""

from __future__ import annotations

import time
from typing import Literal

from config import ChatEndpoint, get_config
from errors import DwarError, TransportError
from inference.anthropic import AnthropicAdapter
from inference.gemini import GeminiAdapter
from inference.openai import OpenAIAdapter
from inference.types import ChatRequest, ChatResponse, EmbedResult
from lanes import CONVERSATION_LANE_BLOCK, REASONING_LANE_BLOCK

Lane = Literal["reasoning", "conversation"]

_reasoning: AnthropicAdapter | GeminiAdapter | None = None
_conversation: AnthropicAdapter | GeminiAdapter | None = None
_embed: OpenAIAdapter | None = None


def _chat_adapter(endpoint: ChatEndpoint) -> AnthropicAdapter | GeminiAdapter:
    cfg = get_config()
    timeout = cfg.retry.timeout_seconds
    if endpoint.provider == "anthropic":
        return AnthropicAdapter(
            api_key=cfg.env.anthropic_api_key,
            model=endpoint.model,
            max_tokens=endpoint.max_tokens,
            timeout_seconds=timeout,
        )
    if endpoint.provider == "gemini":
        return GeminiAdapter(
            api_key=cfg.env.gemini_api_key,
            model=endpoint.model,
            max_tokens=endpoint.max_tokens,
            timeout_ms=int(timeout * 1000),
        )
    raise RuntimeError(f"unsupported chat provider: {endpoint.provider}")


def _embed_adapter() -> OpenAIAdapter:
    cfg = get_config()
    embed = cfg.embed
    if embed.provider != "openai":
        raise RuntimeError(f"unsupported embed provider: {embed.provider}")
    return OpenAIAdapter(
        api_key=cfg.env.openai_api_key,
        model=embed.model,
        dimensions=embed.dimensions,
        timeout_seconds=cfg.retry.timeout_seconds,
    )


def init_adapters() -> None:
    """Construct adapters at startup so a bad provider config fails before serving."""
    global _reasoning, _conversation, _embed
    cfg = get_config()
    _reasoning = _chat_adapter(cfg.chat.reasoning)
    _conversation = _chat_adapter(cfg.chat.conversation)
    _embed = _embed_adapter()


def _adapters() -> tuple[AnthropicAdapter | GeminiAdapter, AnthropicAdapter | GeminiAdapter, OpenAIAdapter]:
    if _reasoning is None or _conversation is None or _embed is None:
        init_adapters()
    if _reasoning is None or _conversation is None or _embed is None:
        raise RuntimeError("inference adapters failed to initialize")
    return _reasoning, _conversation, _embed


def _with_retry(call):
    retry = get_config().retry
    for attempt in range(retry.attempts):
        try:
            return call()
        except TransportError as exc:
            if attempt + 1 == retry.attempts:
                raise DwarError(exc.status_code, exc.type, exc.message) from exc
            delay = retry.backoff_seconds[min(attempt, len(retry.backoff_seconds) - 1)]
            time.sleep(delay)


def complete_chat(lane: Lane, request: ChatRequest) -> ChatResponse:
    reasoning, conversation, _ = _adapters()
    if lane == "reasoning":
        return _with_retry(lambda: reasoning.complete(request, REASONING_LANE_BLOCK))
    return _with_retry(lambda: conversation.complete(request, CONVERSATION_LANE_BLOCK))


def embed(texts: list[str]) -> EmbedResult:
    _, _, adapter = _adapters()
    return _with_retry(lambda: adapter.embed(texts))
