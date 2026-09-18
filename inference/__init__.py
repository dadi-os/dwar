"""Provider-agnostic inference entry. Routers call these functions; they never import an SDK."""

from __future__ import annotations

import time
from typing import Literal

from config import ChatEndpoint, get_config
from errors import DwarError, TransportError
from inference.anthropic import AnthropicAdapter
from inference.deepgram import DeepgramAdapter
from inference.gemini import GeminiAdapter
from inference.openai import OpenAIAdapter
from inference.types import (
    ChatRequest,
    ChatResponse,
    CreateResult,
    DescribeResult,
    EmbedResult,
    TranscribeResult,
)
from lanes import (
    conversation_lane_block,
    describe_instruction,
    reasoning_lane_block,
)

Lane = Literal["reasoning", "conversation"]


def _require_key(value: str, name: str) -> str:
    if not value:
        raise DwarError(503, "provider_unconfigured", f"{name} is not set")
    return value


def _chat_adapter(endpoint: ChatEndpoint) -> AnthropicAdapter | GeminiAdapter:
    cfg = get_config()
    timeout = cfg.retry.timeout_seconds
    if endpoint.provider == "anthropic":
        return AnthropicAdapter(
            api_key=_require_key(cfg.env.anthropic_api_key, "ANTHROPIC_API_KEY"),
            model=endpoint.model,
            max_tokens=endpoint.max_tokens,
            timeout_seconds=timeout,
        )
    if endpoint.provider == "gemini":
        return GeminiAdapter(
            api_key=_require_key(cfg.env.gemini_api_key, "GEMINI_API_KEY"),
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
        api_key=_require_key(cfg.env.openai_api_key, "OPENAI_API_KEY"),
        model=embed.model,
        dimensions=embed.dimensions,
        timeout_seconds=cfg.retry.timeout_seconds,
    )


def _describe_adapter() -> GeminiAdapter:
    cfg = get_config()
    endpoint = cfg.image.describe
    if endpoint.provider != "gemini":
        raise RuntimeError(f"unsupported image describe provider: {endpoint.provider}")
    return GeminiAdapter(
        api_key=_require_key(cfg.env.gemini_api_key, "GEMINI_API_KEY"),
        model=endpoint.model,
        max_tokens=endpoint.max_tokens,
        timeout_ms=int(cfg.retry.timeout_seconds * 1000),
    )


def _create_adapter() -> GeminiAdapter:
    cfg = get_config()
    endpoint = cfg.image.create
    if endpoint.provider != "gemini":
        raise RuntimeError(f"unsupported image create provider: {endpoint.provider}")
    return GeminiAdapter(
        api_key=_require_key(cfg.env.gemini_api_key, "GEMINI_API_KEY"),
        model=endpoint.model,
        max_tokens=None,
        timeout_ms=int(cfg.retry.timeout_seconds * 1000),
    )


def _transcribe_adapter() -> DeepgramAdapter:
    cfg = get_config()
    endpoint = cfg.speech.transcribe
    if endpoint.provider != "deepgram":
        raise RuntimeError(f"unsupported speech transcribe provider: {endpoint.provider}")
    return DeepgramAdapter(
        api_key=_require_key(cfg.env.deepgram_api_key, "DEEPGRAM_API_KEY"),
        model=endpoint.model,
        language=endpoint.language,
        timeout_seconds=cfg.retry.timeout_seconds,
    )


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
    cfg = get_config()
    if lane == "reasoning":
        endpoint = cfg.chat.reasoning
        lane_block = reasoning_lane_block()
    else:
        endpoint = cfg.chat.conversation
        lane_block = conversation_lane_block()
    adapter = _chat_adapter(endpoint)
    return _with_retry(lambda: adapter.complete(request, lane_block))


def complete_text(request: ChatRequest) -> ChatResponse:
    """Run chat with the caller's system prompt only — no Dwar lane block."""
    adapter = _chat_adapter(get_config().chat.complete)
    return _with_retry(lambda: adapter.complete(request, None))


def embed(texts: list[str]) -> EmbedResult:
    return _with_retry(lambda: _embed_adapter().embed(texts))


def describe_image(
    image: bytes, media_type: str, prompt: str | None
) -> DescribeResult:
    instruction = prompt if prompt is not None else describe_instruction()
    return _with_retry(
        lambda: _describe_adapter().describe_image(image, media_type, instruction)
    )


def create_image(prompt: str) -> CreateResult:
    return _with_retry(lambda: _create_adapter().create_image(prompt))


def transcribe(audio: bytes, media_type: str) -> TranscribeResult:
    return _with_retry(lambda: _transcribe_adapter().transcribe(audio, media_type))
