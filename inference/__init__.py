"""Provider-agnostic inference entry. Routers call these functions; they never import an SDK."""

from __future__ import annotations

import time
from dataclasses import dataclass
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
from lanes import CONVERSATION_LANE_BLOCK, DESCRIBE_INSTRUCTION, REASONING_LANE_BLOCK

Lane = Literal["reasoning", "conversation"]


@dataclass
class _Adapters:
    reasoning: AnthropicAdapter | GeminiAdapter
    conversation: AnthropicAdapter | GeminiAdapter
    embed: OpenAIAdapter
    describe: GeminiAdapter
    create: GeminiAdapter
    transcribe: DeepgramAdapter


_state: _Adapters | None = None


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


def _describe_adapter() -> GeminiAdapter:
    cfg = get_config()
    endpoint = cfg.image.describe
    if endpoint.provider != "gemini":
        raise RuntimeError(f"unsupported image describe provider: {endpoint.provider}")
    return GeminiAdapter(
        api_key=cfg.env.gemini_api_key,
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
        api_key=cfg.env.gemini_api_key,
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
        api_key=cfg.env.deepgram_api_key,
        model=endpoint.model,
        language=endpoint.language,
        timeout_seconds=cfg.retry.timeout_seconds,
    )


def init_adapters() -> None:
    """Construct adapters at startup so a bad provider config fails before serving."""
    global _state
    cfg = get_config()
    _state = _Adapters(
        reasoning=_chat_adapter(cfg.chat.reasoning),
        conversation=_chat_adapter(cfg.chat.conversation),
        embed=_embed_adapter(),
        describe=_describe_adapter(),
        create=_create_adapter(),
        transcribe=_transcribe_adapter(),
    )


def _adapters() -> _Adapters:
    if _state is None:
        init_adapters()
    if _state is None:
        raise RuntimeError("inference adapters failed to initialize")
    return _state


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
    adapters = _adapters()
    if lane == "reasoning":
        return _with_retry(lambda: adapters.reasoning.complete(request, REASONING_LANE_BLOCK))
    return _with_retry(lambda: adapters.conversation.complete(request, CONVERSATION_LANE_BLOCK))


def embed(texts: list[str]) -> EmbedResult:
    return _with_retry(lambda: _adapters().embed.embed(texts))


def describe_image(
    image: bytes, media_type: str, prompt: str | None
) -> DescribeResult:
    instruction = prompt if prompt is not None else DESCRIBE_INSTRUCTION
    return _with_retry(
        lambda: _adapters().describe.describe_image(image, media_type, instruction)
    )


def create_image(prompt: str) -> CreateResult:
    return _with_retry(lambda: _adapters().create.create_image(prompt))


def transcribe(audio: bytes, media_type: str) -> TranscribeResult:
    return _with_retry(lambda: _adapters().transcribe.transcribe(audio, media_type))
