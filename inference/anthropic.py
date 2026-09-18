"""Anthropic adapter. This is the only module that imports the Anthropic SDK."""

from __future__ import annotations

from typing import Any

import anthropic
import httpx2

from errors import DwarError, TransportError
from inference.types import (
    ChatRequest,
    ChatResponse,
    Message,
    StopReason,
    TextBlock,
    Tool,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "end_turn",
    "pause_turn": "end_turn",
    "refusal": "error",
}

_THINKING_TYPES = {"thinking", "redacted_thinking"}


class AnthropicAdapter:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int,
        timeout_seconds: float,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(
            api_key=api_key,
            max_retries=0,
            timeout=timeout_seconds,
        )

    def complete(self, request: ChatRequest, lane_block: str | None) -> ChatResponse:
        system: list[dict[str, Any]] = [{"type": "text", "text": request.system}]
        if lane_block is not None:
            system.append(
                {
                    "type": "text",
                    "text": lane_block,
                    "cache_control": {"type": "ephemeral"},
                }
            )
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "thinking": {"type": "adaptive"},
            "system": system,
            "messages": [_to_message(message) for message in request.messages],
        }
        if request.tools:
            kwargs["tools"] = [_to_tool(tool) for tool in request.tools]
            # Force at least one tool call; text may still accompany it.
            kwargs["tool_choice"] = {"type": "any"}

        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.RateLimitError as exc:
            raise TransportError(429, "rate_limit", str(exc)) from exc
        except anthropic.APITimeoutError as exc:
            raise TransportError(504, "upstream_timeout", str(exc)) from exc
        except anthropic.APIConnectionError as exc:
            raise TransportError(502, "upstream_unreachable", str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code == 429:
                raise TransportError(429, "rate_limit", str(exc)) from exc
            if exc.status_code in (408, 504):
                raise TransportError(504, "upstream_timeout", str(exc)) from exc
            if exc.status_code >= 500:
                raise TransportError(502, "provider", str(exc)) from exc
            raise DwarError(502, "provider", str(exc)) from exc
        except httpx2.TimeoutException as exc:
            raise TransportError(504, "upstream_timeout", str(exc)) from exc
        except httpx2.RequestError as exc:
            raise TransportError(502, "upstream_unreachable", str(exc)) from exc

        content = []
        for block in response.content:
            if block.type in _THINKING_TYPES:
                continue
            if block.type == "text":
                content.append(TextBlock(type="text", text=block.text))
            elif block.type == "tool_use":
                content.append(
                    ToolUseBlock(
                        type="tool_use",
                        id=block.id,
                        name=block.name,
                        input=dict(block.input) if block.input is not None else {},
                    )
                )

        stop = _STOP_REASONS.get(response.stop_reason or "", "error")
        usage = response.usage
        return ChatResponse(
            content=content,
            stop_reason=stop,
            usage=Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            ),
        )


def _to_tool(tool: Tool) -> dict[str, Any]:
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    }


def _to_message(message: Message) -> dict[str, Any]:
    if isinstance(message.content, str):
        return {"role": message.role, "content": message.content}
    return {
        "role": message.role,
        "content": [_to_block(block) for block in message.content],
    }


def _to_block(block: TextBlock | ToolUseBlock | ToolResultBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": block.input,
        }
    payload: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": block.tool_use_id,
        "content": block.content,
    }
    if block.is_error:
        payload["is_error"] = True
    return payload
