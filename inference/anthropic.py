"""Anthropic adapter. This is the only module that imports the Anthropic SDK."""

from __future__ import annotations

from typing import Any

import anthropic
import httpx2

from errors import DwarError, TransportError
from inference.history import foreign_turns_as_text, merge_turns
from inference.types import (
    ChatRequest,
    ChatResponse,
    ContentBlock,
    Message,
    RedactedThinkingBlock,
    ResponseBlock,
    StopReason,
    TextBlock,
    ThinkingBlock,
    Tool,
    ToolUseBlock,
    Usage,
)

_LONG_LIVED = {"type": "ephemeral", "ttl": "1h"}
"""Cache entry for prefixes reused across loops: the system lane block and a cache_breakpoint turn."""

_SHORT_LIVED = {"type": "ephemeral"}
"""Five-minute cache entry on the newest message, read by the loop's next step."""

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "stop_sequence": "end_turn",
    "pause_turn": "end_turn",
    "refusal": "error",
}


class AnthropicAdapter:
    """Chat completions on the Anthropic Messages API, mapping provider failures to Dwar errors."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int,
        effort: str,
        timeout_seconds: float,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._effort = effort
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
                    "cache_control": _LONG_LIVED,
                }
            )
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "thinking": {"type": "adaptive", "display": "summarized"},
            "output_config": {"effort": self._effort},
            "system": system,
            "messages": _with_breakpoints(_shape_history(request.messages)),
        }
        if request.tools:
            kwargs["tools"] = [_to_tool(tool) for tool in request.tools]
            kwargs["tool_choice"] = {"type": request.tool_choice}

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

        content: list[ResponseBlock] = []
        for block in response.content:
            if block.type == "thinking":
                content.append(
                    ThinkingBlock(type="thinking", thinking=block.thinking, signature=block.signature)
                )
            elif block.type == "redacted_thinking":
                content.append(RedactedThinkingBlock(type="redacted_thinking", data=block.data))
            elif block.type == "text":
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
            provider="anthropic",
            content=content,
            stop_reason=stop,
            provider_stop_reason=response.stop_reason,
            usage=Usage(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_input_tokens=usage.cache_read_input_tokens or 0,
                cache_creation_input_tokens=usage.cache_creation_input_tokens or 0,
            ),
        )


def _with_breakpoints(messages: list[Message]) -> list[dict[str, Any]]:
    """Render the history with its cache breakpoints.

    Agent loops resend the whole history every step. The last message keeps a
    five-minute entry the loop's next step reads. A cache_breakpoint turn keeps a
    one-hour entry: it ends the prefix the caller's next loop starts from too (an
    agent's next wake opens on the same transcript), so the steps in between keep
    it warm and the next loop reads it instead of rewriting it. Anthropic wants
    longer-lived entries ahead of shorter ones, which is why the system lane block
    is one-hour as well.
    """
    rendered = [_to_message(message) for message in messages]
    for index, message in enumerate(messages):
        if message.cache_breakpoint:
            rendered[index] = _cached(rendered[index], _LONG_LIVED)
    if messages and not messages[-1].cache_breakpoint:
        rendered[-1] = _cached(rendered[-1], _SHORT_LIVED)
    return rendered


def _cached(message: dict[str, Any], cache_control: dict[str, str]) -> dict[str, Any]:
    """_cached returns `message` with `cache_control` on its last block."""
    content = message["content"]
    if isinstance(content, str):
        blocks: list[dict[str, Any]] = [{"type": "text", "text": content}]
    else:
        blocks = [dict(block) for block in content]
    blocks[-1]["cache_control"] = cache_control
    return {"role": message["role"], "content": blocks}


def _shape_history(messages: list[Message]) -> list[Message]:
    """Keep Anthropic's own turns verbatim (thinking must be replayed unchanged
    inside a tool loop); Anthropic cannot validate another provider's tool calls
    or thinking, so those turns become labelled text."""
    return merge_turns(
        foreign_turns_as_text(messages, "anthropic"), split_before_thinking=True
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


def _to_block(block: ContentBlock) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"type": "text", "text": block.text}
    if isinstance(block, ThinkingBlock):
        if block.signature is None:
            raise DwarError(422, "invalid_request", "anthropic thinking block is missing its signature")
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    if isinstance(block, RedactedThinkingBlock):
        return {"type": "redacted_thinking", "data": block.data}
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
