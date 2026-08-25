"""Gemini adapter. This is the only module that imports the Google GenAI SDK."""

from __future__ import annotations

from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

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


class GeminiAdapter:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_tokens: int,
        timeout_ms: int,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(
                timeout=timeout_ms,
                retry_options=genai_types.HttpRetryOptions(attempts=1),
            ),
        )

    def complete(self, request: ChatRequest, lane_block: str) -> ChatResponse:
        config_kwargs: dict[str, Any] = {
            "system_instruction": genai_types.Content(
                parts=[
                    genai_types.Part.from_text(text=request.system),
                    genai_types.Part.from_text(text=lane_block),
                ]
            ),
            "max_output_tokens": self._max_tokens,
            "thinking_config": genai_types.ThinkingConfig(
                thinking_level=genai_types.ThinkingLevel.MINIMAL,
                include_thoughts=False,
            ),
            "automatic_function_calling": genai_types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if request.tools:
            config_kwargs["tools"] = [
                genai_types.Tool(
                    function_declarations=[_to_declaration(tool) for tool in request.tools]
                )
            ]

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=_to_contents(request.messages),
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
        except genai_errors.ClientError as exc:
            code = exc.code
            if code in (408, 429):
                raise TransportError(
                    429 if code == 429 else 504,
                    "rate_limit" if code == 429 else "timeout",
                    exc.message,
                ) from exc
            raise DwarError(502, "provider", exc.message) from exc
        except genai_errors.ServerError as exc:
            if exc.code == 504:
                raise TransportError(504, "timeout", exc.message) from exc
            raise TransportError(502, "provider", exc.message) from exc
        except httpx.TimeoutException as exc:
            raise TransportError(504, "timeout", str(exc)) from exc
        except httpx.RequestError as exc:
            raise TransportError(502, "connection", str(exc)) from exc

        if not response.candidates:
            raise DwarError(502, "provider", "Gemini returned no candidates")
        candidate = response.candidates[0]
        parts = candidate.content.parts if candidate.content and candidate.content.parts else []

        content = []
        for part in parts:
            if getattr(part, "thought", False):
                continue
            if part.function_call:
                call = part.function_call
                if not call.id:
                    raise DwarError(502, "provider", "Gemini function call missing id")
                content.append(
                    ToolUseBlock(
                        type="tool_use",
                        id=call.id,
                        name=call.name,
                        input=dict(call.args) if call.args is not None else {},
                    )
                )
            elif part.text:
                content.append(TextBlock(type="text", text=part.text))

        if response.usage_metadata is None:
            raise DwarError(502, "provider", "Gemini response missing usage")
        usage_meta = response.usage_metadata
        if usage_meta.prompt_token_count is None or usage_meta.candidates_token_count is None:
            raise DwarError(502, "provider", "Gemini usage is missing token counts")
        thoughts = usage_meta.thoughts_token_count or 0
        return ChatResponse(
            content=content,
            stop_reason=_stop_reason(candidate.finish_reason, content),
            usage=Usage(
                input_tokens=usage_meta.prompt_token_count,
                output_tokens=usage_meta.candidates_token_count + thoughts,
            ),
        )


def _stop_reason(finish_reason: Any, content: list[TextBlock | ToolUseBlock]) -> StopReason:
    if any(isinstance(block, ToolUseBlock) for block in content):
        return "tool_use"
    raw = getattr(finish_reason, "name", None) or str(finish_reason or "")
    raw = raw.rsplit(".", 1)[-1].upper()
    if raw == "STOP":
        return "end_turn"
    if raw == "MAX_TOKENS":
        return "max_tokens"
    return "error"


def _to_declaration(tool: Tool) -> genai_types.FunctionDeclaration:
    return genai_types.FunctionDeclaration(
        name=tool.name,
        description=tool.description,
        parameters_json_schema=tool.input_schema,
    )


def _to_contents(messages: list[Message]) -> list[genai_types.Content]:
    names_by_id: dict[str, str] = {}
    for message in messages:
        if isinstance(message.content, str):
            continue
        for block in message.content:
            if isinstance(block, ToolUseBlock):
                names_by_id[block.id] = block.name

    contents: list[genai_types.Content] = []
    for message in messages:
        role = "model" if message.role == "assistant" else "user"
        contents.append(
            genai_types.Content(role=role, parts=_to_parts(message, names_by_id))
        )
    return contents


def _to_parts(
    message: Message, names_by_id: dict[str, str]
) -> list[genai_types.Part]:
    if isinstance(message.content, str):
        return [genai_types.Part.from_text(text=message.content)]

    parts: list[genai_types.Part] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append(genai_types.Part.from_text(text=block.text))
        elif isinstance(block, ToolUseBlock):
            parts.append(
                genai_types.Part(
                    function_call=genai_types.FunctionCall(
                        name=block.name,
                        args=block.input,
                        id=block.id,
                    )
                )
            )
        else:
            name = names_by_id.get(block.tool_use_id)
            if name is None:
                raise DwarError(
                    422,
                    "invalid_request",
                    f"tool_result {block.tool_use_id} has no matching tool_use",
                )
            response = (
                {"error": block.content} if block.is_error else {"result": block.content}
            )
            parts.append(
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        name=name,
                        response=response,
                        id=block.tool_use_id,
                    )
                )
            )
    return parts
