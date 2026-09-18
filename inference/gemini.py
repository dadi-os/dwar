"""Gemini adapter. This is the only module that imports the Google GenAI SDK."""

from __future__ import annotations

import base64
import binascii
from typing import Any

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from errors import DwarError, TransportError
from inference.types import (
    ChatRequest,
    ChatResponse,
    CreateResult,
    DescribeResult,
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
        max_tokens: int | None,
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

    def _generate(
        self, contents: Any, config: genai_types.GenerateContentConfig
    ) -> Any:
        try:
            return self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=config,
            )
        except genai_errors.ClientError as exc:
            code = exc.code
            if code in (408, 429):
                raise TransportError(
                    429 if code == 429 else 504,
                    "rate_limit" if code == 429 else "upstream_timeout",
                    exc.message,
                ) from exc
            raise DwarError(502, "provider", exc.message) from exc
        except genai_errors.ServerError as exc:
            if exc.code == 504:
                raise TransportError(504, "upstream_timeout", exc.message) from exc
            raise TransportError(502, "provider", exc.message) from exc
        except httpx.TimeoutException as exc:
            raise TransportError(504, "upstream_timeout", str(exc)) from exc
        except httpx.RequestError as exc:
            raise TransportError(502, "upstream_unreachable", str(exc)) from exc

    def complete(self, request: ChatRequest, lane_block: str | None) -> ChatResponse:
        parts = [genai_types.Part.from_text(text=request.system)]
        if lane_block is not None:
            parts.append(genai_types.Part.from_text(text=lane_block))
        config_kwargs: dict[str, Any] = {
            "system_instruction": genai_types.Content(parts=parts),
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
            # Force at least one tool call; text may still accompany it.
            config_kwargs["tool_config"] = genai_types.ToolConfig(
                function_calling_config=genai_types.FunctionCallingConfig(
                    mode=genai_types.FunctionCallingConfigMode.ANY
                )
            )

        response = self._generate(
            _to_contents(request.messages),
            genai_types.GenerateContentConfig(**config_kwargs),
        )
        parts = _candidate_parts(response)

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
                        thought_signature=_encode_thought_signature(part),
                    )
                )
            elif part.text:
                content.append(TextBlock(type="text", text=part.text))

        return ChatResponse(
            content=content,
            stop_reason=_stop_reason(_finish_reason(response), content),
            usage=_usage(response),
        )

    def describe_image(
        self, image: bytes, media_type: str, instruction: str
    ) -> DescribeResult:
        response = self._generate(
            [
                genai_types.Part.from_bytes(data=image, mime_type=media_type),
                genai_types.Part.from_text(text=instruction),
            ],
            genai_types.GenerateContentConfig(
                max_output_tokens=self._max_tokens,
                thinking_config=genai_types.ThinkingConfig(
                    thinking_level=genai_types.ThinkingLevel.MINIMAL,
                    include_thoughts=False,
                ),
            ),
        )
        texts: list[str] = []
        for part in _candidate_parts(response):
            if getattr(part, "thought", False):
                continue
            if part.text:
                texts.append(part.text)
        description = "".join(texts).strip()
        if not description:
            raise DwarError(502, "provider", "Gemini returned no description")
        return DescribeResult(description=description, usage=_usage(response))

    def create_image(self, prompt: str) -> CreateResult:
        response = self._generate(
            prompt,
            genai_types.GenerateContentConfig(response_modalities=["IMAGE"]),
        )
        image: bytes | None = None
        media_type: str | None = None
        for part in _candidate_parts(response):
            if getattr(part, "thought", False):
                continue
            inline = part.inline_data
            if inline is None or inline.data is None:
                continue
            raw = inline.data
            if isinstance(raw, str):
                try:
                    image = base64.b64decode(raw)
                except binascii.Error as exc:
                    raise DwarError(
                        502, "provider", "Gemini image data is not valid base64"
                    ) from exc
            else:
                image = bytes(raw)
            media_type = inline.mime_type
            break
        if image is None:
            raise DwarError(502, "provider", "Gemini returned no image")
        if not media_type:
            raise DwarError(502, "provider", "Gemini image missing media type")
        return CreateResult(media_type=media_type, image=image, usage=_usage(response))


def _candidate_parts(response: Any) -> list[Any]:
    if not response.candidates:
        raise DwarError(502, "provider", "Gemini returned no candidates")
    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return []
    return list(candidate.content.parts)


def _finish_reason(response: Any) -> Any:
    if not response.candidates:
        raise DwarError(502, "provider", "Gemini returned no candidates")
    return response.candidates[0].finish_reason


def _usage(response: Any) -> Usage:
    if response.usage_metadata is None:
        raise DwarError(502, "provider", "Gemini response missing usage")
    usage_meta = response.usage_metadata
    if usage_meta.prompt_token_count is None or usage_meta.candidates_token_count is None:
        raise DwarError(502, "provider", "Gemini usage is missing token counts")
    thoughts = usage_meta.thoughts_token_count or 0
    return Usage(
        input_tokens=usage_meta.prompt_token_count,
        output_tokens=usage_meta.candidates_token_count + thoughts,
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


def _encode_thought_signature(part: Any) -> str | None:
    raw = getattr(part, "thought_signature", None)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        return base64.b64encode(raw).decode("ascii")
    if isinstance(raw, str) and raw:
        return raw
    return None


def _decode_thought_signature(value: str | None) -> bytes | None:
    if value is None or value == "":
        return None
    try:
        return base64.b64decode(value, validate=True)
    except binascii.Error as exc:
        raise DwarError(
            422,
            "invalid_request",
            "tool_use thought_signature is not valid base64",
        ) from exc


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
            part_kwargs: dict[str, Any] = {
                "function_call": genai_types.FunctionCall(
                    name=block.name,
                    args=block.input,
                    id=block.id,
                )
            }
            signature = _decode_thought_signature(block.thought_signature)
            if signature is not None:
                part_kwargs["thought_signature"] = signature
            parts.append(genai_types.Part(**part_kwargs))
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
