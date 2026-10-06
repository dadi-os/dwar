"""v1 HTTP contract. Independent of provider adapters."""

import base64
import binascii
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import get_config

StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]
Role = Literal["user", "assistant"]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


Provider = Literal["anthropic", "gemini"]
Lane = Literal["reasoning", "conversation"]
ToolChoice = Literal["auto", "any"]


class TextBlock(_Contract):
    type: Literal["text"]
    text: str
    thought_signature: str | None = None
    """Opaque provider state (Gemini signs a trailing text part). Round-trip unchanged."""


class ThinkingBlock(_Contract):
    type: Literal["thinking"]
    thinking: str
    signature: str | None = None
    """Anthropic's thinking signature, or Gemini's signature on a thought part."""


class RedactedThinkingBlock(_Contract):
    type: Literal["redacted_thinking"]
    data: str


class ToolUseBlock(_Contract):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]
    thought_signature: str | None = None
    """Opaque provider state (Gemini thought signatures). Round-trip unchanged."""


class ToolResultBlock(_Contract):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    Union[TextBlock, ThinkingBlock, RedactedThinkingBlock, ToolUseBlock, ToolResultBlock],
    Field(discriminator="type"),
]

ResponseBlock = Annotated[
    Union[TextBlock, ThinkingBlock, RedactedThinkingBlock, ToolUseBlock],
    Field(discriminator="type"),
]

_ASSISTANT_ONLY = {"thinking", "redacted_thinking", "tool_use"}


class Message(_Contract):
    role: Role
    content: str | list[ContentBlock]
    provider: Provider | None = None
    """Who produced an assistant turn. Adapters replay their own provider's turns
    verbatim (thinking, signatures) and translate the other provider's turns. None is
    plain text with no provider state (a transcript line).
    """
    lane: Lane | None = None
    cache_breakpoint: bool = False
    """Ends a prefix the caller resends unchanged across requests beyond the one in
    flight (Hath: the transcript a wake starts from). Providers that cache by prefix
    keep a long-lived entry here. At most one message per request.
    """


class Tool(_Contract):
    name: str
    description: str
    input_schema: dict[str, Any]


class Usage(_Contract):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


class ChatRequest(_Contract):
    system: str
    messages: list[Message]
    tools: list[Tool] = Field(default_factory=list)
    tool_choice: ToolChoice
    """auto lets the model think and write before (or instead of) a tool call; any
    forces a tool call on every turn and leaves no room to think.
    """

    @model_validator(mode="after")
    def check_block_roles(self) -> "ChatRequest":
        if sum(message.cache_breakpoint for message in self.messages) > 1:
            raise ValueError("at most one message sets cache_breakpoint")
        for message in self.messages:
            if message.role != "assistant" and (
                message.provider is not None or message.lane is not None
            ):
                raise ValueError("provider and lane appear only on assistant messages")
            if (message.provider is None) != (message.lane is None):
                raise ValueError("provider and lane are set together")
            if isinstance(message.content, str):
                continue
            for block in message.content:
                if block.type in _ASSISTANT_ONLY and message.role != "assistant":
                    raise ValueError(f"{block.type} blocks appear only in assistant messages")
                if block.type == "tool_result" and message.role != "user":
                    raise ValueError("tool_result blocks appear only in user messages")
        return self


class ChatResponse(_Contract):
    content: list[ResponseBlock]
    stop_reason: StopReason
    usage: Usage
    provider: Provider


class EmbedRequest(_Contract):
    texts: list[str]

    @model_validator(mode="after")
    def check_batch(self) -> "EmbedRequest":
        if not self.texts:
            raise ValueError("texts must not be empty")
        embed = get_config().embed
        if len(self.texts) > embed.max_batch_size:
            raise ValueError(
                f"texts exceeds max batch size of {embed.max_batch_size}"
            )
        for index, text in enumerate(self.texts):
            if len(text) > embed.max_text_length:
                raise ValueError(
                    f"texts[{index}] exceeds max length of {embed.max_text_length}"
                )
        return self


class EmbedUsage(_Contract):
    input_tokens: int


class EmbedResponse(_Contract):
    embeddings: list[list[float]]
    dimensions: int
    usage: EmbedUsage


class MediaBlob(_Contract):
    media_type: str
    data: str

    def decoded(self) -> bytes:
        return base64.b64decode(self.data, validate=True)


def _check_media(
    blob: MediaBlob, *, allowed: tuple[str, ...], max_bytes: int, field: str
) -> None:
    if blob.media_type not in allowed:
        raise ValueError(f"{field}.media_type is not an allowed type")
    try:
        raw = base64.b64decode(blob.data, validate=True)
    except binascii.Error as exc:
        raise ValueError(f"{field}.data is not valid base64") from exc
    if not raw:
        raise ValueError(f"{field}.data must not be empty")
    if len(raw) > max_bytes:
        raise ValueError(f"{field} exceeds max size of {max_bytes} bytes")


class DescribeRequest(_Contract):
    image: MediaBlob
    prompt: str | None = None

    @model_validator(mode="after")
    def check_image(self) -> "DescribeRequest":
        describe = get_config().image.describe
        _check_media(
            self.image,
            allowed=describe.allowed_media_types,
            max_bytes=describe.max_bytes,
            field="image",
        )
        if self.prompt is not None and not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.prompt is not None and len(self.prompt) > describe.max_prompt_length:
            raise ValueError(
                f"prompt exceeds max length of {describe.max_prompt_length}"
            )
        return self


class DescribeResponse(_Contract):
    description: str
    usage: Usage


class CreateRequest(_Contract):
    prompt: str

    @model_validator(mode="after")
    def check_prompt(self) -> "CreateRequest":
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        limit = get_config().image.create.max_prompt_length
        if len(self.prompt) > limit:
            raise ValueError(f"prompt exceeds max length of {limit}")
        return self


class CreateResponse(_Contract):
    image: MediaBlob
    usage: Usage


class TranscribeRequest(_Contract):
    audio: MediaBlob

    @model_validator(mode="after")
    def check_audio(self) -> "TranscribeRequest":
        transcribe = get_config().speech.transcribe
        _check_media(
            self.audio,
            allowed=transcribe.allowed_media_types,
            max_bytes=transcribe.max_bytes,
            field="audio",
        )
        return self


class TranscribeResponse(_Contract):
    text: str
    duration_seconds: float
