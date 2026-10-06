"""Version-agnostic inference types. HTTP contracts live with their version routers."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]
Role = Literal["user", "assistant"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


Provider = Literal["anthropic", "gemini"]
Lane = Literal["reasoning", "conversation"]
ToolChoice = Literal["auto", "any"]


class TextBlock(_Model):
    type: Literal["text"]
    text: str
    thought_signature: str | None = None
    """Opaque provider state (Gemini signs a trailing text part). Round-trip unchanged."""


class ThinkingBlock(_Model):
    type: Literal["thinking"]
    thinking: str
    signature: str | None = None
    """Anthropic's thinking signature, or Gemini's signature on a thought part."""


class RedactedThinkingBlock(_Model):
    type: Literal["redacted_thinking"]
    data: str


class ToolUseBlock(_Model):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]
    thought_signature: str | None = None
    """Opaque provider state (Gemini thought signatures). Round-trip unchanged."""


class ToolResultBlock(_Model):
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


class Message(_Model):
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
    flight. Adapters that cache by prefix keep a long-lived entry here; merging or
    translating the turn carries the mark onto the turn it lands in.
    """


class Tool(_Model):
    name: str
    description: str
    input_schema: dict[str, Any]


class Usage(_Model):
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int


class ChatRequest(_Model):
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


class ChatResponse(_Model):
    content: list[ResponseBlock]
    stop_reason: StopReason
    usage: Usage
    provider: Provider


class EmbedResult(_Model):
    embeddings: list[list[float]]
    dimensions: int
    input_tokens: int


class DescribeResult(_Model):
    description: str
    usage: Usage


class CreateResult(_Model):
    media_type: str
    image: bytes
    usage: Usage


class TranscribeResult(_Model):
    text: str
    duration_seconds: float
