"""Version-agnostic inference types. HTTP contracts live with their version routers."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]
Role = Literal["user", "assistant"]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextBlock(_Model):
    type: Literal["text"]
    text: str


class ToolUseBlock(_Model):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]
    # Opaque provider state (Gemini thought signatures). Round-trip unchanged.
    thought_signature: str | None = None


class ToolResultBlock(_Model):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str
    is_error: bool = False


ContentBlock = Annotated[
    Union[TextBlock, ToolUseBlock, ToolResultBlock],
    Field(discriminator="type"),
]

ResponseBlock = Annotated[
    Union[TextBlock, ToolUseBlock],
    Field(discriminator="type"),
]


class Message(_Model):
    role: Role
    content: str | list[ContentBlock]


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

    @model_validator(mode="after")
    def check_block_roles(self) -> "ChatRequest":
        for message in self.messages:
            if isinstance(message.content, str):
                continue
            for block in message.content:
                if block.type == "tool_use" and message.role != "assistant":
                    raise ValueError("tool_use blocks appear only in assistant messages")
                if block.type == "tool_result" and message.role != "user":
                    raise ValueError("tool_result blocks appear only in user messages")
        return self


class ChatResponse(_Model):
    content: list[ResponseBlock]
    stop_reason: StopReason
    usage: Usage


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
