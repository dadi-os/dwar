"""v1 HTTP contract. Independent of provider adapters."""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from config import get_config

StopReason = Literal["end_turn", "tool_use", "max_tokens", "error"]
Role = Literal["user", "assistant"]


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextBlock(_Contract):
    type: Literal["text"]
    text: str


class ToolUseBlock(_Contract):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ToolResultBlock(_Contract):
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


class Message(_Contract):
    role: Role
    content: str | list[ContentBlock]


class Tool(_Contract):
    name: str
    description: str
    input_schema: dict[str, Any]


class Usage(_Contract):
    input_tokens: int
    output_tokens: int


class ChatRequest(_Contract):
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


class ChatResponse(_Contract):
    content: list[ResponseBlock]
    stop_reason: StopReason
    usage: Usage


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
