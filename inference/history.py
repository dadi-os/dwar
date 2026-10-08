"""Provider-neutral history shaping shared by the chat adapters.

A request's history holds assistant turns from both of an agent's lanes, which
share one wake and may run on different providers. A lane's own turns (same
provider, same lane) replay verbatim with their opaque state (thinking
signatures, Gemini thought signatures). Every other turn is foreign: the other
provider cannot replay its state, and the other lane's tool calls must not read
as calls this lane made, so each adapter uses this module to reshape them.
"""

from __future__ import annotations

import json

from inference.types import (
    ContentBlock,
    Lane,
    Message,
    Provider,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)

CONTINUE = "[continue]"
"""Neutral user turn that separates two assistant turns without adding meaning."""


def is_foreign(message: Message, provider: Provider, lane: Lane | None) -> bool:
    """is_foreign reports whether an assistant turn came from another provider or
    another lane. lane is None outside the lanes (chat.complete), where every
    lane's turn is foreign."""
    return (
        message.role == "assistant"
        and message.provider is not None
        and (message.provider != provider or message.lane != lane)
    )


def thinking_as_text(message: Message, block: ThinkingBlock) -> TextBlock:
    """thinking_as_text renders another lane's thinking as a labelled text block."""
    return TextBlock(type="text", text=f"[{message.lane} lane thinking]\n{block.thinking}")


def foreign_turns_as_text(
    messages: list[Message], provider: Provider, lane: Lane | None
) -> list[Message]:
    """foreign_turns_as_text rewrites foreign assistant turns, and the results of
    their tool calls, as labelled text, so the model neither replays state it did
    not sign nor reads the other lane's tool calls as its own. A foreign turn with
    nothing readable (only redacted thinking) is dropped."""
    foreign_calls: dict[str, tuple[str, str]] = {}
    out: list[Message] = []
    for message in messages:
        if is_foreign(message, provider, lane):
            text = _assistant_as_text(message, foreign_calls)
            if text:
                out.append(
                    Message(
                        role="assistant",
                        content=text,
                        cache_breakpoint=message.cache_breakpoint,
                    )
                )
            continue
        if message.role == "user" and not isinstance(message.content, str):
            out.append(
                Message(
                    role="user",
                    content=_results_as_text(message.content, foreign_calls),
                    cache_breakpoint=message.cache_breakpoint,
                )
            )
            continue
        out.append(message)
    return out


def _assistant_as_text(message: Message, foreign_calls: dict[str, tuple[str, str]]) -> str:
    """_assistant_as_text flattens one foreign assistant turn to labelled text and
    records its tool call ids in foreign_calls so their results follow suit."""
    if isinstance(message.content, str):
        return message.content
    label = f"{message.lane} lane"
    lines: list[str] = []
    for block in message.content:
        if isinstance(block, ThinkingBlock):
            lines.append(thinking_as_text(message, block).text)
        elif isinstance(block, TextBlock):
            lines.append(block.text)
        elif isinstance(block, ToolUseBlock):
            foreign_calls[block.id] = (label, block.name)
            lines.append(f"[{label} called {block.name}] {json.dumps(block.input)}")
    return "\n\n".join(line for line in lines if line.strip())


def _results_as_text(
    blocks: list[ContentBlock], foreign_calls: dict[str, tuple[str, str]]
) -> list[ContentBlock]:
    """_results_as_text turns results of foreign tool calls into labelled text,
    leaving results of the provider's own calls untouched."""
    out: list[ContentBlock] = []
    for block in blocks:
        if isinstance(block, ToolResultBlock) and block.tool_use_id in foreign_calls:
            label, name = foreign_calls[block.tool_use_id]
            status = "error" if block.is_error else "result"
            out.append(TextBlock(type="text", text=f"[{label} {name} {status}]\n{block.content}"))
        else:
            out.append(block)
    return out


def merge_turns(messages: list[Message], *, split_before_thinking: bool) -> list[Message]:
    """merge_turns folds consecutive same-role turns into one so roles alternate.

    Merged user turns put tool_result blocks first, as providers require right
    after a tool call. Assistant turns from different providers or lanes are
    never merged; a CONTINUE user turn separates them so each keeps its own
    provider state. With split_before_thinking, an assistant turn that opens
    with thinking is also kept whole behind a CONTINUE turn (Anthropic wants
    thinking to lead the turn it belongs to). A merged turn keeps the
    cache_breakpoint mark of any turn folded into it.
    """
    out: list[Message] = []
    for message in messages:
        prev = out[-1] if out else None
        if prev is None or prev.role != message.role:
            out.append(message)
            continue
        if message.role == "assistant":
            opens_with_thinking = (
                not isinstance(message.content, str)
                and bool(message.content)
                and message.content[0].type in ("thinking", "redacted_thinking")
            )
            if (
                (split_before_thinking and opens_with_thinking)
                or prev.provider != message.provider
                or prev.lane != message.lane
            ):
                out.append(Message(role="user", content=CONTINUE))
                out.append(message)
                continue
            out[-1] = Message(
                role="assistant",
                content=_blocks(prev) + _blocks(message),
                provider=prev.provider,
                lane=prev.lane,
                cache_breakpoint=prev.cache_breakpoint or message.cache_breakpoint,
            )
            continue
        blocks = _blocks(prev) + _blocks(message)
        results = [b for b in blocks if isinstance(b, ToolResultBlock)]
        rest = [b for b in blocks if not isinstance(b, ToolResultBlock)]
        out[-1] = Message(
            role="user",
            content=[*results, *rest],
            cache_breakpoint=prev.cache_breakpoint or message.cache_breakpoint,
        )
    return out


def _blocks(message: Message) -> list[ContentBlock]:
    """_blocks returns a message's content as a block list."""
    if isinstance(message.content, str):
        return [TextBlock(type="text", text=message.content)]
    return list(message.content)
