"""Behavior tests for thinking round-trip and cross-provider history shaping."""

from __future__ import annotations

from types import SimpleNamespace

from inference.types import ChatRequest

_USAGE = SimpleNamespace(
    input_tokens=1,
    output_tokens=1,
    cache_read_input_tokens=0,
    cache_creation_input_tokens=0,
)


def _anthropic(sent: dict[str, object], content: list[SimpleNamespace]):
    from inference.anthropic import AnthropicAdapter

    def create(**kwargs: object) -> SimpleNamespace:
        sent.update(kwargs)
        return SimpleNamespace(content=content, stop_reason="tool_use", usage=_USAGE)

    adapter = AnthropicAdapter(api_key="k", model="m", max_tokens=10, timeout_seconds=1)
    adapter._client = SimpleNamespace(messages=SimpleNamespace(create=create))
    return adapter


def _tools() -> list[dict[str, object]]:
    return [{"name": "browser_screenshot", "description": "d", "input_schema": {"type": "object"}}]


def test_anthropic_returns_thinking_text_and_tool_in_one_turn() -> None:
    sent: dict[str, object] = {}
    adapter = _anthropic(
        sent,
        [
            SimpleNamespace(type="thinking", thinking="tree is empty", signature="sig-1"),
            SimpleNamespace(type="text", text="Accessibility tree appears empty, taking a screenshot instead."),
            SimpleNamespace(type="tool_use", id="t1", name="browser_screenshot", input={}),
        ],
    )
    request = ChatRequest.model_validate(
        {
            "system": "sys",
            "tool_choice": "auto",
            "tools": _tools(),
            "messages": [{"role": "user", "content": "check the inbox"}],
        }
    )

    response = adapter.complete(request, "lane")

    assert sent["tool_choice"] == {"type": "auto"}
    assert sent["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert response.provider == "anthropic"
    assert [block.type for block in response.content] == ["thinking", "text", "tool_use"]
    assert response.content[0].signature == "sig-1"


def test_anthropic_replays_own_thinking_and_turns_gemini_turns_into_text() -> None:
    sent: dict[str, object] = {}
    adapter = _anthropic(sent, [SimpleNamespace(type="tool_use", id="t9", name="yield", input={})])
    request = ChatRequest.model_validate(
        {
            "system": "sys",
            "tool_choice": "auto",
            "tools": _tools(),
            "messages": [
                {"role": "user", "content": "[From: Ankur]\ncheck my email"},
                {
                    "role": "assistant",
                    "provider": "gemini",
                    "lane": "conversation",
                    "content": [
                        {"type": "thinking", "thinking": "route to reasoning"},
                        {"type": "tool_use", "id": "c1", "name": "steer_reasoning", "input": {"instruction": "open Gmail"}, "thought_signature": "c2ln"},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "queued"}]},
                {
                    "role": "assistant",
                    "provider": "anthropic",
                    "lane": "reasoning",
                    "content": [
                        {"type": "thinking", "thinking": "open gmail first", "signature": "sig-a"},
                        {"type": "tool_use", "id": "r1", "name": "browser_screenshot", "input": {}},
                    ],
                },
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "r1", "content": "a page", "is_error": False}]},
            ],
        }
    )

    adapter.complete(request, "lane")

    messages = sent["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    foreign = messages[1]["content"]
    assert isinstance(foreign, str)
    assert "[conversation lane thinking]\nroute to reasoning" in foreign
    assert "[conversation lane called steer_reasoning]" in foreign
    assert messages[2]["content"][0] == {"type": "text", "text": "[conversation lane steer_reasoning result]\nqueued"}
    assert messages[3]["content"][0] == {"type": "thinking", "thinking": "open gmail first", "signature": "sig-a"}
    assert messages[4]["content"][0]["type"] == "tool_result"


def test_anthropic_keeps_thinking_leading_its_turn_after_other_assistant_text() -> None:
    sent: dict[str, object] = {}
    adapter = _anthropic(sent, [SimpleNamespace(type="tool_use", id="t9", name="yield", input={})])
    request = ChatRequest.model_validate(
        {
            "system": "sys",
            "tool_choice": "auto",
            "tools": _tools(),
            "messages": [
                {"role": "user", "content": "[From: Ankur]\nhi"},
                {"role": "assistant", "content": "[To: Ankur]\non it"},
                {
                    "role": "assistant",
                    "provider": "anthropic",
                    "lane": "reasoning",
                    "content": [
                        {"type": "thinking", "thinking": "plan", "signature": "s"},
                        {"type": "text", "text": "Starting."},
                    ],
                },
                {"role": "user", "content": "[continue]"},
            ],
        }
    )

    adapter.complete(request, "lane")

    messages = sent["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[2]["content"] == "[continue]"
    assert messages[3]["content"][0]["type"] == "thinking"


def test_gemini_gives_claude_tool_calls_the_placeholder_signature() -> None:
    from google.genai import types as genai_types

    from inference.gemini import _SKIP_SIGNATURE, _to_contents
    from inference.types import Message

    messages = [
        Message.model_validate({"role": "user", "content": "check my email"}),
        Message.model_validate(
            {
                "role": "assistant",
                "provider": "anthropic",
                "lane": "reasoning",
                "content": [
                    {"type": "thinking", "thinking": "browser is down", "signature": "sig-a"},
                    {"type": "redacted_thinking", "data": "xyz"},
                    {"type": "tool_use", "id": "r1", "name": "browser_navigate", "input": {"url": "u"}},
                ],
            }
        ),
        Message.model_validate(
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "r1", "content": "not_found", "is_error": True}]}
        ),
        Message.model_validate(
            {
                "role": "assistant",
                "provider": "gemini",
                "lane": "conversation",
                "content": [
                    {"type": "thinking", "thinking": "tell Ankur", "signature": "c2ln"},
                    {"type": "tool_use", "id": "c1", "name": "dispatch_message", "input": {}, "thought_signature": "c2ln"},
                ],
            }
        ),
    ]

    contents = _to_contents(messages)

    claude = contents[1].parts
    assert claude[0].text == "[reasoning lane thinking]\nbrowser is down"
    assert not claude[0].thought
    assert len(claude) == 2
    assert claude[1].function_call.name == "browser_navigate"
    assert claude[1].thought_signature == _SKIP_SIGNATURE
    assert contents[2].parts[0].function_response.response == {"error": "not_found"}
    own = contents[3].parts
    assert own[0].thought is True and own[0].thought_signature == b"sig"
    assert own[1].thought_signature == b"sig"
    assert isinstance(own[1], genai_types.Part)


def test_merged_user_turns_put_tool_results_first() -> None:
    from inference.history import merge_turns
    from inference.types import Message

    merged = merge_turns(
        [
            Message.model_validate({"role": "user", "content": "[Steer]\nstop clicking"}),
            Message.model_validate(
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}
            ),
        ],
        split_before_thinking=True,
    )

    assert len(merged) == 1
    assert [block.type for block in merged[0].content] == ["tool_result", "text"]


def test_tool_choice_is_required() -> None:
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ChatRequest.model_validate({"system": "s", "messages": [{"role": "user", "content": "hi"}]})
