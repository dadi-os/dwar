from pathlib import Path

_PROMPTS = Path(__file__).resolve().parent / "prompts"


def _load(name: str) -> str:
    text = (_PROMPTS / name).read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"lane prompt {name} is empty")
    return text


REASONING_LANE_BLOCK = _load("reasoning.txt")
CONVERSATION_LANE_BLOCK = _load("conversation.txt")
