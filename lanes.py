from pathlib import Path

_PROMPTS = Path(__file__).resolve().parent / "prompts"
_cache: dict[str, tuple[float, str]] = {}


def _load(name: str) -> str:
    path = _PROMPTS / name
    mtime = path.stat().st_mtime
    cached = _cache.get(name)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"lane prompt {name} is empty")
    _cache[name] = (mtime, text)
    return text


def reasoning_lane_block() -> str:
    return _load("reasoning.txt")


def conversation_lane_block() -> str:
    return _load("conversation.txt")


def describe_instruction() -> str:
    return _load("describe.txt")
