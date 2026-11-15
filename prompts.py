from pathlib import Path

_PROMPTS = Path(__file__).resolve().parent / "prompts"
_cache: dict[str, tuple[float, str]] = {}


def _load(name: str) -> str:
    """Read a prompt file, cached until its mtime changes, so prompt edits apply without a restart. An empty file is an error."""
    path = _PROMPTS / name
    mtime = path.stat().st_mtime
    cached = _cache.get(name)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise RuntimeError(f"prompt {name} is empty")
    _cache[name] = (mtime, text)
    return text


def agent_prompt_block() -> str:
    """The system block Dwar prepends on /chat/agent (`prompts/agent.txt`)."""
    return _load("agent.txt")


def describe_instruction() -> str:
    """The default image describe instruction (`prompts/describe.txt`)."""
    return _load("describe.txt")
