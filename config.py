"""Typed process config. Every model, budget and limit is fixed here in code, so changing one is a push to Dwar; only provider keys come from the environment."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Env(BaseSettings):
    """Provider API keys from the environment. A missing key fails the request that needs it (503 provider_unconfigured)."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    gemini_api_key: str = ""
    openai_api_key: str = ""
    deepgram_api_key: str = ""


class ChatEndpoint(BaseModel):
    """One chat endpoint: provider, model, output budget, and Anthropic thinking effort (None for other providers)."""

    provider: Literal["anthropic", "gemini"]
    model: str
    max_tokens: int
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None

    @model_validator(mode="after")
    def effort_matches_provider(self) -> ChatEndpoint:
        if self.provider == "anthropic" and self.effort is None:
            raise ValueError("anthropic chat endpoints require effort")
        if self.provider != "anthropic" and self.effort is not None:
            raise ValueError(f"effort applies only to anthropic, not {self.provider}")
        return self


class Embed(BaseModel):
    provider: Literal["openai"]
    model: str
    dimensions: int
    max_batch_size: int
    max_text_length: int


class ImageDescribe(BaseModel):
    provider: Literal["gemini"]
    model: str
    max_tokens: int
    max_bytes: int
    max_prompt_length: int
    allowed_media_types: tuple[str, ...]


class ImageCreate(BaseModel):
    provider: Literal["gemini"]
    model: str
    max_prompt_length: int


class SpeechTranscribe(BaseModel):
    provider: Literal["deepgram"]
    model: str
    language: str
    max_bytes: int
    allowed_media_types: tuple[str, ...]


class Retry(BaseModel):
    """Bounded retries for transport failures: attempt count, backoff per attempt, per-call timeout."""

    attempts: int
    backoff_seconds: tuple[float, ...]
    timeout_seconds: float


class Web(BaseModel):
    """Hath's web tools (inference/web.py): the small model that runs searches and answers
    questions from fetched pages, and the bounds on what one call may fetch."""

    model: str
    search_tool: str
    search_max_tokens: int
    answer_max_tokens: int
    answer_page_chars: int
    max_page_chars: int
    max_page_bytes: int
    max_urls: int
    max_parallel: int
    max_redirects: int
    fetch_timeout_seconds: float
    user_agent: str


class Chat(BaseModel):
    """The chat endpoints: an agent's two lanes, work (the job, with its tools) and
    process (reading what arrives and answering it), plus complete for callers that
    bring their own system prompt."""

    work: ChatEndpoint
    process: ChatEndpoint
    complete: ChatEndpoint


class Image(BaseModel):
    describe: ImageDescribe
    create: ImageCreate


class Speech(BaseModel):
    transcribe: SpeechTranscribe


class Config(BaseModel):
    """The service config: environment keys plus the settings below."""

    env: Env
    chat: Chat
    embed: Embed
    image: Image
    speech: Speech
    retry: Retry
    web: Web


CHAT = Chat(
    work=ChatEndpoint(provider="anthropic", model="claude-haiku-5-5", max_tokens=32768, effort="xhigh"),
    process=ChatEndpoint(provider="anthropic", model="claude-haiku-5-5", max_tokens=32768, effort="medium"),
    complete=ChatEndpoint(provider="gemini", model="gemini-3.1-pro-preview", max_tokens=16384, effort=None),
)

EMBED = Embed(
    provider="openai",
    model="text-embedding-3-small",
    dimensions=1536,
    max_batch_size=64,
    max_text_length=8192,
)

IMAGE = Image(
    describe=ImageDescribe(
        provider="gemini",
        model="gemini-3.6-flash",
        max_tokens=2048,
        max_bytes=10_485_760,
        max_prompt_length=4000,
        allowed_media_types=("image/jpeg", "image/png", "image/webp", "image/gif"),
    ),
    create=ImageCreate(provider="gemini", model="gemini-3.1-flash-image", max_prompt_length=4000),
)

SPEECH = Speech(
    transcribe=SpeechTranscribe(
        provider="deepgram",
        model="nova-3",
        language="multi",
        max_bytes=26_214_400,
        allowed_media_types=(
            "audio/wav",
            "audio/mpeg",
            "audio/mp4",
            "audio/ogg",
            "audio/webm",
            "audio/flac",
            "audio/aac",
            "audio/x-wav",
        ),
    ),
)

RETRY = Retry(attempts=3, backoff_seconds=(0.5, 1.0, 2.0), timeout_seconds=300)

WEB = Web(
    model="claude-haiku-5-5",
    search_tool="web_search_20250305",
    search_max_tokens=2048,
    answer_max_tokens=1024,
    answer_page_chars=60_000,
    max_page_chars=200_000,
    max_page_bytes=5_000_000,
    max_urls=10,
    max_parallel=5,
    max_redirects=5,
    fetch_timeout_seconds=20.0,
    user_agent="Mozilla/5.0 (compatible; DadiFetch/1.0; +https://github.com/dadi-os)",
)


@lru_cache
def get_config() -> Config:
    return Config(env=Env(), chat=CHAT, embed=EMBED, image=IMAGE, speech=SPEECH, retry=RETRY, web=WEB)
