"""Typed process config. The only module that reads the environment or config.toml."""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_SERVICE_ROOT = Path(__file__).resolve().parent
_TOML_PATH = _SERVICE_ROOT / "config.toml"


class Env(BaseSettings):
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
    provider: str
    model: str
    max_tokens: int
    thinking_budget: int | None = None

    @field_validator("provider", "model")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("max_tokens")
    @classmethod
    def positive_tokens(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("thinking_budget")
    @classmethod
    def positive_budget(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("must be >= 1")
        return value


class Embed(BaseModel):
    provider: str
    model: str
    dimensions: int
    max_batch_size: int
    max_text_length: int

    @field_validator("provider", "model")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("dimensions", "max_batch_size", "max_text_length")
    @classmethod
    def positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value


class ImageDescribe(BaseModel):
    provider: str
    model: str
    max_tokens: int
    max_bytes: int
    max_prompt_length: int
    allowed_media_types: tuple[str, ...]

    @field_validator("provider", "model")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("max_tokens", "max_bytes", "max_prompt_length")
    @classmethod
    def positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("allowed_media_types")
    @classmethod
    def media_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("must not be empty")
        if any(not item.strip() for item in value):
            raise ValueError("must not contain empty values")
        return value


class ImageCreate(BaseModel):
    provider: str
    model: str
    max_prompt_length: int

    @field_validator("provider", "model")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("max_prompt_length")
    @classmethod
    def positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value


class SpeechTranscribe(BaseModel):
    provider: str
    model: str
    language: str
    max_bytes: int
    allowed_media_types: tuple[str, ...]

    @field_validator("provider", "model", "language")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value

    @field_validator("max_bytes")
    @classmethod
    def positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("allowed_media_types")
    @classmethod
    def media_types(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("must not be empty")
        if any(not item.strip() for item in value):
            raise ValueError("must not contain empty values")
        return value


class Retry(BaseModel):
    attempts: int
    backoff_seconds: tuple[float, ...]
    timeout_seconds: float

    @field_validator("attempts")
    @classmethod
    def positive_attempts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("backoff_seconds")
    @classmethod
    def valid_backoff(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value:
            raise ValueError("must not be empty")
        if any(item < 0 for item in value):
            raise ValueError("values must be >= 0")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("must be > 0")
        return value


class ChatFile(BaseModel):
    reasoning: ChatEndpoint
    conversation: ChatEndpoint


class ImageFile(BaseModel):
    describe: ImageDescribe
    create: ImageCreate


class SpeechFile(BaseModel):
    transcribe: SpeechTranscribe


class FileConfig(BaseModel):
    chat: ChatFile
    embed: Embed
    image: ImageFile
    speech: SpeechFile
    retry: Retry


class Config(BaseModel):
    env: Env
    chat: ChatFile
    embed: Embed
    image: ImageFile
    speech: SpeechFile
    retry: Retry


def _load_toml() -> FileConfig:
    if not _TOML_PATH.is_file():
        raise FileNotFoundError(f"missing config file: {_TOML_PATH}")
    with _TOML_PATH.open("rb") as fh:
        raw = tomllib.load(fh)
    return FileConfig.model_validate(raw)


@lru_cache
def get_config() -> Config:
    file_cfg = _load_toml()
    return Config(
        env=Env(),
        chat=file_cfg.chat,
        embed=file_cfg.embed,
        image=file_cfg.image,
        speech=file_cfg.speech,
        retry=file_cfg.retry,
    )
