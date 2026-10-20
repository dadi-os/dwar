"""Hath's web tools: one search, or a batch of public pages fetched as text (see inference/web.py)."""

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from config import get_config
from errors import DwarError
from inference import web

router = APIRouter()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    max_results: int = Field(default=10, ge=1, le=20)
    allowed_domains: list[str] | None = None
    blocked_domains: list[str] | None = None


class FetchRequest(BaseModel):
    urls: list[str] = Field(min_length=1)
    question: str | None = Field(default=None, max_length=2000)

    @field_validator("urls")
    @classmethod
    def unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("urls must not repeat")
        return value


@router.post("/search")
def search(body: SearchRequest) -> dict[str, Any]:
    if body.allowed_domains and body.blocked_domains:
        raise DwarError(422, "invalid_request", "pass allowed_domains or blocked_domains, not both")
    return web.search(body.query, body.max_results, body.allowed_domains, body.blocked_domains)


@router.post("/fetch")
def fetch(body: FetchRequest) -> dict[str, Any]:
    limit = get_config().web.max_urls
    if len(body.urls) > limit:
        raise DwarError(422, "invalid_request", f"at most {limit} urls per fetch")
    return web.fetch(body.urls, body.question)
