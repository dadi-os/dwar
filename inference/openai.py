"""OpenAI adapter. This is the only module that imports the OpenAI SDK."""

from __future__ import annotations

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from errors import DwarError, TransportError
from inference.types import EmbedResult


class OpenAIAdapter:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        timeout_seconds: float,
    ) -> None:
        self._model = model
        self._dimensions = dimensions
        self._client = OpenAI(api_key=api_key, max_retries=0, timeout=timeout_seconds)

    def embed(self, texts: list[str]) -> EmbedResult:
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=texts,
                dimensions=self._dimensions,
            )
        except RateLimitError as exc:
            raise TransportError(429, "rate_limit", str(exc)) from exc
        except APITimeoutError as exc:
            raise TransportError(504, "timeout", str(exc)) from exc
        except APIConnectionError as exc:
            raise TransportError(502, "connection", str(exc)) from exc
        except APIStatusError as exc:
            if exc.status_code == 429:
                raise TransportError(429, "rate_limit", str(exc)) from exc
            if exc.status_code in (408, 504):
                raise TransportError(504, "timeout", str(exc)) from exc
            if exc.status_code >= 500:
                raise TransportError(502, "provider", str(exc)) from exc
            raise DwarError(502, "provider", str(exc)) from exc
        except httpx.TimeoutException as exc:
            raise TransportError(504, "timeout", str(exc)) from exc
        except httpx.RequestError as exc:
            raise TransportError(502, "connection", str(exc)) from exc

        if not response.data:
            raise DwarError(502, "provider", "OpenAI returned no embeddings")
        ordered = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in ordered] != list(range(len(texts))):
            raise DwarError(502, "provider", "OpenAI embedding indexes do not match the input batch")
        embeddings = [list(item.embedding) for item in ordered]
        if len(embeddings) != len(texts):
            raise DwarError(502, "provider", "OpenAI embedding count does not match the input batch")
        widths = {len(vector) for vector in embeddings}
        if len(widths) != 1:
            raise DwarError(502, "provider", "OpenAI returned embeddings of mixed dimensions")
        if response.usage is None:
            raise DwarError(502, "provider", "OpenAI embedding response missing usage")
        return EmbedResult(
            embeddings=embeddings,
            dimensions=next(iter(widths)),
            input_tokens=response.usage.prompt_tokens,
        )
