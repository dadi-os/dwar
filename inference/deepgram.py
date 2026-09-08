"""Deepgram adapter. This is the only module that imports the Deepgram SDK."""

from __future__ import annotations

import httpx
from deepgram import DeepgramClient
from deepgram.core.api_error import ApiError

from errors import DwarError, TransportError
from inference.types import TranscribeResult


class DeepgramAdapter:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        language: str,
        timeout_seconds: float,
    ) -> None:
        self._model = model
        self._language = language
        self._timeout = timeout_seconds
        self._client = DeepgramClient(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=0,
        )

    def transcribe(self, audio: bytes, media_type: str) -> TranscribeResult:
        try:
            response = self._client.listen.v1.media.transcribe_file(
                request=audio,
                model=self._model,
                language=self._language,
                smart_format=True,
                request_options={
                    "timeout": self._timeout,
                    "max_retries": 0,
                    "additional_headers": {"Content-Type": media_type},
                },
            )
        except ApiError as exc:
            message = _message(exc)
            code = exc.status_code
            if code is None:
                raise TransportError(502, "upstream_unreachable", message) from exc
            if code == 429:
                raise TransportError(429, "rate_limit", message) from exc
            if code in (408, 504):
                raise TransportError(504, "upstream_timeout", message) from exc
            if code >= 500:
                raise TransportError(502, "provider", message) from exc
            if code == 400:
                raise DwarError(422, "invalid_request", message) from exc
            raise DwarError(502, "provider", message) from exc
        except httpx.TimeoutException as exc:
            raise TransportError(504, "upstream_timeout", str(exc)) from exc
        except httpx.RequestError as exc:
            raise TransportError(502, "upstream_unreachable", str(exc)) from exc

        try:
            text = response.results.channels[0].alternatives[0].transcript
        except (AttributeError, IndexError, TypeError) as exc:
            raise DwarError(
                502, "provider", "Deepgram response missing transcript"
            ) from exc
        if text is None:
            raise DwarError(502, "provider", "Deepgram response missing transcript")

        metadata = response.metadata
        if metadata is None or metadata.duration is None:
            raise DwarError(502, "provider", "Deepgram response missing duration")

        return TranscribeResult(text=text, duration_seconds=float(metadata.duration))


def _message(exc: ApiError) -> str:
    body = exc.body
    if isinstance(body, dict):
        err_msg = body.get("err_msg")
        if isinstance(err_msg, str) and err_msg:
            return err_msg
    if isinstance(body, str) and body:
        return body
    return str(exc)
