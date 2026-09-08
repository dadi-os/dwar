"""JSON stdout logging aligned with the nas contract.

Fields: time (RFC3339), level, service, msg; optional code, request_id,
method, path, status, duration_ms.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record on a single line."""

    _LEVELS = {
        logging.DEBUG: "debug",
        logging.INFO: "info",
        logging.WARNING: "warn",
        logging.ERROR: "error",
        logging.CRITICAL: "error",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": self._LEVELS.get(record.levelno, record.levelname.lower()),
            "service": "dwar",
            "msg": record.getMessage(),
        }
        for key in (
            "code",
            "request_id",
            "method",
            "path",
            "status",
            "duration_ms",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["msg"] = payload["msg"] + "\n" + self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging() -> None:
    """Install JSON logging on the root logger; silence uvicorn access spam."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    for name in ("uvicorn", "uvicorn.error"):
        logger = logging.getLogger(name)
        logger.handlers.clear()
        logger.propagate = True

    access = logging.getLogger("uvicorn.access")
    access.handlers.clear()
    access.propagate = False
    access.disabled = True


def log_extra(**fields: Any) -> dict[str, Any]:
    """Build a logging `extra` dict for structured fields."""
    return fields
