"""FastAPI application: routes, exception handlers, request logging."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from config import get_config
from errors import DwarError
from logutil import configure_logging, log_extra
from routers.v1 import router as v1_router

configure_logging()
logger = logging.getLogger("dwar")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_config()
    yield


def create_app() -> FastAPI:
    """Build the Dwar ASGI app."""
    app = FastAPI(title="Dwar", lifespan=lifespan)
    _install_middleware(app)
    _register_exception_handlers(app)
    app.include_router(v1_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_log(request: Request, call_next) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        start = time.perf_counter()
        response: Response | None = None
        try:
            response = await call_next(request)
            return response
        finally:
            status = response.status_code if response is not None else 500
            duration_ms = int((time.perf_counter() - start) * 1000)
            level = logging.INFO
            if status >= 500:
                level = logging.ERROR
            elif status >= 400:
                level = logging.WARNING
            logger.log(
                level,
                "request",
                extra=log_extra(
                    request_id=request_id,
                    method=request.method,
                    path=request.url.path,
                    status=status,
                    duration_ms=duration_ms,
                ),
            )
            if response is not None:
                response.headers["X-Request-Id"] = request_id


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DwarError)
    async def handle_dwar_error(request: Request, exc: DwarError) -> JSONResponse:
        logger.log(
            logging.ERROR if exc.status_code >= 500 else logging.WARNING,
            exc.message,
            extra=log_extra(
                code=exc.type,
                request_id=request.headers.get("x-request-id"),
                status=exc.status_code,
            ),
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"type": exc.type, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        parts = []
        for err in exc.errors():
            loc = ".".join(str(item) for item in err["loc"] if item != "body")
            parts.append(f"{loc}: {err['msg']}" if loc else err["msg"])
        message = "; ".join(parts)
        logger.warning(
            message,
            extra=log_extra(
                code="invalid_request",
                request_id=request.headers.get("x-request-id"),
                status=422,
            ),
        )
        return JSONResponse(
            status_code=422,
            content={"error": {"type": "invalid_request", "message": message}},
        )


app = create_app()
