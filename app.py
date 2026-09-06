from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from config import get_config
from errors import DwarError
from routers.v1 import router as v1_router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    get_config()
    yield


def _install_middleware(_app: FastAPI) -> None:
    """Single place to register process-wide middleware.

    Dwar is unauthenticated. Device auth belongs in a Dadi-wide module, not here.
    """


def create_app() -> FastAPI:
    app = FastAPI(title="Dwar", lifespan=lifespan)
    _install_middleware(app)
    _register_exception_handlers(app)
    app.include_router(v1_router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(DwarError)
    async def handle_dwar_error(_request: Request, exc: DwarError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"type": exc.type, "message": exc.message}},
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        parts = []
        for err in exc.errors():
            loc = ".".join(str(item) for item in err["loc"] if item != "body")
            parts.append(f"{loc}: {err['msg']}" if loc else err["msg"])
        return JSONResponse(
            status_code=422,
            content={"error": {"type": "invalid_request", "message": "; ".join(parts)}},
        )


app = create_app()
