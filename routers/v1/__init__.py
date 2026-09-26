from typing import Annotated

from fastapi import APIRouter, Depends, Header

from errors import DwarError
from logutil import caller_var
from routers.v1.chat import router as chat_router
from routers.v1.embed import router as embed_router
from routers.v1.image import router as image_router
from routers.v1.speech import router as speech_router


async def require_caller(x_dadi_caller: Annotated[str, Header()]) -> None:
    """require_caller demands X-Dadi-Caller (e.g. `dimaag/browser-manager`) so
    every inference log line is attributable to the service and agent that paid for it."""
    if not x_dadi_caller.strip():
        raise DwarError(422, "invalid_request", "X-Dadi-Caller must not be empty")
    caller_var.set(x_dadi_caller)


router = APIRouter(dependencies=[Depends(require_caller)])
router.include_router(chat_router, prefix="/chat")
router.include_router(embed_router)
router.include_router(image_router, prefix="/image")
router.include_router(speech_router, prefix="/speech")
