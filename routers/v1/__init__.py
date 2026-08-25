from fastapi import APIRouter

from routers.v1.chat import router as chat_router
from routers.v1.embed import router as embed_router

router = APIRouter()
router.include_router(chat_router, prefix="/chat")
router.include_router(embed_router)
