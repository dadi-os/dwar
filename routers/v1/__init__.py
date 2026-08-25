from fastapi import APIRouter

from routers.v1.chat import router as chat_router
from routers.v1.embed import router as embed_router
from routers.v1.image import router as image_router
from routers.v1.speech import router as speech_router

router = APIRouter()
router.include_router(chat_router, prefix="/chat")
router.include_router(embed_router)
router.include_router(image_router, prefix="/image")
router.include_router(speech_router, prefix="/speech")
