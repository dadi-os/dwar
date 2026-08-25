from fastapi import APIRouter

from inference import transcribe as run_transcribe
from routers.v1.schemas import TranscribeRequest, TranscribeResponse

router = APIRouter()


@router.post("/transcribe", response_model=TranscribeResponse)
def transcribe(body: TranscribeRequest) -> TranscribeResponse:
    result = run_transcribe(body.audio.decoded(), body.audio.media_type)
    return TranscribeResponse(text=result.text, duration_seconds=result.duration_seconds)
