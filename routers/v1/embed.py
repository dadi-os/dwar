from fastapi import APIRouter

from inference import embed as run_embed
from routers.v1.schemas import EmbedRequest, EmbedResponse, EmbedUsage

router = APIRouter()


@router.post("/embed", response_model=EmbedResponse)
def embed(body: EmbedRequest) -> EmbedResponse:
    result = run_embed(body.texts)
    return EmbedResponse(
        embeddings=result.embeddings,
        dimensions=result.dimensions,
        usage=EmbedUsage(input_tokens=result.input_tokens),
    )
