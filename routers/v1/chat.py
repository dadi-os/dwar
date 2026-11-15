from fastapi import APIRouter

from inference import complete_agent, complete_text
from inference.types import ChatRequest as InternalChatRequest
from inference.types import ChatResponse as InternalChatResponse
from routers.v1.schemas import ChatRequest, ChatResponse

router = APIRouter()


def _to_internal(request: ChatRequest) -> InternalChatRequest:
    return InternalChatRequest.model_validate(request.model_dump())


def _to_http(response: InternalChatResponse) -> ChatResponse:
    return ChatResponse.model_validate(response.model_dump(exclude={"provider_stop_reason"}))


@router.post("/agent", response_model=ChatResponse)
def agent(body: ChatRequest) -> ChatResponse:
    return _to_http(complete_agent(_to_internal(body)))


@router.post("/complete", response_model=ChatResponse)
def complete(body: ChatRequest) -> ChatResponse:
    return _to_http(complete_text(_to_internal(body)))
