import base64

from fastapi import APIRouter

from inference import create_image, describe_image
from routers.v1.schemas import (
    CreateRequest,
    CreateResponse,
    DescribeRequest,
    DescribeResponse,
    MediaBlob,
    Usage,
)

router = APIRouter()


@router.post("/describe", response_model=DescribeResponse)
def describe(body: DescribeRequest) -> DescribeResponse:
    result = describe_image(body.image.decoded(), body.image.media_type, body.prompt)
    return DescribeResponse(
        description=result.description,
        usage=Usage(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        ),
    )


@router.post("/create", response_model=CreateResponse)
def create(body: CreateRequest) -> CreateResponse:
    result = create_image(body.prompt)
    return CreateResponse(
        image=MediaBlob(
            media_type=result.media_type,
            data=base64.b64encode(result.image).decode("ascii"),
        ),
        usage=Usage(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
        ),
    )
