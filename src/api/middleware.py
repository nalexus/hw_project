from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


CLASSIFY_DOCUMENT_PATH = "/classify_document"


def is_json_content_type(content_type: str | None) -> bool:
    if content_type is None:
        return False
    media_type = content_type.split(";", maxsplit=1)[0].strip().lower()
    return media_type == "application/json"


def register_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def enforce_json_content_type(request: Request, call_next):
        if (
            request.method == "POST"
            and request.url.path == CLASSIFY_DOCUMENT_PATH
            and not is_json_content_type(request.headers.get("content-type"))
        ):
            return JSONResponse(
                status_code=415,
                content={"detail": "Content-Type must be application/json"},
            )
        return await call_next(request)
