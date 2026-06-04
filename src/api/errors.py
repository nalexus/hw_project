from fastapi import FastAPI, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    if any(error.get("type") == "json_invalid" for error in exc.errors()):
        return JSONResponse(
            status_code=422,
            content={"detail": "Malformed JSON request body"},
        )
    return await request_validation_exception_handler(request, exc)


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
