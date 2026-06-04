"""FastAPI application factory."""

from fastapi import FastAPI

from src.api.errors import register_exception_handlers
from src.api.lifespan import lifespan
from src.api.middleware import register_middleware
from src.api.routes import router
from src.api.settings import ApiSettings


def create_app(settings: ApiSettings | None = None) -> FastAPI:
    """Create and configure the document classification API app."""

    app = FastAPI(
        title="Document Classification API",
        description=(
            "Synchronous API for classifying a single text document with a "
            "preloaded scikit-learn model."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    if settings is not None:
        app.state.settings = settings

    register_exception_handlers(app)
    register_middleware(app)
    app.include_router(router)
    return app


app = create_app()
