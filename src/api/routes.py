"""Aggregate API routers for application registration."""

from fastapi import APIRouter

from src.api.routes_classification import router as classification_router
from src.api.routes_health import router as health_router


router = APIRouter()
router.include_router(health_router)
router.include_router(classification_router)
