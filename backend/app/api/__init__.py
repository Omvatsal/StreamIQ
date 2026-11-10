"""API router initialization."""

from fastapi import APIRouter
from .abr import router as abr_router
from .traces import router as traces_router
from .simulation import router as simulation_router
from .video import router as video_router

api_router = APIRouter(prefix="/api")
api_router.include_router(traces_router)
api_router.include_router(abr_router)
api_router.include_router(simulation_router)
api_router.include_router(video_router)

__all__ = ["api_router"]

