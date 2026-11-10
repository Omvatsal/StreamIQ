"""StreamIQ FastAPI application entrypoint."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api import api_router
from backend.app.core.config import settings, PROJECT_ROOT
from backend.app.services.rl_service import rl_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure the trained DQN model is warm and loaded into memory
    _ = rl_service.predict(
        buffer_occupancy_seconds=10.0,
        throughput_history=[2.5, 3.0, 2.8, 3.2, 3.5],
        previous_bitrate_mbps=2.5,
        previous_download_duration_seconds=2.0,
        remaining_segments=10,
    )
    yield


app = FastAPI(
    title=settings.app_name,
    description="RL-Driven Adaptive Video Streaming Platform API",
    version=settings.app_version,
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include main API router
app.include_router(api_router)

# Mount frontend production build if available
frontend_dist = PROJECT_ROOT / "frontend" / "dist"
if frontend_dist.exists():
    assets_dir = frontend_dist / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")
    app.mount("/ui", StaticFiles(directory=str(frontend_dist), html=True), name="ui")


@app.get("/")
async def root():
    return {
        "project": "StreamIQ",
        "status": "running",
        "version": settings.app_version,
        "ui": "/ui/",
        "docs": "/docs",
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": rl_service.agent is not None,
    }
