from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api.streaming import router as streaming_router
from backend.app.api.sessions import router as sessions_router
from backend.app.config import get_settings
from backend.app.logging_config import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

app = FastAPI(
    title="StreamIQ API",
    description="RL-driven adaptive video streaming platform",
    version="0.1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(streaming_router)
app.include_router(sessions_router)

processed_root = settings.processed_video_root
if not processed_root.is_absolute():
    processed_root = Path(__file__).resolve().parents[2] / processed_root
processed_root.mkdir(parents=True, exist_ok=True)
app.mount("/media", StaticFiles(directory=processed_root), name="media")


@app.get("/")
async def root():
    return {
        "project": "StreamIQ",
        "status": "running",
        "version": "0.1.0"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy"
    }


@app.get("/ready")
async def ready():
    from backend.app.api.streaming import controller

    checkpoint_ready = controller.readiness()
    return {
        "status": "ready" if checkpoint_ready else "degraded",
        "model_loaded": checkpoint_ready,
        "video_root": str(processed_root),
    }