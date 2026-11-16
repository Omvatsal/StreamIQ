"""Streaming API routes."""

from pathlib import Path
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.schemas.streaming import DecisionRequest, PlaybackSegment
from backend.app.services.dqn_controller import (
    DQNController,
    DecisionRequest as ControllerDecisionRequest,
)
from backend.app.services.playback_metrics import SegmentPlayback, summarize_playback
from backend.app.services.streaming_service import StreamingService
from backend.app.config import get_settings


router = APIRouter(prefix="/api")
settings = get_settings()
logger = logging.getLogger(__name__)
streaming = StreamingService(settings.processed_video_root)
controller = DQNController(
    settings.dqn_checkpoint
)


@router.get("/videos")
async def videos():
    return {"videos": streaming.list_videos()}


@router.get("/videos/{video_id}/manifest")
async def manifest(video_id: str):
    try:
        return FileResponse(streaming.manifest_path(video_id), media_type="application/dash+xml")
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/abr/decision")
async def abr_decision(request: DecisionRequest):
    return controller.decide(
        ControllerDecisionRequest(
            buffer_seconds=request.buffer_seconds,
            throughput_history_mbps=tuple(request.throughput_history_mbps),
            previous_bitrate_mbps=request.previous_bitrate_mbps,
            previous_download_duration_seconds=request.previous_download_duration_seconds,
            remaining_segments=request.remaining_segments,
        )
    )


@router.post("/playback/metrics")
async def playback_metrics(segments: list[PlaybackSegment]):
    return summarize_playback(
        [SegmentPlayback(**segment.model_dump()) for segment in segments]
    )
