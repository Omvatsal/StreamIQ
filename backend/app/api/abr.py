"""Adaptive Bitrate (ABR) inference endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.app.services.rl_service import rl_service

router = APIRouter(prefix="/abr", tags=["ABR Policy"])


class DecisionRequest(BaseModel):
    buffer_occupancy_seconds: float = Field(
        ..., ge=0.0, description="Current client video buffer occupancy in seconds"
    )
    throughput_history_mbps: List[float] = Field(
        default_factory=list, description="Recent segment throughputs in Mbps"
    )
    previous_bitrate_mbps: Optional[float] = Field(
        default=None, description="Bitrate of the segment downloaded immediately prior"
    )
    previous_download_duration_seconds: float = Field(
        default=0.0, ge=0.0, description="Time taken to download the last segment"
    )
    remaining_segments: int = Field(
        default=10, ge=0, description="Number of segments remaining in playback"
    )


@router.post("/predict")
async def predict_bitrate(request: DecisionRequest) -> Dict[str, Any]:
    """Query the trained DQN model for the recommended bitrate action."""
    return rl_service.predict(
        buffer_occupancy_seconds=request.buffer_occupancy_seconds,
        throughput_history=request.throughput_history_mbps,
        previous_bitrate_mbps=request.previous_bitrate_mbps,
        previous_download_duration_seconds=request.previous_download_duration_seconds,
        remaining_segments=request.remaining_segments,
    )

