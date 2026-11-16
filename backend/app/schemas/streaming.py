"""API schemas for streaming and DQN decisions."""

from pydantic import BaseModel, Field


class DecisionRequest(BaseModel):
    buffer_seconds: float = Field(ge=0)
    throughput_history_mbps: list[float] = Field(min_length=5, max_length=5)
    previous_bitrate_mbps: float = Field(ge=0)
    previous_download_duration_seconds: float = Field(ge=0)
    remaining_segments: float = Field(ge=0)


class PlaybackSegment(BaseModel):
    segment_index: int = Field(ge=0)
    bitrate_mbps: float = Field(ge=0)
    throughput_mbps: float = Field(ge=0)
    download_duration_seconds: float = Field(ge=0)
    buffer_seconds: float = Field(ge=0)
    rebuffer_seconds: float = Field(default=0, ge=0)
