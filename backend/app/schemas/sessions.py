"""Playback session API schemas."""

from pydantic import BaseModel, Field

from backend.app.schemas.streaming import PlaybackSegment


class SessionCreate(BaseModel):
    video_id: str = Field(min_length=1, max_length=128)
    policy: str = Field(pattern=r"^(dqn|dashjs|manual)$")


class SessionResponse(BaseModel):
    session_id: str
    video_id: str
    policy: str
    model_checkpoint: str
    created_at: str
    summary: dict[str, float | int]


class SessionSegment(PlaybackSegment):
    pass
