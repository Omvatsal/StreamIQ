"""Playback session routes."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from backend.app.config import get_settings
from backend.app.schemas.sessions import SessionCreate, SessionResponse, SessionSegment
from backend.app.services.playback_metrics import SegmentPlayback
from backend.app.services.session_service import SessionService

router = APIRouter(prefix="/api/sessions")
settings = get_settings()
sessions = SessionService(settings.experiment_root)


def _response(session) -> SessionResponse:
    return SessionResponse(
        session_id=session.session_id,
        video_id=session.video_id,
        policy=session.policy,
        model_checkpoint=session.model_checkpoint,
        created_at=session.created_at,
        summary=session.as_dict()["summary"],
    )


@router.post("", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    return _response(sessions.create(request.video_id, request.policy, str(settings.dqn_checkpoint)))


@router.post("/{session_id}/segments", response_model=SessionResponse)
async def add_segment(session_id: str, segment: SessionSegment):
    try:
        return _response(sessions.add_segment(session_id, SegmentPlayback(**segment.model_dump())))
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    try:
        return _response(sessions.get(session_id))
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.get("/{session_id}/export.csv")
async def export_session(session_id: str):
    try:
        return FileResponse(sessions.export_csv(session_id), media_type="text/csv")
    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
