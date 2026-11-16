"""Persistent playback session records."""

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from threading import Lock

from backend.app.services.playback_metrics import SegmentPlayback, summarize_playback


@dataclass
class PlaybackSession:
    session_id: str
    video_id: str
    policy: str
    model_checkpoint: str
    created_at: str
    segments: list[SegmentPlayback] = field(default_factory=list)

    def add_segment(self, segment: SegmentPlayback) -> None:
        if self.segments and segment.segment_index <= self.segments[-1].segment_index:
            raise ValueError("segment_index must increase within a session")
        self.segments.append(segment)

    def as_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "video_id": self.video_id,
            "policy": self.policy,
            "model_checkpoint": self.model_checkpoint,
            "created_at": self.created_at,
            "segments": [asdict(segment) for segment in self.segments],
            "summary": summarize_playback(self.segments),
        }


class SessionService:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def create(self, video_id: str, policy: str, model_checkpoint: str) -> PlaybackSession:
        session = PlaybackSession(
            uuid4().hex,
            video_id,
            policy,
            model_checkpoint,
            datetime.now(timezone.utc).isoformat(),
        )
        self._write_json(session)
        return session

    def get(self, session_id: str) -> PlaybackSession:
        if Path(session_id).name != session_id:
            raise ValueError("invalid session id")
        path = self.root / f"{session_id}.json"
        if not path.is_file():
            raise FileNotFoundError(session_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        session = PlaybackSession(
            data["session_id"],
            data["video_id"],
            data["policy"],
            data["model_checkpoint"],
            data["created_at"],
        )
        session.segments = [SegmentPlayback(**segment) for segment in data["segments"]]
        return session

    def add_segment(self, session_id: str, segment: SegmentPlayback) -> PlaybackSession:
        with self._lock:
            session = self.get(session_id)
            session.add_segment(segment)
            self._write_json(session)
            return session

    def export_csv(self, session_id: str) -> Path:
        session = self.get(session_id)
        path = self.root / f"{session.session_id}.csv"
        fields = list(asdict(session.segments[0]).keys()) if session.segments else []
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if fields:
                writer.writeheader()
                writer.writerows(asdict(segment) for segment in session.segments)
        return path

    def _write_json(self, session: PlaybackSession) -> None:
        temporary = self.root / f".{session.session_id}.json.tmp"
        temporary.write_text(json.dumps(session.as_dict(), indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.root / f"{session.session_id}.json")
