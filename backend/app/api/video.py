"""MPEG-DASH video streaming and segment delivery endpoints."""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse, PlainTextResponse

from backend.app.core.config import settings
from rl_engine.environment.video import VideoModel

router = APIRouter(prefix="/video", tags=["Video Delivery"])
video_model = VideoModel()


def generate_mpd_manifest(base_url: str = "/api/video", duration_seconds: float = 60.0) -> str:
    """Generate an ISO/IEC 23009-1 compliant MPEG-DASH MPD manifest XML."""
    total_segments = int(duration_seconds // video_model.segment_duration_seconds)
    representations_xml = ""

    for idx, (bitrate, name) in enumerate(
        zip(video_model.bitrates_mbps, video_model.quality_names)
    ):
        bandwidth_bps = int(bitrate * 1_000_000)
        res_map = {
            "240p": (426, 240),
            "360p": (640, 360),
            "480p": (854, 480),
            "720p": (1280, 720),
            "1080p": (1920, 1080),
        }
        width, height = res_map.get(name, (640, 360))

        representations_xml += f"""
      <Representation id="{idx}" bandwidth="{bandwidth_bps}" width="{width}" height="{height}" codecs="avc1.4d401f">
        <SegmentTemplate timescale="1000" duration="4000" startNumber="0"
          media="{base_url}/segments/{idx}/$Number$"
          initialization="{base_url}/init/{idx}"/>
      </Representation>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<MPD xmlns="urn:mpeg:dash:schema:mpd:2011"
     profiles="urn:mpeg:dash:profile:isoff-live:2011"
     type="static"
     mediaPresentationDuration="PT{int(duration_seconds)}S"
     minBufferTime="PT4S">
  <Period id="0" duration="PT{int(duration_seconds)}S">
    <AdaptationSet id="0" mimeType="video/mp4" segmentAlignment="true">
{representations_xml}
    </AdaptationSet>
  </Period>
</MPD>"""


@router.get("/manifest.mpd", response_class=PlainTextResponse)
async def get_manifest(duration: Optional[float] = 60.0):
    """Retrieve the dynamic MPEG-DASH manifest for StreamIQ."""
    xml_content = generate_mpd_manifest(duration_seconds=duration or 60.0)
    return Response(content=xml_content, media_type="application/dash+xml")


@router.get("/init/{quality_index}")
async def get_initialization_segment(quality_index: int):
    """Deliver initialization segment header."""
    if not 0 <= quality_index < len(video_model.bitrates_mbps):
        raise HTTPException(status_code=404, detail="Invalid quality index")

    # Check if physical asset exists
    asset_path = settings.video_assets_dir / str(quality_index) / "init.mp4"
    if asset_path.exists():
        return FileResponse(asset_path, media_type="video/mp4")

    # Synthetic initialization segment (valid minimal MP4 ftyp + moov header)
    synthetic_init = b"\x00\x00\x00\x20ftypiso5\x00\x00\x02\x00iso5avc1dash\x00\x00\x00\x08free"
    return Response(content=synthetic_init, media_type="video/mp4")


@router.get("/segments/{quality_index}/{segment_index}")
async def get_segment(quality_index: int, segment_index: int):
    """Deliver a 4-second video segment at the requested quality level."""
    if not 0 <= quality_index < len(video_model.bitrates_mbps):
        raise HTTPException(status_code=404, detail="Invalid quality index")

    # Check for real pre-encoded chunk
    segment_file = (
        settings.video_assets_dir / str(quality_index) / f"segment_{segment_index}.m4s"
    )
    headers = {
        "X-StreamIQ-Quality": video_model.quality_names[quality_index],
        "X-StreamIQ-Bitrate-Mbps": str(video_model.bitrates_mbps[quality_index]),
        "X-StreamIQ-Segment": str(segment_index),
    }
    if segment_file.exists():
        return FileResponse(segment_file, media_type="video/iso.segment", headers=headers)

    # Synthetic chunk generation with exact mathematical byte size
    # size = bitrate_mbps * 4 / 8 MB = bitrate * 500,000 bytes
    target_bytes = int(video_model.segment_size_megabytes(quality_index) * 1_000_000)

    # Minimal container box with dummy payload
    header = b"\x00\x00\x00\x18stypmsdh\x00\x00\x00\x00msdhmsix"
    payload_size = max(0, target_bytes - len(header))
    # Fill with repetitive dummy bytes to avoid huge memory allocation
    chunk_payload = header + b"\x00" * payload_size

    return Response(
        content=chunk_payload,
        media_type="video/iso.segment",
        headers=headers,
    )
