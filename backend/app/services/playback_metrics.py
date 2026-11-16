"""Segment-level playback metrics used by the browser client."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentPlayback:
    segment_index: int
    bitrate_mbps: float
    throughput_mbps: float
    download_duration_seconds: float
    buffer_seconds: float
    rebuffer_seconds: float = 0.0


def summarize_playback(segments: list[SegmentPlayback]) -> dict[str, float | int]:
    if not segments:
        return {
            "average_bitrate_mbps": 0.0,
            "rebuffer_seconds": 0.0,
            "bitrate_switches": 0,
            "switching_magnitude_mbps": 0.0,
        }
    changes = [
        current.bitrate_mbps - previous.bitrate_mbps
        for previous, current in zip(segments, segments[1:])
    ]
    return {
        "average_bitrate_mbps": sum(item.bitrate_mbps for item in segments) / len(segments),
        "rebuffer_seconds": sum(item.rebuffer_seconds for item in segments),
        "bitrate_switches": sum(change != 0 for change in changes),
        "switching_magnitude_mbps": sum(abs(change) for change in changes),
    }
