"""Playback buffer accounting."""

from dataclasses import dataclass


@dataclass
class BufferState:
    occupancy_seconds: float = 0.0
    capacity_seconds: float = 60.0
    rebuffer_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.capacity_seconds <= 0:
            raise ValueError("buffer capacity must be positive")

    def download_and_play(self, download_seconds: float, segment_seconds: float) -> float:
        if download_seconds < 0 or segment_seconds <= 0:
            raise ValueError("download and segment durations are invalid")
        stall = max(0.0, download_seconds - self.occupancy_seconds)
        self.rebuffer_seconds += stall
        self.occupancy_seconds = max(0.0, self.occupancy_seconds - download_seconds)
        self.occupancy_seconds = min(self.capacity_seconds, self.occupancy_seconds + segment_seconds)
        return stall
