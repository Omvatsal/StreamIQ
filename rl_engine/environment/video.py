"""Video segment and bitrate model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VideoModel:
    segment_duration_seconds: float = 4.0
    total_segments: int = 10
    bitrates_mbps: tuple[float, ...] = (0.3, 0.7, 1.2, 2.5, 5.0)
    quality_names: tuple[str, ...] = ("240p", "360p", "480p", "720p", "1080p")

    def __post_init__(self) -> None:
        if self.segment_duration_seconds <= 0:
            raise ValueError("segment_duration_seconds must be positive")
        if self.total_segments <= 0:
            raise ValueError("total_segments must be positive")
        if len(self.bitrates_mbps) != 5 or len(self.quality_names) != 5:
            raise ValueError("the video model must have exactly five quality levels")
        if any(bitrate <= 0 for bitrate in self.bitrates_mbps):
            raise ValueError("bitrates must be positive")

    @property
    def segment_count(self) -> int:
        return self.total_segments

    def segment_size_megabytes(self, quality_index: int) -> float:
        self._validate_quality(quality_index)
        return self.bitrates_mbps[quality_index] * self.segment_duration_seconds / 8

    def _validate_quality(self, quality_index: int) -> None:
        if not 0 <= quality_index < len(self.bitrates_mbps):
            raise ValueError(f"quality index must be between 0 and {len(self.bitrates_mbps) - 1}")
