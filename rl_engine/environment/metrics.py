"""Episode metrics and CSV export."""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SegmentMetric:
    segment_index: int
    quality_index: int
    quality_name: str
    bitrate_mbps: float
    throughput_mbps: float
    download_duration_seconds: float
    buffer_occupancy_seconds: float
    rebuffer_seconds: float
    reward: float
    bandwidth_mbps: float


class EpisodeMetrics:
    def __init__(self) -> None:
        self.segments: list[SegmentMetric] = []

    def add(self, metric: SegmentMetric) -> None:
        self.segments.append(metric)

    @property
    def accumulated_reward(self) -> float:
        return sum(metric.reward for metric in self.segments)

    @property
    def total_rebuffer_seconds(self) -> float:
        return sum(metric.rebuffer_seconds for metric in self.segments)

    @property
    def switching_frequency(self) -> int:
        return sum(
            current.quality_index != previous.quality_index
            for previous, current in zip(self.segments, self.segments[1:])
        )

    def to_csv(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(asdict(self.segments[0]).keys()) if self.segments else [])
            if self.segments:
                writer.writeheader()
                writer.writerows(asdict(metric) for metric in self.segments)
