"""Canonical telemetry record and validation."""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Optional


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _non_negative(name: str, value: Optional[float]) -> None:
    if value is not None and (not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class Measurement:
    experiment_id: str
    timestamp: datetime
    configured_bandwidth_mbps: Optional[float] = None
    throughput_mbps: Optional[float] = None
    rtt_ms: Optional[float] = None
    packet_loss_percent: Optional[float] = None
    download_duration_seconds: Optional[float] = None
    trace_id: Optional[str] = None
    configured_delay_ms: Optional[float] = None
    configured_packet_loss_percent: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must not be empty")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        for name in (
            "configured_bandwidth_mbps",
            "configured_delay_ms",
            "throughput_mbps",
            "rtt_ms",
            "download_duration_seconds",
        ):
            _non_negative(name, getattr(self, name))
        for name in ("configured_packet_loss_percent", "packet_loss_percent"):
            value = getattr(self, name)
            if value is not None and (
                not math.isfinite(value) or not 0 <= value <= 100
            ):
                raise ValueError(f"{name} must be between 0 and 100")
        if self.trace_id is not None and not self.trace_id.strip():
            raise ValueError("trace_id must not be empty")

    def as_csv_row(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "configured_bandwidth_mbps": self.configured_bandwidth_mbps,
            "configured_delay_ms": self.configured_delay_ms,
            "configured_packet_loss_percent": self.configured_packet_loss_percent,
            "throughput_mbps": self.throughput_mbps,
            "rtt_ms": self.rtt_ms,
            "observed_packet_loss_percent": self.packet_loss_percent,
            "download_duration_seconds": self.download_duration_seconds,
        }
