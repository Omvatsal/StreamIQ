"""Validated trace representation shared by generators and environments."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Optional


def _finite_non_negative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class NetworkCondition:
    timestamp_seconds: float
    bandwidth_mbps: float
    delay_ms: float = 5.0
    packet_loss_percent: float = 0.0

    def __post_init__(self) -> None:
        _finite_non_negative("timestamp_seconds", self.timestamp_seconds)
        _finite_non_negative("bandwidth_mbps", self.bandwidth_mbps)
        _finite_non_negative("delay_ms", self.delay_ms)
        if not math.isfinite(self.packet_loss_percent) or not 0 <= self.packet_loss_percent <= 100:
            raise ValueError("packet_loss_percent must be between 0 and 100")


@dataclass(frozen=True)
class TraceConfig:
    duration_seconds: float
    interval_seconds: float = 1.0
    bandwidth_mbps: float = 10.0
    min_bandwidth_mbps: float = 1.0
    max_bandwidth_mbps: float = 10.0
    delay_ms: float = 5.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        _finite_non_negative("duration_seconds", self.duration_seconds)
        if self.duration_seconds == 0:
            raise ValueError("duration_seconds must be positive")
        if not math.isfinite(self.interval_seconds) or self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        for name in ("bandwidth_mbps", "min_bandwidth_mbps", "max_bandwidth_mbps", "delay_ms"):
            _finite_non_negative(name, getattr(self, name))
        if self.min_bandwidth_mbps > self.max_bandwidth_mbps:
            raise ValueError("min_bandwidth_mbps cannot exceed max_bandwidth_mbps")


@dataclass(frozen=True)
class NetworkTrace:
    trace_id: str
    profile: str
    seed: Optional[int]
    duration_seconds: float
    interval_seconds: float
    conditions: tuple[NetworkCondition, ...]

    def __post_init__(self) -> None:
        if not self.trace_id.strip() or not self.profile.strip():
            raise ValueError("trace_id and profile must not be empty")
        if not self.conditions:
            raise ValueError("trace must contain conditions")
        if self.conditions[0].timestamp_seconds != 0:
            raise ValueError("trace must start at timestamp zero")
        timestamps = [condition.timestamp_seconds for condition in self.conditions]
        if timestamps != sorted(timestamps) or len(timestamps) != len(set(timestamps)):
            raise ValueError("trace timestamps must be strictly ordered")
        if timestamps[-1] >= self.duration_seconds:
            raise ValueError("conditions must fall within the trace duration")

    def to_dict(self) -> dict[str, object]:
        return {
            "trace_id": self.trace_id,
            "profile": self.profile,
            "seed": self.seed,
            "duration_seconds": self.duration_seconds,
            "interval_seconds": self.interval_seconds,
            "conditions": [asdict(condition) for condition in self.conditions],
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NetworkTrace":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            trace_id=data["trace_id"],
            profile=data["profile"],
            seed=data.get("seed"),
            duration_seconds=float(data["duration_seconds"]),
            interval_seconds=float(data["interval_seconds"]),
            conditions=tuple(NetworkCondition(**condition) for condition in data["conditions"]),
        )
