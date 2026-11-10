"""Schemas and data models for StreamIQ simulation."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class NetworkTelemetry:
    timestamp_seconds: float
    bandwidth_mbps: float
    delay_ms: float
    packet_loss_percent: float


@dataclass
class PolicyStepTelemetry:
    policy_name: str
    action: int
    quality_name: str
    bitrate_mbps: float
    buffer_seconds: float
    download_duration_seconds: float
    rebuffer_seconds: float
    throughput_mbps: float
    step_reward: float
    accumulated_qoe: float


@dataclass
class SimulationStepEvent:
    step_index: int
    segment_index: int
    time_seconds: float
    network: NetworkTelemetry
    policy_states: Dict[str, PolicyStepTelemetry]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "segment_index": self.segment_index,
            "time_seconds": self.time_seconds,
            "network": asdict(self.network),
            "policy_states": {
                name: asdict(state) for name, state in self.policy_states.items()
            },
        }


@dataclass
class PolicySummary:
    policy_name: str
    average_bitrate_mbps: float
    rebuffer_seconds: float
    rebuffer_ratio: float
    bitrate_switches: int
    average_bitrate_change_mbps: float
    total_bitrate_variation_mbps: float
    upward_switches: int
    downward_switches: int
    switching_magnitude: float
    average_download_duration_seconds: float
    qoe: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchSimulationResult:
    trace_id: str
    profile_name: str
    duration_seconds: float
    segment_count: int
    timeline: List[SimulationStepEvent] = field(default_factory=list)
    summaries: Dict[str, PolicySummary] = field(default_factory=dict)
    winner_policy: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "profile_name": self.profile_name,
            "duration_seconds": self.duration_seconds,
            "segment_count": self.segment_count,
            "timeline": [event.to_dict() for event in self.timeline],
            "summaries": {k: v.to_dict() for k, v in self.summaries.items()},
            "winner_policy": self.winner_policy,
        }

