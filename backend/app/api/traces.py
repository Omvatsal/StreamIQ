"""Network traces and profile configuration endpoints."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.app.services.sim_service import sim_service
from network_lab.traces.generator import PROFILE_NAMES

router = APIRouter(prefix="/traces", tags=["Network Traces"])

PROFILE_METADATA = {
    "stable": {
        "name": "Stable",
        "description": "Consistent high bandwidth with minor Gaussian jitter (ideal network conditions).",
        "default_bandwidth_mbps": 10.0,
        "difficulty": "Easy",
    },
    "random": {
        "name": "Random Walk",
        "description": "Stochastic bandwidth shifts with random hold durations (cellular / mobile data).",
        "default_bandwidth_mbps": 6.0,
        "difficulty": "Medium",
    },
    "oscillating": {
        "name": "Oscillating (Sinusoidal)",
        "description": "Periodic wave-like bandwidth changes testing algorithm temporal stability against rapid switching.",
        "default_bandwidth_mbps": 5.0,
        "difficulty": "Hard",
    },
    "sudden_drop": {
        "name": "Sudden Drop",
        "description": "Bandwidth plummets to 20% mid-stream, testing rebuffer prevention and rapid down-adaptation.",
        "default_bandwidth_mbps": 10.0,
        "difficulty": "Hard",
    },
    "recovery": {
        "name": "Gradual Recovery",
        "description": "Severe degradation followed by slow, linear capacity recovery, testing cautious up-adaptation.",
        "default_bandwidth_mbps": 8.0,
        "difficulty": "Medium",
    },
    "congestion": {
        "name": "Congestion (High Delay & Loss)",
        "description": "Concurrent degradation in throughput, inflated RTT (+35ms), and 5-10% packet loss.",
        "difficulty": "Extreme",
    },
}


class GenerateTraceRequest(BaseModel):
    profile: str = Field(default="sudden_drop", description="Profile name")
    duration_seconds: float = Field(default=60.0, ge=5.0, le=600.0)
    interval_seconds: float = Field(default=1.0, ge=0.1, le=10.0)
    bandwidth_mbps: float = Field(default=10.0, ge=0.5, le=100.0)
    min_bandwidth_mbps: float = Field(default=1.0, ge=0.1, le=50.0)
    max_bandwidth_mbps: float = Field(default=10.0, ge=1.0, le=100.0)
    delay_ms: float = Field(default=5.0, ge=0.0, le=1000.0)
    seed: Optional[int] = Field(default=42)


@router.get("/profiles")
async def get_profiles() -> Dict[str, Any]:
    """List available network trace profiles and their descriptions."""
    return {
        "profiles": [
            {
                "id": profile,
                **PROFILE_METADATA.get(profile, {"name": profile, "description": ""}),
            }
            for profile in PROFILE_NAMES
        ]
    }


@router.post("/generate")
async def generate_trace_endpoint(request: GenerateTraceRequest) -> Dict[str, Any]:
    """Generate and preview network condition timeline for the requested profile."""
    trace = sim_service.create_trace(
        profile=request.profile,
        duration_seconds=request.duration_seconds,
        interval_seconds=request.interval_seconds,
        bandwidth_mbps=request.bandwidth_mbps,
        min_bandwidth_mbps=request.min_bandwidth_mbps,
        max_bandwidth_mbps=request.max_bandwidth_mbps,
        delay_ms=request.delay_ms,
        seed=request.seed,
    )
    return {
        "trace_id": trace.trace_id,
        "profile": trace.profile,
        "seed": trace.seed,
        "duration_seconds": trace.duration_seconds,
        "interval_seconds": trace.interval_seconds,
        "condition_count": len(trace.conditions),
        "conditions": [
            {
                "timestamp_seconds": c.timestamp_seconds,
                "bandwidth_mbps": c.bandwidth_mbps,
                "delay_ms": c.delay_ms,
                "packet_loss_percent": c.packet_loss_percent,
            }
            for c in trace.conditions
        ],
    }

