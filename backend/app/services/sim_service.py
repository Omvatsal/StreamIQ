"""Simulation service bridging API requests to the core simulation engine."""

import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from network_lab.traces.generator import generate_trace
from network_lab.traces.models import NetworkTrace, TraceConfig
from rl_engine.simulation.runner import SimulationRunner
from backend.app.core.config import settings


class SimulationService:
    """Manages trace generation and multi-policy simulations."""

    def __init__(self) -> None:
        self.runner = SimulationRunner(checkpoint_path=settings.model_checkpoint)

    def create_trace(
        self,
        profile: str,
        duration_seconds: float = 60.0,
        interval_seconds: float = 1.0,
        bandwidth_mbps: float = 10.0,
        min_bandwidth_mbps: float = 1.0,
        max_bandwidth_mbps: float = 10.0,
        delay_ms: float = 5.0,
        seed: Optional[int] = 42,
        trace_id: Optional[str] = None,
    ) -> NetworkTrace:
        trace_id = trace_id or f"{profile}_{seed or uuid.uuid4().hex[:6]}"
        config = TraceConfig(
            duration_seconds=duration_seconds,
            interval_seconds=interval_seconds,
            bandwidth_mbps=bandwidth_mbps,
            min_bandwidth_mbps=min_bandwidth_mbps,
            max_bandwidth_mbps=max_bandwidth_mbps,
            delay_ms=delay_ms,
            seed=seed,
        )
        return generate_trace(profile, trace_id, config)

    def run_batch(
        self,
        profile: str,
        duration_seconds: float = 60.0,
        interval_seconds: float = 1.0,
        seed: Optional[int] = 42,
        policies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        trace = self.create_trace(
            profile=profile,
            duration_seconds=duration_seconds,
            interval_seconds=interval_seconds,
            seed=seed,
        )
        result = self.runner.run_batch(trace, policy_names=policies)
        return result.to_dict()

    async def stream_live(
        self,
        profile: str,
        duration_seconds: float = 60.0,
        interval_seconds: float = 1.0,
        seed: Optional[int] = 42,
        policies: Optional[List[str]] = None,
        step_delay: float = 0.2,
    ) -> AsyncIterator[Dict[str, Any]]:
        trace = self.create_trace(
            profile=profile,
            duration_seconds=duration_seconds,
            interval_seconds=interval_seconds,
            seed=seed,
        )
        async for step in self.runner.stream_simulation(
            trace,
            policy_names=policies,
            step_delay=step_delay,
        ):
            yield step


sim_service = SimulationService()

