"""Tests for the multi-policy simulation runner."""

import asyncio
from network_lab.traces.generator import generate_trace
from network_lab.traces.models import TraceConfig
from rl_engine.simulation.runner import SimulationRunner


def test_batch_simulation_deterministic():
    config = TraceConfig(duration_seconds=20, interval_seconds=1.0, seed=42)
    trace = generate_trace("sudden_drop", "test_trace_1", config)
    runner = SimulationRunner()
    
    result = runner.run_batch(trace, policy_names=["dqn", "buffer_based", "throughput_based", "fixed_240p"])
    
    assert result.trace_id == trace.trace_id
    assert result.profile_name == "sudden_drop"
    assert len(result.timeline) == result.segment_count
    assert "dqn" in result.summaries
    assert "buffer_based" in result.summaries
    assert "throughput_based" in result.summaries
    assert "fixed_240p" in result.summaries
    assert result.winner_policy in result.summaries


def test_stream_simulation():
    config = TraceConfig(duration_seconds=12, interval_seconds=1.0, seed=123)
    trace = generate_trace("oscillating", "test_trace_2", config)
    runner = SimulationRunner()
    
    async def _run():
        events = []
        async for event in runner.stream_simulation(trace, policy_names=["dqn", "bba"], step_delay=0.0):
            events.append(event)
        return events

    events = asyncio.run(_run())
        
    assert len(events) >= 3  # init, steps..., complete
    assert events[0]["type"] == "init"
    assert events[-1]["type"] == "complete"
    assert "winner_policy" in events[-1]
