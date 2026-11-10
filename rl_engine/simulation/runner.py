"""Multi-policy ABR simulation runner for StreamIQ.

Supports both fast synchronous batch simulation and asynchronous
real-time step streaming over WebSockets or event loops.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import numpy as np

from network_lab.traces.models import NetworkTrace
from rl_engine.agents.dqn import DQNAgent
from rl_engine.environment.abr_env import ABREnvironment
from rl_engine.environment.video import VideoModel
from rl_engine.evaluation.baselines import buffer_based, fixed_bitrate, throughput_based
from rl_engine.evaluation.metrics import summarize
from .schemas import (
    BatchSimulationResult,
    NetworkTelemetry,
    PolicyStepTelemetry,
    PolicySummary,
    SimulationStepEvent,
)

DEFAULT_CHECKPOINT_PATH = (
    Path(__file__).resolve().parents[2] / "experiments" / "results" / "dqn" / "model_final.pt"
)


class SimulationRunner:
    """Orchestrates multi-policy execution on identical network traces."""

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        agent: Optional[DQNAgent] = None,
        segment_duration_seconds: float = 4.0,
    ) -> None:
        self.segment_duration_seconds = segment_duration_seconds
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
        self._agent = agent

    def get_agent(self, state_size: int = 9) -> DQNAgent:
        """Lazily load and return the DQN agent."""
        if self._agent is None:
            self._agent = DQNAgent(state_size=state_size, seed=42)
            if self.checkpoint_path.exists():
                self._agent.load(self.checkpoint_path)
        return self._agent

    def resolve_policies(
        self,
        policy_names: Optional[List[str]] = None,
        state_size: int = 9,
    ) -> Dict[str, Callable[[np.ndarray, ABREnvironment], int]]:
        """Map user-requested policy names to executable callable functions."""
        agent = self.get_agent(state_size=state_size)
        available: Dict[str, Callable[[np.ndarray, ABREnvironment], int]] = {
            "dqn": lambda state, _env: agent.select_action(state, explore=False),
            "buffer_based": buffer_based,
            "bba": buffer_based,
            "throughput_based": throughput_based,
            "rate_based": throughput_based,
            "fixed_240p": fixed_bitrate(0),
            "fixed_720p": fixed_bitrate(3),
            "fixed_1080p": fixed_bitrate(4),
        }

        if not policy_names:
            return {
                "dqn": available["dqn"],
                "buffer_based": available["buffer_based"],
                "throughput_based": available["throughput_based"],
                "fixed_240p": available["fixed_240p"],
            }

        resolved: Dict[str, Callable[[np.ndarray, ABREnvironment], int]] = {}
        for name in policy_names:
            normalized = name.lower().strip()
            if normalized in available:
                resolved[normalized] = available[normalized]
            else:
                raise ValueError(
                    f"Unknown policy '{name}'. Supported policies: {list(available.keys())}"
                )
        return resolved

    def _setup_environments(
        self,
        trace: NetworkTrace,
        policy_names: List[str],
        total_segments: Optional[int] = None,
    ) -> Tuple[Dict[str, ABREnvironment], Dict[str, np.ndarray]]:
        """Create synchronized environments initialized with the given trace."""
        if total_segments is None:
            total_segments = max(1, int(trace.duration_seconds // self.segment_duration_seconds))

        video = VideoModel(
            segment_duration_seconds=self.segment_duration_seconds,
            total_segments=total_segments,
        )

        envs: Dict[str, ABREnvironment] = {}
        states: Dict[str, np.ndarray] = {}

        for name in policy_names:
            env = ABREnvironment(trace=trace, video=video)
            state, _ = env.reset(seed=trace.seed)
            envs[name] = env
            states[name] = state

        return envs, states

    def run_batch(
        self,
        trace: NetworkTrace,
        policy_names: Optional[List[str]] = None,
        total_segments: Optional[int] = None,
    ) -> BatchSimulationResult:
        """Run full multi-policy simulation synchronously and return comparative results."""
        policies = self.resolve_policies(policy_names)
        envs, states = self._setup_environments(
            trace, list(policies.keys()), total_segments=total_segments
        )

        first_env = next(iter(envs.values()))
        segment_count = first_env.video.segment_count
        timeline: List[SimulationStepEvent] = []

        for step_idx in range(segment_count):
            step_policy_states: Dict[str, PolicyStepTelemetry] = {}
            current_time = 0.0

            for name, policy_fn in policies.items():
                env = envs[name]
                state = states[name]

                action = policy_fn(state, env)
                next_state, reward, done, _, info = env.step(action)
                states[name] = next_state

                last_metric = env.metrics.segments[-1]
                cum_eval = summarize(env.metrics)
                current_time = max(current_time, info.get("time_seconds", 0.0))

                step_policy_states[name] = PolicyStepTelemetry(
                    policy_name=name,
                    action=int(action),
                    quality_name=last_metric.quality_name,
                    bitrate_mbps=float(last_metric.bitrate_mbps),
                    buffer_seconds=float(last_metric.buffer_occupancy_seconds),
                    download_duration_seconds=float(last_metric.download_duration_seconds),
                    rebuffer_seconds=float(last_metric.rebuffer_seconds),
                    throughput_mbps=float(last_metric.throughput_mbps),
                    step_reward=float(reward),
                    accumulated_qoe=float(cum_eval.qoe),
                )

            # Read network conditions at current simulation timeline
            net_condition = first_env.network.condition_at(current_time)
            network_telemetry = NetworkTelemetry(
                timestamp_seconds=round(current_time, 3),
                bandwidth_mbps=float(net_condition.bandwidth_mbps),
                delay_ms=float(net_condition.delay_ms),
                packet_loss_percent=float(net_condition.packet_loss_percent),
            )

            event = SimulationStepEvent(
                step_index=step_idx,
                segment_index=step_idx,
                time_seconds=round(current_time, 3),
                network=network_telemetry,
                policy_states=step_policy_states,
            )
            timeline.append(event)

        # Summarize performance
        summaries: Dict[str, PolicySummary] = {}
        best_policy = ""
        best_qoe = -float("inf")

        for name, env in envs.items():
            metrics = summarize(env.metrics)
            summary = PolicySummary(
                policy_name=name,
                average_bitrate_mbps=round(metrics.average_bitrate_mbps, 4),
                rebuffer_seconds=round(metrics.rebuffer_seconds, 4),
                rebuffer_ratio=round(metrics.rebuffer_ratio, 4),
                bitrate_switches=metrics.bitrate_switches,
                average_bitrate_change_mbps=round(metrics.average_bitrate_change_mbps, 4),
                total_bitrate_variation_mbps=round(metrics.total_bitrate_variation_mbps, 4),
                upward_switches=metrics.upward_switches,
                downward_switches=metrics.downward_switches,
                switching_magnitude=round(metrics.switching_magnitude, 4),
                average_download_duration_seconds=round(
                    metrics.average_download_duration_seconds, 4
                ),
                qoe=round(metrics.qoe, 4),
            )
            summaries[name] = summary
            if metrics.qoe > best_qoe:
                best_qoe = metrics.qoe
                best_policy = name

        return BatchSimulationResult(
            trace_id=trace.trace_id,
            profile_name=trace.profile,
            duration_seconds=trace.duration_seconds,
            segment_count=segment_count,
            timeline=timeline,
            summaries=summaries,
            winner_policy=best_policy,
        )

    async def stream_simulation(
        self,
        trace: NetworkTrace,
        policy_names: Optional[List[str]] = None,
        step_delay: float = 0.1,
        total_segments: Optional[int] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Asynchronously stream simulation steps tick-by-tick for real-time visualization."""
        policies = self.resolve_policies(policy_names)
        envs, states = self._setup_environments(
            trace, list(policies.keys()), total_segments=total_segments
        )

        first_env = next(iter(envs.values()))
        segment_count = first_env.video.segment_count

        yield {
            "type": "init",
            "trace_id": trace.trace_id,
            "profile_name": trace.profile,
            "duration_seconds": trace.duration_seconds,
            "segment_count": segment_count,
            "policies": list(policies.keys()),
        }

        for step_idx in range(segment_count):
            step_policy_states: Dict[str, PolicyStepTelemetry] = {}
            current_time = 0.0

            for name, policy_fn in policies.items():
                env = envs[name]
                state = states[name]

                action = policy_fn(state, env)
                next_state, reward, done, _, info = env.step(action)
                states[name] = next_state

                last_metric = env.metrics.segments[-1]
                cum_eval = summarize(env.metrics)
                current_time = max(current_time, info.get("time_seconds", 0.0))

                step_policy_states[name] = PolicyStepTelemetry(
                    policy_name=name,
                    action=int(action),
                    quality_name=last_metric.quality_name,
                    bitrate_mbps=float(last_metric.bitrate_mbps),
                    buffer_seconds=float(last_metric.buffer_occupancy_seconds),
                    download_duration_seconds=float(last_metric.download_duration_seconds),
                    rebuffer_seconds=float(last_metric.rebuffer_seconds),
                    throughput_mbps=float(last_metric.throughput_mbps),
                    step_reward=float(reward),
                    accumulated_qoe=float(cum_eval.qoe),
                )

            net_condition = first_env.network.condition_at(current_time)
            network_telemetry = NetworkTelemetry(
                timestamp_seconds=round(current_time, 3),
                bandwidth_mbps=float(net_condition.bandwidth_mbps),
                delay_ms=float(net_condition.delay_ms),
                packet_loss_percent=float(net_condition.packet_loss_percent),
            )

            event = SimulationStepEvent(
                step_index=step_idx,
                segment_index=step_idx,
                time_seconds=round(current_time, 3),
                network=network_telemetry,
                policy_states=step_policy_states,
            )

            yield {
                "type": "step",
                "event": event.to_dict(),
            }

            if step_delay > 0:
                await asyncio.sleep(step_delay)

        # Emit completion summary
        summaries: Dict[str, Any] = {}
        best_policy = ""
        best_qoe = -float("inf")

        for name, env in envs.items():
            metrics = summarize(env.metrics)
            summary_dict = {
                "policy_name": name,
                "average_bitrate_mbps": round(metrics.average_bitrate_mbps, 4),
                "rebuffer_seconds": round(metrics.rebuffer_seconds, 4),
                "rebuffer_ratio": round(metrics.rebuffer_ratio, 4),
                "bitrate_switches": metrics.bitrate_switches,
                "average_bitrate_change_mbps": round(metrics.average_bitrate_change_mbps, 4),
                "total_bitrate_variation_mbps": round(metrics.total_bitrate_variation_mbps, 4),
                "upward_switches": metrics.upward_switches,
                "downward_switches": metrics.downward_switches,
                "switching_magnitude": round(metrics.switching_magnitude, 4),
                "average_download_duration_seconds": round(
                    metrics.average_download_duration_seconds, 4
                ),
                "qoe": round(metrics.qoe, 4),
            }
            summaries[name] = summary_dict
            if metrics.qoe > best_qoe:
                best_qoe = metrics.qoe
                best_policy = name

        yield {
            "type": "complete",
            "summaries": summaries,
            "winner_policy": best_policy,
        }

