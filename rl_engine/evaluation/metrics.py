"""Comparable policy metrics."""

from dataclasses import dataclass

from rl_engine.environment.metrics import EpisodeMetrics


@dataclass(frozen=True)
class EvaluationMetrics:
    average_bitrate_mbps: float
    rebuffer_seconds: float
    bitrate_switches: int
    switching_magnitude: int
    qoe: float


def summarize(metrics: EpisodeMetrics) -> EvaluationMetrics:
    if not metrics.segments:
        return EvaluationMetrics(0, 0, 0, 0, 0)
    average = sum(item.bitrate_mbps for item in metrics.segments) / len(metrics.segments)
    magnitude = sum(abs(current.quality_index - previous.quality_index)
                    for previous, current in zip(metrics.segments, metrics.segments[1:]))
    return EvaluationMetrics(average, metrics.total_rebuffer_seconds, metrics.switching_frequency,
                              magnitude, metrics.accumulated_reward)
