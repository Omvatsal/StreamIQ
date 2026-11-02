"""Comparable policy metrics."""

from dataclasses import dataclass

from rl_engine.environment.metrics import EpisodeMetrics


@dataclass(frozen=True)
class EvaluationMetrics:
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


def summarize(metrics: EpisodeMetrics) -> EvaluationMetrics:
    if not metrics.segments:
        return EvaluationMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    average = sum(item.bitrate_mbps for item in metrics.segments) / len(metrics.segments)
    changes = [
        current.bitrate_mbps - previous.bitrate_mbps
        for previous, current in zip(metrics.segments, metrics.segments[1:])
    ]
    absolute_changes = [abs(change) for change in changes]
    total_variation = sum(absolute_changes)
    qoe = average - 4.0 * metrics.total_rebuffer_seconds - 0.5 * total_variation
    return EvaluationMetrics(
        average,
        metrics.total_rebuffer_seconds,
        metrics.total_rebuffer_seconds / (len(metrics.segments) * 4.0),
        metrics.switching_frequency,
        sum(absolute_changes) / len(absolute_changes) if absolute_changes else 0,
        total_variation,
        sum(change > 0 for change in changes),
        sum(change < 0 for change in changes),
        sum(absolute_changes),
        sum(item.download_duration_seconds for item in metrics.segments) / len(metrics.segments),
        qoe,
    )
