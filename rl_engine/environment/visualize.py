"""Episode metric visualization."""

from pathlib import Path

from .metrics import EpisodeMetrics


def plot_episode(metrics: EpisodeMetrics, output: str | Path) -> Path:
    import matplotlib.pyplot as plt

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    indices = [metric.segment_index for metric in metrics.segments]
    figure, axes = plt.subplots(4, 1, sharex=True, figsize=(9, 8))
    axes[0].plot(indices, [metric.bitrate_mbps for metric in metrics.segments])
    axes[0].set_ylabel("Bitrate (Mbps)")
    axes[1].plot(indices, [metric.bandwidth_mbps for metric in metrics.segments])
    axes[1].set_ylabel("Bandwidth (Mbps)")
    axes[2].plot(indices, [metric.buffer_occupancy_seconds for metric in metrics.segments])
    axes[2].set_ylabel("Buffer (s)")
    axes[3].bar(indices, [metric.rebuffer_seconds for metric in metrics.segments])
    axes[3].set_ylabel("Rebuffer (s)")
    axes[3].set_xlabel("Segment")
    figure.tight_layout()
    figure.savefig(destination)
    plt.close(figure)
    return destination
