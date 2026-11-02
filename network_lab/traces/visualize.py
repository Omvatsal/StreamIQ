"""Optional matplotlib visualization for generated traces."""

from pathlib import Path

from .models import NetworkTrace


def plot_trace(trace: NetworkTrace, output: str | Path) -> Path:
    import matplotlib.pyplot as plt

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    times = [c.timestamp_seconds for c in trace.conditions]
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    axes[0].plot(times, [c.bandwidth_mbps for c in trace.conditions])
    axes[0].set_ylabel("Mbps")
    axes[1].plot(times, [c.delay_ms for c in trace.conditions])
    axes[1].set_ylabel("Delay (ms)")
    axes[2].plot(times, [c.packet_loss_percent for c in trace.conditions])
    axes[2].set_ylabel("Loss (%)")
    axes[2].set_xlabel("Time (s)")
    fig.suptitle(f"{trace.trace_id} ({trace.profile})")
    fig.tight_layout()
    fig.savefig(destination)
    plt.close(fig)
    return destination
