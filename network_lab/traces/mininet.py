"""Mininet traffic-control adapter for trace replay."""

from .models import NetworkCondition


class MininetTrafficShaper:
    """Apply bandwidth, delay, and loss to the server egress interface."""

    def __init__(self, interface: object, restore: NetworkCondition | None = None):
        self.interface = interface
        self.restore = restore

    def apply(self, condition: NetworkCondition) -> None:
        self.interface.config(
            bw=condition.bandwidth_mbps,
            delay=f"{condition.delay_ms}ms",
            loss=condition.packet_loss_percent,
        )

    def close(self) -> None:
        if self.restore is not None:
            self.apply(self.restore)
