"""Trace-backed network adapter for segment download simulation."""

from dataclasses import dataclass
from bisect import bisect_right

from network_lab.traces.models import NetworkCondition, NetworkTrace


@dataclass(frozen=True)
class DownloadResult:
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    throughput_mbps: float
    condition: NetworkCondition


class TraceNetworkAdapter:
    """Consume link capacity over time without depending on Mininet."""

    def __init__(self, trace: NetworkTrace, packet_loss_retries: bool = False):
        self.trace = trace
        self._timestamps = [condition.timestamp_seconds for condition in trace.conditions]
        self.packet_loss_retries = packet_loss_retries

    def condition_at(self, time_seconds: float) -> NetworkCondition:
        index = max(0, bisect_right(self._timestamps, time_seconds) - 1)
        return self.trace.conditions[index]

    def download(self, size_megabytes: float, start_time_seconds: float) -> DownloadResult:
        if size_megabytes < 0 or start_time_seconds < 0:
            raise ValueError("size and start time must be non-negative")
        if size_megabytes == 0:
            condition = self.condition_at(start_time_seconds)
            return DownloadResult(start_time_seconds, start_time_seconds, 0.0, 0.0, condition)

        remaining_bits = size_megabytes * 8_000_000
        current = start_time_seconds
        initial = current
        while remaining_bits > 1e-6:
            condition = self.condition_at(current)
            next_index = bisect_right(self._timestamps, current)
            boundary = (
                self._timestamps[next_index]
                if next_index < len(self._timestamps)
                else self.trace.duration_seconds
            )
            effective_mbps = condition.bandwidth_mbps * (1 - condition.packet_loss_percent / 100)
            if self.packet_loss_retries:
                effective_mbps *= max(0.01, 1 - condition.packet_loss_percent / 100)
            effective_bits_per_second = effective_mbps * 1_000_000
            if effective_bits_per_second <= 0:
                raise RuntimeError("trace has zero effective bandwidth")
            available_seconds = max(boundary - current, 0.0)
            bits_available = effective_bits_per_second * available_seconds
            if bits_available >= remaining_bits or available_seconds == 0:
                duration = remaining_bits / effective_bits_per_second
                current += duration
                remaining_bits = 0
            else:
                remaining_bits -= bits_available
                current = boundary
        duration = current - initial
        return DownloadResult(
            initial,
            current,
            duration,
            size_megabytes * 8 / duration,
            self.condition_at(initial),
        )
