"""Structured command probes used by the Mininet adapter."""

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional, Sequence

from .metrics import Measurement


CommandRunner = Callable[[Sequence[str], float], str]


@dataclass
class IperfResult:
    throughput_mbps: Optional[float]
    duration_seconds: Optional[float]


def subprocess_runner(command: Sequence[str], timeout: float) -> str:
    result = subprocess.run(
        list(command), capture_output=True, text=True, timeout=timeout, check=True
    )
    return result.stdout


def iperf3_probe(
    runner: CommandRunner, server: str, duration_seconds: float, timeout: float
) -> IperfResult:
    output = runner(
        ("iperf3", "-c", server, "-R", "-t", str(max(1, int(duration_seconds))), "-J"),
        timeout,
    )
    payload = json.loads(output)
    summary = payload["end"].get("sum_received") or payload["end"]["sum"]
    return IperfResult(
        throughput_mbps=float(summary["bits_per_second"]) / 1_000_000,
        duration_seconds=float(summary["seconds"]),
    )


def ping_probe(
    runner: CommandRunner, server: str, count: int, timeout: float
) -> tuple[Optional[float], Optional[float]]:
    output = runner(("ping", "-n", "-c", str(count), server), timeout)
    loss_match = re.search(r"(\d+(?:\.\d+)?)% packet loss", output)
    rtt_match = re.search(r"=\s*[\d.]+/([\d.]+)/", output)
    return (
        float(rtt_match.group(1)) if rtt_match else None,
        float(loss_match.group(1)) if loss_match else None,
    )


class MininetSampler:
    """Collect one sequential iperf3 and ping sample through an injected runner."""

    def __init__(
        self,
        experiment_id: str,
        server_ip: str,
        runner: CommandRunner,
        bandwidth_reader: Callable[[], Optional[float]],
        trace_id: Optional[str] = None,
        configured_reader: Optional[Callable[[], tuple[Optional[float], Optional[float], Optional[float]]]] = None,
        probe_seconds: float = 1,
        command_timeout: float = 10,
    ):
        self.experiment_id = experiment_id
        self.server_ip = server_ip
        self.runner = runner
        self.bandwidth_reader = bandwidth_reader
        self.trace_id = trace_id
        self.configured_reader = configured_reader
        self.probe_seconds = probe_seconds
        self.command_timeout = command_timeout

    def __call__(self, timestamp: datetime) -> Measurement:
        configured = None
        configured_delay = configured_loss = None
        try:
            if self.configured_reader is not None:
                configured, configured_delay, configured_loss = self.configured_reader()
            else:
                configured = self.bandwidth_reader()
        except (OSError, RuntimeError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        throughput = duration = rtt = loss = None
        try:
            result = iperf3_probe(
                self.runner, self.server_ip, self.probe_seconds, self.command_timeout
            )
            throughput, duration = result.throughput_mbps, result.duration_seconds
        except (
            OSError,
            RuntimeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
            ValueError,
        ):
            pass
        try:
            rtt, loss = ping_probe(
                self.runner, self.server_ip, 3, self.command_timeout
            )
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
            pass
        return Measurement(
            experiment_id=self.experiment_id,
            timestamp=timestamp,
            trace_id=self.trace_id,
            configured_bandwidth_mbps=configured,
            configured_delay_ms=configured_delay,
            configured_packet_loss_percent=configured_loss,
            throughput_mbps=throughput,
            rtt_ms=rtt,
            packet_loss_percent=loss,
            download_duration_seconds=duration,
        )
