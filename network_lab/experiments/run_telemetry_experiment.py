"""Run dynamic Mininet bandwidth changes while writing a network trace.

Run as root on a host with Mininet, iperf3, ping, and ``tc`` installed:
    sudo python -m network_lab.experiments.run_telemetry_experiment
"""

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Optional, Sequence

TRACE_DIR = Path(__file__).resolve().parents[2] / "experiments" / "network_traces"


class MininetHostRunner:
    """Execute commands in a Mininet host namespace, outside the CLI shell."""

    def __init__(self, host: object):
        self.host = host

    def __call__(self, command: Sequence[str], timeout: float) -> str:
        process = self.host.popen(list(command), stdout=-1, stderr=-1, text=True)
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode:
            raise RuntimeError(f"{command[0]} failed: {stderr.strip()}")
        return stdout


def read_tc_bandwidth(runner: MininetHostRunner, interface: str) -> Optional[float]:
    """Read configured rate from tc's machine-readable JSON output."""
    output = runner(("tc", "-j", "class", "show", "dev", interface), 5)
    entries = json.loads(output)
    for entry in entries:
        rate = entry.get("options", {}).get("rate")
        if rate:
            match = re.fullmatch(r"([\d.]+)\s*(Kbit|Mbit|Gbit)", rate)
            if match:
                multiplier = {"Kbit": 0.001, "Mbit": 1, "Gbit": 1000}[match.group(2)]
                return float(match.group(1)) * multiplier
    return None


def run(
    bandwidths: Sequence[float] = (2.0, 5.0, 8.0),
    phase_seconds: float = 6,
    interval_seconds: float = 2,
) -> Path:
    from network_lab.traces.models import NetworkCondition, NetworkTrace
    from network_lab.experiments.run import _replay_in_mininet

    conditions = tuple(
        NetworkCondition(index * phase_seconds, bandwidth)
        for index, bandwidth in enumerate(bandwidths)
    )
    trace = NetworkTrace(
        f"dynamic-{uuid.uuid4().hex[:8]}",
        "manual_phases",
        None,
        len(bandwidths) * phase_seconds,
        interval_seconds,
        conditions,
    )
    return _replay_in_mininet(trace, interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-seconds", type=float, default=6)
    parser.add_argument("--interval-seconds", type=float, default=2)
    args = parser.parse_args()
    print(run(phase_seconds=args.phase_seconds, interval_seconds=args.interval_seconds))


if __name__ == "__main__":
    main()
