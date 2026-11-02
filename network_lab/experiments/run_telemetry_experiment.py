"""Run dynamic Mininet bandwidth changes while writing a network trace.

Run as root on a host with Mininet, iperf3, ping, and ``tc`` installed:
    sudo python -m network_lab.experiments.run_telemetry_experiment
"""

import argparse
import json
import re
import time
import uuid
from pathlib import Path
from typing import Optional, Sequence

from network_lab.telemetry.collector import MeasurementCollector
from network_lab.telemetry.probes import MininetSampler
from network_lab.telemetry.trace_writer import CsvTraceWriter


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
    from mininet.link import TCLink
    from mininet.net import Mininet
    from mininet.node import OVSSwitch
    from network_lab.topologies.dynamic_topology import StreamTopology

    experiment_id = f"dynamic-{uuid.uuid4().hex[:8]}"
    output = TRACE_DIR / f"{experiment_id}.csv"
    net = Mininet(
        topo=StreamTopology(),
        link=TCLink,
        switch=OVSSwitch,
        controller=None,
        autoSetMacs=True,
    )
    net.start()
    server, client = net.get("h1"), net.get("h2")
    server_process = server.popen(["iperf3", "-s"], stdout=-1, stderr=-1, text=True)
    server_runner = MininetHostRunner(server)
    client_runner = MininetHostRunner(client)

    def configured_bandwidth() -> Optional[float]:
        return read_tc_bandwidth(server_runner, "h1-eth0")

    sampler = MininetSampler(
        experiment_id,
        "10.0.0.1",
        client_runner,
        configured_bandwidth,
        probe_seconds=min(1, interval_seconds),
    )
    collector = MeasurementCollector(
        experiment_id, interval_seconds, sampler, CsvTraceWriter(output)
    )
    try:
        collector.start()
        for bandwidth in bandwidths:
            server.intf("h1-eth0").config(bw=bandwidth, delay="5ms")
            time.sleep(phase_seconds)
    finally:
        collector.stop(timeout=max(10, interval_seconds + 5))
        server_process.terminate()
        server_process.wait(timeout=5)
        net.stop()
    if collector.errors:
        print(f"Telemetry probe errors: {len(collector.errors)} (missing values retained)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-seconds", type=float, default=6)
    parser.add_argument("--interval-seconds", type=float, default=2)
    args = parser.parse_args()
    print(run(phase_seconds=args.phase_seconds, interval_seconds=args.interval_seconds))


if __name__ == "__main__":
    main()
