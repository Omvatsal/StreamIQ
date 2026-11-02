"""Generate traces and optionally replay them in Mininet.

Examples:
    python -m network_lab.experiments.run --profile recovery --duration 20 --seed 42
    sudo python -m network_lab.experiments.run --profile congestion --duration 30 --replay
"""

import argparse
import time
import uuid
from pathlib import Path
from typing import Optional

from network_lab.telemetry.probes import MininetSampler
from network_lab.telemetry.trace_writer import CsvTraceWriter
from network_lab.traces.generator import PROFILE_NAMES, generate_trace
from network_lab.traces.models import NetworkCondition, NetworkTrace, TraceConfig
from network_lab.traces.mininet import MininetTrafficShaper
from network_lab.traces.visualize import plot_trace

ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = ROOT / "experiments" / "network_traces"
RESULT_DIR = ROOT / "experiments" / "results"


def _replay_in_mininet(trace: NetworkTrace, interval_seconds: float) -> Path:
    from mininet.link import TCLink
    from mininet.net import Mininet
    from mininet.node import OVSSwitch
    from network_lab.topologies.dynamic_topology import StreamTopology
    from .run_telemetry_experiment import MininetHostRunner

    from network_lab.experiments.run_telemetry_experiment import read_tc_bandwidth

    experiment_id = f"{trace.trace_id}-{uuid.uuid4().hex[:8]}"
    output = TRACE_DIR / f"{experiment_id}.csv"
    net = Mininet(topo=StreamTopology(), link=TCLink, switch=OVSSwitch, controller=None, autoSetMacs=True)
    net.start()
    server, client = net.get("h1"), net.get("h2")
    server_process = server.popen(["iperf3", "-s"], stdout=-1, stderr=-1, text=True)
    client_runner = MininetHostRunner(client)
    server_runner = MininetHostRunner(server)
    current = trace.conditions[0]

    def configured() -> tuple[Optional[float], Optional[float], Optional[float]]:
        return (
            read_tc_bandwidth(server_runner, "h1-eth0"),
            current.delay_ms,
            current.packet_loss_percent,
        )

    sampler = MininetSampler(
        experiment_id,
        "10.0.0.1",
        client_runner,
        lambda: configured()[0],
        trace_id=trace.trace_id,
        configured_reader=configured,
        probe_seconds=min(1, interval_seconds),
    )
    try:
        with CsvTraceWriter(output) as writer:
            started = time.monotonic()
            for condition in trace.conditions:
                target = started + condition.timestamp_seconds
                if target > time.monotonic():
                    time.sleep(target - time.monotonic())
                current = condition
                MininetTrafficShaper(server.intf("h1-eth0")).apply(condition)
                writer.write(sampler(sampler_timestamp()))
    finally:
        server_process.terminate()
        server_process.wait(timeout=5)
        net.stop()
    return output


def sampler_timestamp():
    from network_lab.telemetry.metrics import utc_now

    return utc_now()


def generate(profile: str, duration: float, interval: float, seed: Optional[int]) -> NetworkTrace:
    config = TraceConfig(
        duration_seconds=duration,
        interval_seconds=interval,
        seed=seed,
    )
    trace = generate_trace(profile, f"{profile}_{seed if seed is not None else 'default'}", config)
    trace.save(TRACE_DIR / f"{trace.trace_id}.json")
    plot_trace(trace, RESULT_DIR / f"{trace.trace_id}.png")
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=PROFILE_NAMES, default="stable")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    trace = generate(args.profile, args.duration, args.interval, args.seed)
    print(f"Generated {trace.trace_id}: {TRACE_DIR / (trace.trace_id + '.json')}")
    if args.replay:
        print(f"Telemetry CSV: {_replay_in_mininet(trace, args.interval)}")


if __name__ == "__main__":
    main()
