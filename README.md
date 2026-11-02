# StreamIQ

RL-Driven Adaptive Video Streaming Platform
with Temporal Stability Optimization.

## Core Components

- Adaptive Bitrate Streaming
- Deep Q-Network Controller
- Network Simulation and Emulation
- MPEG-DASH Video Delivery
- Streaming Analytics Dashboard

## Architecture

- Frontend: React + TypeScript
- Backend: FastAPI
- RL: PyTorch
- Network Lab: Mininet
- Video: FFmpeg + MPEG-DASH

## Network telemetry

Telemetry is implemented in [`network_lab/telemetry/`](/home/om_vatsal/coding/projects/StreamIQ/network_lab/telemetry).
It writes one incrementally flushed CSV per experiment under
[`experiments/network_traces/`](/home/om_vatsal/coding/projects/StreamIQ/experiments/network_traces/).

On a Linux host with Mininet, Open vSwitch, `iperf3`, `ping`, and `tc` installed,
run the validation experiment as root:

```bash
sudo python -m network_lab.experiments.run_telemetry_experiment \
  --phase-seconds 6 --interval-seconds 2
```

The runner starts the existing dynamic topology, starts an `iperf3` server in
`h1`, and runs sequential reverse-mode `iperf3` probes from `h2` while the
bandwidth changes through 2, 5, and 8 Mbps phases. Probe commands run through
Mininet host namespaces using `popen`; they do not use or manipulate the
interactive CLI shell.

Configured bandwidth is read from `tc -j` output, throughput and download
duration come from `iperf3 -J`, and RTT/loss come from `ping`. The iperf probe
is active measurement traffic, so it can compete with application traffic and
its result is not a passive observation of another flow. A failed probe leaves
that metric blank in the CSV; no value is estimated. Ping output has no
portable JSON mode, so only its summary line is parsed. The resulting CSV
contains UTC timestamps, which should be ordered by the collector.

The records are suitable for a future DQN state adapter: each row can be
normalized into configured capacity, observed throughput, latency, loss, and
download-time features. The telemetry layer has no dependency on the DQN,
video player, or frontend.
