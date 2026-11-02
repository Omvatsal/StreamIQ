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
The independent trace engine is implemented in
[`network_lab/traces/`](/home/om_vatsal/coding/projects/StreamIQ/network_lab/traces).
It writes one incrementally flushed CSV per experiment under
[`experiments/network_traces/`](/home/om_vatsal/coding/projects/StreamIQ/experiments/network_traces/).

Generate any deterministic profile and a PNG inspection plot without Mininet:

```bash
python -m network_lab.experiments.run --profile recovery --duration 60 --seed 42
```

Supported profiles are `stable`, `random`, `oscillating`, `sudden_drop`,
`recovery`, and `congestion`. The JSON trace is reusable by other environments;
the generator uses a local seeded RNG, so identical inputs produce identical
conditions. Plots are saved under
[`experiments/results/`](/home/om_vatsal/coding/projects/StreamIQ/experiments/results/).

On a Linux host with Mininet, Open vSwitch, `iperf3`, `ping`, and `tc` installed,
replay and measure a generated profile as root:

```bash
sudo python -m network_lab.experiments.run \
  --profile recovery --duration 60 --interval 1 --seed 42 --replay
```

The runner starts the existing topology and an `iperf3` server in `h1`, applies
each condition to `h1-eth0`, and runs one sequential reverse-mode `iperf3`
probe from `h2` per trace interval. Replay and measurement are deliberately
interleaved in one process: there are no concurrent Mininet CLI or host-shell
operations. The legacy command remains available:

```bash
sudo python -m network_lab.experiments.run_telemetry_experiment
```

`TCLink.config` is called with all three shaping parameters (`bw`, `delay`,
and `loss`) on every interval. This is the Mininet traffic-control update
mechanism used by the adapter; the generated trace records the requested
condition, while telemetry records the observed values separately.

Configured bandwidth is read from `tc -j` output, throughput and download
duration come from `iperf3 -J`, and RTT/loss come from `ping`. The iperf probe
is active measurement traffic, so it can compete with application traffic and
its result is not a passive observation of another flow. Configured loss and
observed loss are distinct CSV fields. A failed probe leaves that metric blank;
no value is estimated. Ping output has no portable JSON mode, so only its
summary line is parsed. Mininet, Open vSwitch, and Linux `tc` behavior varies
on WSL2; live shaping accuracy must be validated on the target kernel.

The records are suitable for a future DQN state adapter: each row can be
normalized into configured capacity, observed throughput, configured/observed
latency and loss, and download-time features. The trace ID makes replayed
experiments joinable with their input conditions. The telemetry and trace
layers have no dependency on the DQN, video player, or frontend.

## ABR environment

The Gymnasium-compatible environment is in
[`rl_engine/environment/`](/home/om_vatsal/coding/projects/StreamIQ/rl_engine/environment).
It reuses `NetworkTrace` directly and does not require Mininet:

```bash
python -m rl_engine.environment.sample_experiment \
  --profile sudden_drop --duration 60 --seed 42
```

The sample writes a trace, per-segment CSV metrics, and a four-panel plot under
`experiments/results/`. The five actions select 240p, 360p, 480p, 720p, or
1080p. The default illustrative bitrates are 0.3, 0.7, 1.2, 2.5, and 5.0
Mbps; a four-second segment therefore has size `bitrate * 4 / 8` MB.

Each step downloads one segment by integrating available trace capacity across
trace intervals. Packet loss reduces effective capacity; delay is retained in
the network condition and available to adapters, but this simplified segment
model does not add delay to transfer time. Playback consumes the current buffer
while downloading, and any deficit is rebuffering. Reward is:

`quality_weight * quality_index - rebuffer_penalty * rebuffer_seconds - switch_penalty * abs(index_delta)`.

The observation contains buffer occupancy, recent measured throughputs,
previous bitrate, previous download duration, and remaining segments. It never
includes the next trace condition. The environment is deterministic for a
fixed trace and action sequence; `reset(seed=...)` follows Gymnasium’s API.
Metrics can be exported with `EpisodeMetrics.to_csv()`, and plotted with
`plot_episode()`. Gymnasium is declared in
[`backend/requirements.txt`](/home/om_vatsal/coding/projects/StreamIQ/backend/requirements.txt).

ABR validation currently covers buffer stalls, trace bandwidth integration,
reward switching magnitude, termination, reproducibility, CSV export, and
sudden-drop behavior. The full test suite passed locally with 14 tests.
Mininet integration remains outside this environment because Mininet is not
installed here; the ABR environment itself is independent of that limitation.
