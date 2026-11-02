# StreamIQ Implementation Documentation

This document summarizes the network experimentation, telemetry, adaptive
bitrate, and DQN work completed for StreamIQ.

## 1. Project scope

StreamIQ is an RL-driven adaptive video streaming platform. The implemented
work provides:

1. A reusable network trace engine.
2. Network telemetry collection for Mininet experiments.
3. A trace-backed Gymnasium ABR environment.
4. A PyTorch DQN agent, trainer, checkpoints, baselines, and evaluation.

The DQN, video player, frontend, and production streaming service are not
implemented in these milestones.

## 2. Existing network topology

The existing Mininet topology was preserved:

- `h1`: streaming server, `10.0.0.1`
- `h2`: streaming client, `10.0.0.2`
- `s1`: standalone Open vSwitch
- `h1` to `s1` and `s1` to `h2`: `TCLink`
- Existing dynamic CLI commands:
  - `setbw <mbps>`
  - `profile`

Relevant files:

- [`network_lab/topologies/basic_topology.py`](../network_lab/topologies/basic_topology.py)
- [`network_lab/topologies/dynamic_topology.py`](../network_lab/topologies/dynamic_topology.py)

The telemetry and trace replay code does not use or modify the interactive
Mininet CLI.

## 3. Network trace engine

The trace engine is under [`network_lab/traces/`](../network_lab/traces/).
Trace generation is independent of Mininet and the RL implementation.

### Trace representation

[`models.py`](../network_lab/traces/models.py) defines:

- `NetworkCondition`
  - `timestamp_seconds`
  - `bandwidth_mbps`
  - `delay_ms`
  - `packet_loss_percent`
- `TraceConfig`
- `NetworkTrace`

Validation rejects negative bandwidth, delay, or timestamps and packet-loss
values outside `0` to `100` percent. Trace timestamps must be ordered and
start at zero. Traces can be saved to and loaded from JSON.

### Profiles

[`generator.py`](../network_lab/traces/generator.py) supports:

- `stable`: small seeded bandwidth fluctuations.
- `random`: seeded random bandwidth runs with random hold durations.
- `oscillating`: repeated sinusoidal capacity changes.
- `sudden_drop`: high capacity, degradation, and restoration.
- `recovery`: degradation followed by gradual recovery.
- `congestion`: lower bandwidth plus increased delay and packet loss.

Each generation uses a local `random.Random(seed)`, so identical profile,
configuration, and seed inputs produce identical conditions.

### Replay

[`replay.py`](../network_lab/traces/replay.py) replays conditions in timestamp
order using monotonic time. The replay controller accepts an injected
`ConditionApplier`, which keeps it usable with Mininet, simulation, or another
network environment.

[`mininet.py`](../network_lab/traces/mininet.py) applies all three traffic
control values to the server egress interface:

- bandwidth
- delay
- packet loss

The Mininet adapter calls `TCLink.config(...)` for each condition. Replay runs
serially and does not create concurrent Mininet shell interactions.

### Trace commands

Generate a trace and plot:

```bash
python -m network_lab.experiments.run \
  --profile recovery \
  --duration 60 \
  --interval 1 \
  --seed 42
```

Replay and measure through Mininet:

```bash
sudo python -m network_lab.experiments.run \
  --profile recovery \
  --duration 60 \
  --interval 1 \
  --seed 42 \
  --replay
```

Generated traces are stored in
[`experiments/network_traces/`](../experiments/network_traces/). Trace plots
are stored in [`experiments/results/`](../experiments/results/).

## 4. Network telemetry

Telemetry is under [`network_lab/telemetry/`](../network_lab/telemetry/).

### Collection

[`collector.py`](../network_lab/telemetry/collector.py) provides a periodic
collector that writes completed samples incrementally. It uses a dedicated
timer thread for independent measurement work and never accesses the Mininet
CLI object.

[`probes.py`](../network_lab/telemetry/probes.py) provides:

- `iperf3 -J` for structured throughput and transfer duration.
- `ping` summary output for RTT and observed packet loss.
- `tc -j` output for configured bandwidth.

The probes run sequentially within each sample. Probe failures leave metrics
empty rather than estimating values.

### CSV schema

[`trace_writer.py`](../network_lab/telemetry/trace_writer.py) writes and
flushes one CSV record at a time. The schema is:

```text
experiment_id
trace_id
timestamp
configured_bandwidth_mbps
configured_delay_ms
configured_packet_loss_percent
throughput_mbps
rtt_ms
observed_packet_loss_percent
download_duration_seconds
```

Configured packet loss and measured packet loss are deliberately separate.
`iperf3` is active measurement traffic, so its probe can compete with an
application flow and is not passive observation.

The legacy three-bandwidth experiment remains available:

```bash
sudo python -m network_lab.experiments.run_telemetry_experiment
```

## 5. ABR environment

The Gymnasium-compatible environment is under
[`rl_engine/environment/`](../rl_engine/environment/).

### Video model

[`video.py`](../rl_engine/environment/video.py) models:

- 4-second segments.
- 10 segments by default.
- Five actions and quality levels:
  - action `0`: 240p, `0.3 Mbps`
  - action `1`: 360p, `0.7 Mbps`
  - action `2`: 480p, `1.2 Mbps`
  - action `3`: 720p, `2.5 Mbps`
  - action `4`: 1080p, `5.0 Mbps`

Segment size is calculated as:

```text
segment_size_MB = bitrate_Mbps * segment_duration_seconds / 8
```

### Network adapter

[`network.py`](../rl_engine/environment/network.py) consumes the existing
`NetworkTrace` model. It integrates available bandwidth over trace intervals
to calculate segment download duration and achieved throughput. Packet loss
reduces effective capacity. Delay is retained in each condition but is not
currently added to download duration in this simplified model.

### Buffer and rewards

[`buffer.py`](../rl_engine/environment/buffer.py) tracks:

- playback buffer occupancy
- configured buffer capacity
- rebuffering seconds

[`reward.py`](../rl_engine/environment/reward.py) calculates:

```text
reward =
    quality_weight * quality_index
    - rebuffer_penalty * rebuffer_seconds
    - switch_penalty * abs(current_quality - previous_quality)
```

All coefficients are configurable and the existing reward function was not
changed to optimize DQN results.

### State and API

[`abr_env.py`](../rl_engine/environment/abr_env.py) implements `reset()` and
`step()` using Gymnasium conventions.

The observation contains:

1. buffer occupancy
2. recent throughput history
3. previous bitrate
4. previous download duration
5. remaining segments

The next or future trace condition is not included in the observation.

### Metrics and visualization

[`metrics.py`](../rl_engine/environment/metrics.py) records per-segment:

- selected quality and bitrate
- throughput
- download duration
- buffer occupancy
- rebuffering
- reward
- bandwidth condition

It also computes accumulated reward, total rebuffering, and switch frequency.
[`visualize.py`](../rl_engine/environment/visualize.py) produces bitrate,
bandwidth, buffer, and rebuffer plots.

Run an ABR sample:

```bash
python -m rl_engine.environment.sample_experiment \
  --profile sudden_drop \
  --duration 60 \
  --seed 42
```

## 6. DQN implementation

The DQN implementation is under [`rl_engine/agents/`](../rl_engine/agents/).

### Q-network

[`network.py`](../rl_engine/agents/network.py) uses a configurable fully
connected network with ReLU hidden layers and five output Q-values.

### Replay buffer

[`replay_buffer.py`](../rl_engine/agents/replay_buffer.py) stores:

```text
(state, action, reward, next_state, done)
```

The buffer has bounded capacity and seeded sampling.

### Agent behavior

[`dqn.py`](../rl_engine/agents/dqn.py) implements:

- epsilon-greedy action selection
- online and target networks
- Bellman targets
- Smooth L1 loss
- Adam optimization
- gradient clipping
- periodic target-network synchronization
- configurable epsilon decay
- checkpoint save/load

Checkpoints include online-network weights, target-network weights, optimizer
state, epsilon, training-step count, and episode number.

## 7. Training

Training configuration is defined in
[`config.py`](../rl_engine/training/config.py). Configurable values include:

- episode count
- learning rate
- discount factor
- batch size
- replay capacity
- epsilon start/end/decay
- target update frequency
- gradient clipping
- random seed
- checkpoint frequency

[`trainer.py`](../rl_engine/training/trainer.py) selects environments from the
six deterministic trace profiles and records rewards, losses, and epsilon.

Run training and evaluation:

```bash
python -m rl_engine.training.run \
  --episodes 100 \
  --duration 60 \
  --seed 42 \
  --output-dir experiments/results/dqn
```

Resume from a checkpoint:

```bash
python -m rl_engine.training.run \
  --resume experiments/results/dqn/checkpoint_25.pt \
  --episodes 100 \
  --seed 42
```

Training outputs include:

- `training.csv`
- `training.png`
- `model_final.pt`
- periodic checkpoint files
- `evaluation.txt`
- `comparison.png`

## 8. Evaluation and baselines

Evaluation is under [`rl_engine/evaluation/`](../rl_engine/evaluation/).

Implemented policies:

- DQN greedy policy.
- Fixed-bitrate policy.
- Throughput-based policy.
- Buffer-based policy.

All policies are evaluated on identically generated profile/seed traces.

Reported metrics:

- average bitrate
- total rebuffering seconds
- bitrate switch count
- switching magnitude
- accumulated environment reward, used as QoE

Plots are produced by
[`visualize.py`](../rl_engine/evaluation/visualize.py).

## 9. Generated artifacts

Example trace JSON files and profile plots exist under:

- [`experiments/network_traces/`](../experiments/network_traces/)
- [`experiments/results/`](../experiments/results/)

An ABR sample was generated under:

- [`experiments/results/abr_sample/`](../experiments/results/abr_sample/)

A DQN sample run was generated under:

- [`experiments/results/dqn_sample/`](../experiments/results/dqn_sample/)

The sample DQN evaluation recorded:

| Policy | Average bitrate (Mbps) | Rebuffer (s) | Switches | Switching magnitude | QoE |
|---|---:|---:|---:|---:|---:|
| DQN | 1.8067 | 8.9110 | 14 | 14 | 105.3561 |
| Fixed 240p | 0.3000 | 1.0996 | 0 | 0 | -4.3985 |
| Throughput | 4.7917 | 41.6080 | 1 | 1 | 68.0679 |
| Buffer | 1.0467 | 1.0996 | 22 | 32 | 69.1015 |

These are results from a short three-episode sample run, not a production
benchmark or claim of converged policy quality.

## 10. Tests and validation

Test files:

- [`test_telemetry.py`](../tests/test_telemetry.py)
- [`test_traces.py`](../tests/test_traces.py)
- [`test_abr_environment.py`](../tests/test_abr_environment.py)
- [`test_dqn.py`](../tests/test_dqn.py)

The latest local validation run completed:

```text
21 passed
```

Additional checks completed:

- Python compilation for `network_lab`, `rl_engine`, and `tests`.
- `git diff --check`.
- Gymnasium environment checker passed.
- All six trace profiles generated and validated.
- Deterministic seed behavior validated.
- DQN sample training and evaluation completed on CPU.

Milestone 6 research validation also executed:

- One-seed, 20-episode run across all six profiles and six retrained ablations.
- Disjoint train/validation/test datasets generated with seeds
  `101-106`, `201-206`, and `301-306`.
- Validation-selected checkpoints were written for each ablation.
- Unseen-test policy results, paired confidence intervals, effect sizes, and
  paired t-test p-values were written to the research results directory.
- A one-episode follow-up smoke run verified the paired-statistics output and
  generated training and comparison plots.

These runs are engineering validation and smoke-scale research runs. They do
not establish DQN convergence or prove the primary oscillation-reduction
objective. The one-seed results include cases where DQN underperformed a
baseline, so conclusions require longer multi-seed experiments. The current
evidence does not support claiming that DQN has achieved the primary objective.

## 11. Dependencies

The dependency manifest is
[`backend/requirements.txt`](../backend/requirements.txt). Relevant additions
are:

- `gymnasium==1.2.2`
- `torch==2.9.0`

The local validation used the CPU build `torch==2.9.0+cpu`.

## 12. Known limitations

1. Mininet was not installed in the development runtime, so live Mininet
   replay, Open vSwitch shaping, and `iperf3` network measurements were not
   executed here.
2. Mininet, Open vSwitch, and Linux `tc` behavior can differ on WSL2 and must
   be validated on the target kernel.
3. The ABR network adapter models trace capacity and packet loss but does not
   currently add configured delay to transfer time.
4. The telemetry `iperf3` probe is active traffic and may affect the measured
   network.
5. Ping parsing uses its human-readable summary because a portable ping JSON
   format is not assumed.
6. The DQN sample uses only three episodes and is intended as a smoke test,
   not a tuned training study.
7. Baselines and DQN use the current simplified ABR reward and environment;
   no claim is made that the reward is calibrated for real QoE.
8. No DQN training, video player, frontend, or deployment integration is
   included beyond the reusable interfaces documented here.

## 13. Future extension points

- Add a real video manifest and segment catalog.
- Add delay-aware download timing and more detailed packet-loss/retransmission
  modeling.
- Add separate train/validation/test trace sets.
- Add longer DQN runs and hyperparameter studies.
- Add Double DQN, prioritized replay, or other controlled algorithm variants.
- Connect telemetry records to a deployed ABR controller.
- Add Mininet integration tests on a compatible Linux host.

## 14. Research validation and Milestone 6 corrections

Research-validation additions are under
[`experiments/configurations/`](../experiments/configurations/) and
[`experiments/research_validation.py`](../experiments/research_validation.py).
Training, validation, and unseen test splits use disjoint reproducible seed
sets across all six profiles.

The ABR environment now includes configured one-way delay once per segment
download. Packet loss remains an effective-capacity approximation:
available capacity is multiplied by `1 - loss_percent / 100`; optional retry
mode applies that factor again. This is not a packet-level retransmission
simulator.

Switch penalties now use absolute bitrate change in Mbps rather than quality
index distance. The default quality utility remains quality-index based, so
the existing reward formulation is not silently replaced. State ablations can
disable throughput history or previous bitrate while preserving the default
observation design.

Evaluation metrics are independent of the training reward. They include
average bitrate, rebuffer seconds and ratio, switch count, average absolute
bitrate change, total bitrate variation, upward/downward switches, average
download duration, and an independently calculated QoE summary.

[`statistics.py`](../rl_engine/evaluation/statistics.py) provides paired mean
differences, standard deviations, confidence intervals, Cohen's `d_z`, and
paired t-test p-values when SciPy is available.
[`ablations.py`](../rl_engine/training/ablations.py) defines retrained reward
and state ablations.
