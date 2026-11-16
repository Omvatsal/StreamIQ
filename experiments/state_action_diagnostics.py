"""State and Q-value diagnostics for the existing greedy DQN checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from network_lab.traces.models import NetworkTrace
from rl_engine.agents.dqn import DQNAgent
from rl_engine.environment.abr_env import ABREnvironment


STATE_FEATURES = (
    "buffer_occupancy_seconds",
    "throughput_history_0_mbps",
    "throughput_history_1_mbps",
    "throughput_history_2_mbps",
    "throughput_history_3_mbps",
    "throughput_history_4_mbps",
    "previous_bitrate_mbps",
    "previous_download_duration_seconds",
    "remaining_segments",
)


def extract_q_values(agent: DQNAgent, state: np.ndarray) -> tuple[np.ndarray, int, float]:
    """Extract online-network Q-values without changing model behavior."""
    with torch.no_grad():
        tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        q_values = agent.online_network(tensor).squeeze(0).cpu().numpy()
    selected = int(np.argmax(q_values))
    ordered = np.sort(q_values)
    return q_values, selected, float(ordered[-1] - ordered[-2])


def named_state(state: np.ndarray) -> dict[str, float]:
    if len(state) != len(STATE_FEATURES):
        raise ValueError(f"expected {len(STATE_FEATURES)} state values")
    return {name: float(value) for name, value in zip(STATE_FEATURES, state)}


def collect_dqn_diagnostics(
    trace: NetworkTrace,
    agent: DQNAgent,
) -> list[dict[str, object]]:
    environment = ABREnvironment(trace)
    state, _ = environment.reset(seed=trace.seed)
    rows: list[dict[str, object]] = []
    previous_bandwidth = None
    done = False
    while not done:
        features = named_state(state)
        q_values, action, margin = extract_q_values(agent, state)
        condition = trace.conditions[min(len(rows), len(trace.conditions) - 1)]
        next_state, reward, terminated, truncated, _ = environment.step(action)
        metric = environment.metrics.segments[-1]
        bandwidth_drop = (
            previous_bandwidth is not None
            and metric.bandwidth_mbps < previous_bandwidth
        )
        row: dict[str, object] = {
            "trace_id": trace.trace_id,
            "profile": trace.profile,
            "segment_index": len(rows),
            "bandwidth_mbps": metric.bandwidth_mbps,
            "previous_bandwidth_mbps": "" if previous_bandwidth is None else previous_bandwidth,
            "bandwidth_delta_mbps": (
                "" if previous_bandwidth is None else metric.bandwidth_mbps - previous_bandwidth
            ),
            "bandwidth_drop": bandwidth_drop,
            "sustained_congestion": trace.profile == "congestion",
            "state_is_normalized": False,
            "state_vector": json.dumps([float(value) for value in state]),
            "selected_action": action,
            "selected_bitrate_mbps": metric.bitrate_mbps,
            "q_value_margin": margin,
            "selected_q_value": float(q_values[action]),
            "rebuffer_seconds": metric.rebuffer_seconds,
            "reward": float(reward),
            "throughput_mbps": metric.throughput_mbps,
        }
        row.update(features)
        row.update({f"q_value_action_{index}": float(value) for index, value in enumerate(q_values)})
        rows.append(row)
        previous_bandwidth = metric.bandwidth_mbps
        state = next_state
        done = terminated or truncated
    return rows


def _plot(output: Path, rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    profiles = sorted({str(row["profile"]) for row in rows})
    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    for profile in profiles:
        values = [row for row in rows if row["profile"] == profile]
        x = np.arange(len(values))
        axes[0, 0].plot(x, [float(row["bandwidth_mbps"]) for row in values], label=profile)
        axes[0, 1].plot(x, [float(row["selected_bitrate_mbps"]) for row in values], label=profile)
        axes[1, 0].plot(x, [float(row["buffer_occupancy_seconds"]) for row in values], label=profile)
        axes[1, 1].plot(x, [float(row["q_value_margin"]) for row in values], label=profile)
    axes[0, 0].set_title("Bandwidth before DQN decisions")
    axes[0, 0].set_ylabel("Bandwidth (Mbps)")
    axes[0, 1].set_title("Selected bitrate by profile")
    axes[0, 1].set_ylabel("Bitrate (Mbps)")
    axes[1, 0].set_title("Observed buffer state")
    axes[1, 0].set_ylabel("Buffer (seconds)")
    axes[1, 1].set_title("Q-value separation")
    axes[1, 1].set_ylabel("Top Q − second Q")
    for axis in axes.flat:
        axis.set_xlabel("Decision index")
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "state_qvalues_by_profile.png", dpi=220)
    plt.close(figure)

    congestion = [row for row in rows if row["profile"] == "congestion"]
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(len(congestion))
    axes[0].plot(x, [float(row["bandwidth_mbps"]) for row in congestion], label="Bandwidth")
    axes[0].plot(x, [float(row["selected_bitrate_mbps"]) for row in congestion], label="Selected bitrate")
    drop_indices = [i for i, row in enumerate(congestion) if row["bandwidth_drop"]]
    axes[0].scatter(drop_indices, [float(congestion[i]["bandwidth_mbps"]) for i in drop_indices], marker="x", label="Bandwidth drop")
    axes[0].set_ylabel("Mbps")
    axes[0].set_title("Decisions during sustained congestion and bandwidth drops")
    axes[0].legend()
    for action in range(5):
        axes[1].plot(x, [float(row[f"q_value_action_{action}"]) for row in congestion], label=f"Action {action}")
    axes[1].set_xlabel("Decision index")
    axes[1].set_ylabel("Q-value")
    axes[1].legend(ncol=3)
    figure.tight_layout()
    figure.savefig(output / "congestion_qvalues_and_drops.png", dpi=220)
    plt.close(figure)


def run(output: Path, dataset_root: Path, checkpoint: Path) -> list[dict[str, object]]:
    traces = [NetworkTrace.load(path) for path in sorted((dataset_root / "test").glob("*.json"))]
    if not traces:
        raise FileNotFoundError(f"no test traces found in {dataset_root / 'test'}")
    agent = DQNAgent(9, seed=11, epsilon_start=0.0, epsilon_end=0.0)
    agent.load(checkpoint)
    rows = [row for trace in traces for row in collect_dqn_diagnostics(trace, agent)]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "state_action_decisions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(output, rows)

    drops = [row for row in rows if row["bandwidth_drop"]]
    congestion = [row for row in rows if row["sustained_congestion"]]
    report = [
        "# DQN State-to-Action Diagnostics",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Test dataset: `{dataset_root}`",
        f"- Decision records: {len(rows)}",
        f"- Bandwidth-drop decisions: {len(drops)}",
        f"- Sustained-congestion decisions: {len(congestion)}",
        "",
        "## State representation",
        "",
        "The environment passes the raw nine-value observation directly to the network. "
        "There is no normalization layer in `ABREnvironment` or `DQNAgent`; "
        "`state_is_normalized` is therefore `False` and `state_vector` is the exact model input.",
        "",
        "| Profile | Decisions | Mean buffer (s) | Mean throughput history (Mbps) | Mean selected bitrate (Mbps) | Mean Q margin | Mean Q0 | Mean Q1 | Mean Q2 | Mean Q3 | Mean Q4 | Action 0 | Action 1 | Action 2 | Action 3 | Action 4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in sorted({str(row["profile"]) for row in rows}):
        values = [row for row in rows if row["profile"] == profile]
        actions = [int(row["selected_action"]) for row in values]
        history = [
            np.mean(json.loads(str(row["state_vector"]))[1:6]) for row in values
        ]
        report.append(
            f"| {profile} | {len(values)} | {np.mean([float(row['buffer_occupancy_seconds']) for row in values]):.3f} | "
            f"{np.mean(history):.3f} | {np.mean([float(row['selected_bitrate_mbps']) for row in values]):.3f} | "
            f"{np.mean([float(row['q_value_margin']) for row in values]):.4f} | "
            + " | ".join(
                f"{np.mean([float(row[f'q_value_action_{action}']) for row in values]):.4f}"
                for action in range(5)
            )
            + " | "
            + " | ".join(f"{actions.count(action) / len(actions):.3f}" for action in range(5))
            + " |"
        )
    transitions = []
    for row in rows:
        if not row["bandwidth_drop"]:
            continue
        previous = next(
            (
                candidate
                for candidate in rows
                if candidate["trace_id"] == row["trace_id"]
                and int(candidate["segment_index"]) == int(row["segment_index"]) - 1
            ),
            None,
        )
        if previous is not None:
            transitions.append((previous, row))
    report.extend(
        [
            "",
            "## Drop and congestion analysis",
            "",
            f"At bandwidth-drop decisions, the mean selected bitrate was "
            f"{np.mean([float(row['selected_bitrate_mbps']) for row in drops]):.3f} Mbps "
            f"and mean Q-value margin was {np.mean([float(row['q_value_margin']) for row in drops]):.4f}."
            if drops else "No bandwidth drops were detected.",
            f"Immediately before detected drops, mean selected bitrate was "
            f"{np.mean([float(before['selected_bitrate_mbps']) for before, _ in transitions]):.3f} Mbps; "
            f"at the first decision after the drop it was "
            f"{np.mean([float(after['selected_bitrate_mbps']) for _, after in transitions]):.3f} Mbps "
            f"across {len(transitions)} paired transitions."
            if transitions else "No paired before/after bandwidth-drop transitions were detected.",
            f"The corresponding mean Q-value margin changed from "
            f"{np.mean([float(before['q_value_margin']) for before, _ in transitions]):.4f} "
            f"before the drop to {np.mean([float(after['q_value_margin']) for _, after in transitions]):.4f} "
            f"after the drop."
            if transitions else "",
            f"During sustained congestion, the mean selected bitrate was "
            f"{np.mean([float(row['selected_bitrate_mbps']) for row in congestion]):.3f} Mbps, "
            f"mean rebuffering was {np.mean([float(row['rebuffer_seconds']) for row in congestion]):.3f} seconds, "
            f"and mean Q-value margin was {np.mean([float(row['q_value_margin']) for row in congestion]):.4f}.",
            "",
            "## Evidence-backed interpretation",
            "",
            "Q-value margin measures decisiveness, not correctness. High bitrate selections "
            "are attributable to the observed state only to the extent that the logged "
            "throughput history, buffer, previous bitrate, and Q-values support that conclusion. "
            "This report does not infer causality or propose fixes.",
        ]
    )
    (output / "state_action_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/results/research_multiseed/diagnostics/state_action"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/results/research_multiseed/datasets_11"))
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/results/research_multiseed/training/11/full/model_selected.pt"))
    args = parser.parse_args()
    print(f"wrote {len(run(args.output, args.dataset_root, args.checkpoint))} state-action records")


if __name__ == "__main__":
    main()
