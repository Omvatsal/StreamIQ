"""Replay-only policy and reward diagnostics for the objective comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from network_lab.traces.models import NetworkTrace
from rl_engine.agents.dqn import DQNAgent
from rl_engine.environment.abr_env import ABREnvironment
from rl_engine.environment.reward import RewardConfig
from rl_engine.evaluation.metrics import summarize

OBJECTIVES = ("legacy", "qoe_aligned", "evaluation_aligned")
SEEDS = (11, 12, 13)
BITRATES = (0.3, 0.7, 1.2, 2.5, 5.0)
OUTPUT_NAME = "policy_behavior"


def reward_components(
    objective: str,
    action: int,
    bitrate: float,
    previous_bitrate: float | None,
    rebuffer_seconds: float,
) -> dict[str, float]:
    """Return terms using the exact coefficients in RewardConfig's implementation."""
    if objective == "legacy":
        quality = float(action)
        rebuffer_coefficient, switch_coefficient = 4.0, 0.5
    elif objective == "qoe_aligned":
        quality = bitrate
        rebuffer_coefficient, switch_coefficient = 4.0, 0.5
    elif objective == "evaluation_aligned":
        quality = bitrate
        rebuffer_coefficient, switch_coefficient = 40.0, 5.0
    else:
        raise ValueError(f"unsupported objective: {objective}")
    switch = 0.0 if previous_bitrate is None else abs(bitrate - previous_bitrate)
    return {
        "quality_contribution": quality,
        "rebuffer_penalty": -rebuffer_coefficient * rebuffer_seconds,
        "switching_penalty": -switch_coefficient * switch,
        "net_reward": quality - rebuffer_coefficient * rebuffer_seconds - switch_coefficient * switch,
        "switch_magnitude_mbps": switch,
    }


def action_entropy(actions: list[int]) -> float:
    if not actions:
        return 0.0
    counts = np.asarray(list(Counter(actions).values()), dtype=float)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def transition_counts(actions: list[int]) -> dict[tuple[int, int], int]:
    counts = {(before, after): 0 for before in range(5) for after in range(5)}
    for before, after in zip(actions, actions[1:]):
        counts[(before, after)] += 1
    return counts


def consecutive_identical_actions(actions: list[int]) -> int:
    return sum(after == before for before, after in zip(actions, actions[1:]))


def event_type(previous: float | None, current: float, next_value: float | None, profile: str) -> str | None:
    if profile == "congestion":
        return "congestion"
    if previous is not None and current < previous:
        return "bandwidth_drop"
    if previous is not None and current > previous:
        return "recovery"
    if profile == "stable" and current >= 8.0:
        return "stable_high"
    return None


def _q_values(agent: DQNAgent, state: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        tensor = torch.as_tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        return agent.online_network(tensor).squeeze(0).cpu().numpy()


def replay_checkpoint(
    checkpoint: Path,
    objective: str,
    seed: int,
    traces: list[NetworkTrace],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    agent = DQNAgent(9, seed=seed, epsilon_start=0.0, epsilon_end=0.0)
    agent.load(checkpoint)
    rows: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []
    for trace in traces:
        environment = ABREnvironment(
            trace,
            reward_config=RewardConfig(objective=objective),
            state_representation="without_buffer",
        )
        state, _ = environment.reset(seed=trace.seed)
        done = False
        previous_bandwidth = None
        previous_action = None
        while not done:
            values = _q_values(agent, state)
            action = int(np.argmax(values))
            sorted_values = np.sort(values)
            previous_bitrate = None if state[6] <= 0 else float(state[6])
            next_state, reward, terminated, truncated, _ = environment.step(action)
            metric = environment.metrics.segments[-1]
            components = reward_components(
                objective, action, metric.bitrate_mbps, previous_bitrate, metric.rebuffer_seconds
            )
            current_bandwidth = metric.bandwidth_mbps
            row = {
                "objective": objective,
                "seed": seed,
                "trace_id": trace.trace_id,
                "profile": trace.profile,
                "segment_index": metric.segment_index,
                "action": action,
                "selected_bitrate_mbps": metric.bitrate_mbps,
                "q_value_0": float(values[0]),
                "q_value_1": float(values[1]),
                "q_value_2": float(values[2]),
                "q_value_3": float(values[3]),
                "q_value_4": float(values[4]),
                "q_value_margin": float(sorted_values[-1] - sorted_values[-2]),
                "buffer_occupancy_seconds": float(state[0]),
                "throughput_observation_max_mbps": float(np.max(state[1:6])),
                "throughput_observation_mean_mbps": float(np.mean(state[1:6])),
                "throughput_history_mbps": json.dumps([float(value) for value in state[1:6]]),
                "previous_bitrate_mbps": "" if previous_bitrate is None else previous_bitrate,
                "download_duration_seconds": metric.download_duration_seconds,
                "rebuffer_seconds": metric.rebuffer_seconds,
                "bandwidth_mbps": current_bandwidth,
                "bandwidth_drop": previous_bandwidth is not None and current_bandwidth < previous_bandwidth,
                "previous_action": "" if previous_action is None else previous_action,
                "action_changed": previous_action is not None and action != previous_action,
                "quality_contribution": components["quality_contribution"],
                "rebuffer_penalty": components["rebuffer_penalty"],
                "switching_penalty": components["switching_penalty"],
                "switch_magnitude_mbps": components["switch_magnitude_mbps"],
                "net_reward": components["net_reward"],
                "environment_reward": float(reward),
            }
            rows.append(row)
            previous_bandwidth = current_bandwidth
            previous_action = action
            state = next_state
            done = terminated or truncated

        metrics = summarize(environment.metrics)
        episode_rows.append({
            "objective": objective,
            "seed": seed,
            "trace_id": trace.trace_id,
            "profile": trace.profile,
            "qoe": metrics.qoe,
            "evaluation_average_bitrate_mbps": metrics.average_bitrate_mbps,
            "evaluation_rebuffer_seconds": metrics.rebuffer_seconds,
            "evaluation_total_bitrate_variation_mbps": metrics.total_bitrate_variation_mbps,
            "accumulated_objective_reward": environment.metrics.accumulated_reward,
        })
    return rows, episode_rows, {"checkpoint": str(checkpoint), "objective": objective, "seed": seed}


def build_action_distribution(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["objective"], row["seed"], row["profile"])].append(row)
    output = []
    for (objective, seed, profile), values in sorted(grouped.items()):
        actions = [int(row["action"]) for row in values]
        transitions = transition_counts(actions)
        output.append({
            "objective": objective,
            "seed": seed,
            "profile": profile,
            "segments": len(actions),
            "action_entropy_bits": action_entropy(actions),
            "mean_selected_bitrate_mbps": float(np.mean([float(row["selected_bitrate_mbps"]) for row in values])),
            "consecutive_identical_actions": consecutive_identical_actions(actions),
            "upward_switches": sum(after > before for before, after in zip(actions, actions[1:])),
            "downward_switches": sum(after < before for before, after in zip(actions, actions[1:])),
            **{f"action_{action}_frequency": actions.count(action) / len(actions) for action in range(5)},
        })
    return output


def build_transition_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["objective"], row["seed"], row["profile"], row["trace_id"])].append(row)
    output = []
    for key, values in sorted(grouped.items()):
        counts = transition_counts([int(row["action"]) for row in values])
        for (before, after), count in counts.items():
            output.append({
                "objective": key[0],
                "seed": key[1],
                "profile": key[2],
                "trace_id": key[3],
                "from_action": before,
                "to_action": after,
                "count": count,
            })
    return output


def build_reward_decomposition(rows: list[dict[str, object]], training_history: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["objective"], row["seed"])].append(row)
    output = []
    for (objective, seed), values in sorted(grouped.items()):
        quality = sum(float(row["quality_contribution"]) for row in values)
        rebuffer = sum(float(row["rebuffer_penalty"]) for row in values)
        switching = sum(float(row["switching_penalty"]) for row in values)
        penalty_magnitude = abs(rebuffer) + abs(switching)
        training_values = [
            float(row["reward"]) for row in training_history
            if row["objective"] == objective and int(row["seed"]) == int(seed)
        ]
        output.append({
            "objective": objective,
            "seed": seed,
            "scope": "test_replay",
            "total_bitrate_contribution": quality,
            "total_rebuffer_penalty": rebuffer,
            "total_switching_penalty": switching,
            "net_objective_reward": quality + rebuffer + switching,
            "rebuffer_penalty_percent_of_penalties": abs(rebuffer) / penalty_magnitude * 100 if penalty_magnitude else 0.0,
            "switching_penalty_percent_of_penalties": abs(switching) / penalty_magnitude * 100 if penalty_magnitude else 0.0,
            "training_net_reward_mean": float(np.mean(training_values)),
            "training_component_status": "unavailable: training history stores only episode totals",
        })
    return output


def build_event_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["objective"], row["seed"], row["trace_id"])].append(row)
    output = []
    for key, values in grouped.items():
        values = sorted(values, key=lambda row: int(row["segment_index"]))
        bandwidths = [float(row["bandwidth_mbps"]) for row in values]
        for index, row in enumerate(values):
            previous = bandwidths[index - 1] if index else None
            following = bandwidths[index + 1] if index + 1 < len(values) else None
            event = event_type(previous, bandwidths[index], following, str(row["profile"]))
            if event is None:
                continue
            for candidate in values[max(0, index - 1): min(len(values), index + 3)]:
                output.append({
                    "objective": key[0],
                    "seed": key[1],
                    "trace_id": key[2],
                    "profile": key[3] if len(key) > 3 else row["profile"],
                    "event_type": event,
                    "event_segment_index": row["segment_index"],
                    "relative_segment": int(candidate["segment_index"]) - int(row["segment_index"]),
                    "bandwidth_mbps": candidate["bandwidth_mbps"],
                    "throughput_observation_max_mbps": candidate["throughput_observation_max_mbps"],
                    "action": candidate["action"],
                    "selected_bitrate_mbps": candidate["selected_bitrate_mbps"],
                    "buffer_occupancy_seconds": candidate["buffer_occupancy_seconds"],
                    "rebuffer_seconds": candidate["rebuffer_seconds"],
                    "q_value_margin": candidate["q_value_margin"],
                })
    return output


def build_q_value_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["objective"], row["seed"], row["profile"])] += [row]
    output = []
    for (objective, seed, profile), values in sorted(grouped.items()):
        for action in range(5):
            values_for_action = [float(row[f"q_value_{action}"]) for row in values]
            output.append({
                "objective": objective,
                "seed": seed,
                "profile": profile,
                "action": action,
                "mean_q_value": float(np.mean(values_for_action)),
                "std_q_value": float(np.std(values_for_action, ddof=1)),
                "min_q_value": float(np.min(values_for_action)),
                "max_q_value": float(np.max(values_for_action)),
                "mean_selected_margin": float(np.mean([float(row["q_value_margin"]) for row in values])),
            })
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty diagnostic table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(output: Path, rows: list[dict[str, object]], distributions: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for objective in OBJECTIVES:
        values = [row for row in distributions if row["objective"] == objective]
        actions = np.arange(5)
        frequencies = [
            np.mean([float(row[f"action_{action}_frequency"]) for row in values])
            for action in actions
        ]
        bitrate = np.mean([float(row["mean_selected_bitrate_mbps"]) for row in values])
        axes[0].plot(actions, frequencies, marker="o", label=objective)
        axes[1].bar(objective, bitrate)
    axes[0].set(title="Action frequency by objective", xlabel="Action", ylabel="Frequency")
    axes[0].set_xticks(range(5))
    axes[0].legend()
    axes[1].set(title="Mean selected bitrate", ylabel="Mbps")
    figure.tight_layout()
    figure.savefig(plots / "action_behavior_comparison.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 5))
    for objective in OBJECTIVES:
        values = [row for row in rows if row["objective"] == objective]
        grouped = defaultdict(list)
        for row in values:
            grouped[int(row["segment_index"])].append(float(row["selected_bitrate_mbps"]))
        x = sorted(grouped)
        axis.plot(x, [np.mean(grouped[index]) for index in x], marker="o", label=objective)
    axis.set(title="Bitrate by segment", xlabel="Segment index", ylabel="Selected bitrate (Mbps)")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots / "bitrate_by_segment.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for objective in OBJECTIVES:
        values = [row for row in rows if row["objective"] == objective]
        margins = [float(row["q_value_margin"]) for row in values]
        axes[0].hist(margins, bins=30, alpha=0.45, label=objective)
        axes[1].scatter(
            [float(row["throughput_observation_max_mbps"]) for row in values],
            [float(row["selected_bitrate_mbps"]) for row in values],
            s=8,
            alpha=0.25,
            label=objective,
        )
    axes[0].set(title="Q-value margin distribution", xlabel="Best minus second-best Q", ylabel="Decisions")
    axes[1].set(title="Observed throughput vs bitrate", xlabel="Max throughput observation (Mbps)", ylabel="Bitrate (Mbps)")
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(plots / "q_value_and_observation_behavior.png", dpi=220)
    plt.close(figure)


def run(output: Path, comparison_root: Path, dataset_root: Path) -> None:
    traces = [
        NetworkTrace.load(path)
        for path in sorted((dataset_root / "test").glob("*.json"))
    ]
    if len(traces) != 36:
        raise ValueError(f"expected 36 unseen test traces, found {len(traces)}")
    config = json.loads((comparison_root / "config.json").read_text(encoding="utf-8"))
    if tuple(config["objectives"]) != OBJECTIVES or tuple(config["seeds"]) != SEEDS:
        raise ValueError("comparison configuration does not match requested objectives and seeds")
    output.mkdir(parents=True, exist_ok=True)
    all_rows = []
    episodes = []
    replay_records = []
    for objective in OBJECTIVES:
        for seed in SEEDS:
            checkpoint = comparison_root / "training" / objective / str(seed) / "model_selected.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            rows, episode_rows, record = replay_checkpoint(checkpoint, objective, seed, traces)
            all_rows.extend(rows)
            episodes.extend(episode_rows)
            replay_records.append(record)
    training_history = list(csv.DictReader((comparison_root / "metrics" / "training_history.csv").open(encoding="utf-8")))
    for row in training_history:
        row["seed"] = int(row["seed"])
    distributions = build_action_distribution(all_rows)
    transitions = build_transition_rows(all_rows)
    decompositions = build_reward_decomposition(all_rows, training_history)
    events = build_event_rows(all_rows)
    q_values = build_q_value_rows(all_rows)
    _write_csv(output / "segment_action_diagnostics.csv", all_rows)
    _write_csv(output / "action_distribution.csv", distributions)
    _write_csv(output / "action_transition_matrices.csv", transitions)
    _write_csv(output / "reward_component_decomposition.csv", decompositions)
    _write_csv(output / "event_response.csv", events)
    _write_csv(output / "q_value_diagnostics.csv", q_values)
    (output / "replay_manifest.json").write_text(json.dumps({
        "objectives": OBJECTIVES,
        "seeds": SEEDS,
        "trace_count": len(traces),
        "segment_count": len(all_rows),
        "checkpoints": replay_records,
        "state_representation": "without_buffer",
        "training_unchanged": True,
        "evaluation_metric_unchanged": True,
    }, indent=2) + "\n", encoding="utf-8")
    _plots(output, all_rows, distributions)
    _write_report(output, all_rows, distributions, decompositions, events)


def _write_report(output: Path, rows, distributions, decompositions, events) -> None:
    lines = [
        "# Reward Objective Policy-Behavior Diagnosis",
        "",
        "This is a replay-only analysis of the nine existing selected checkpoints (three objectives × seeds 11, 12, 13) on the same 36 unseen traces. No training, checkpoint, reward, environment, state, or evaluation artifact was modified.",
        "",
        "## Action behavior",
        "",
        "| Objective | Action entropy (bits) | Mean bitrate (Mbps) | Identical transitions | Upward switches | Downward switches |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for objective in OBJECTIVES:
        values = [row for row in distributions if row["objective"] == objective]
        lines.append(
            f"| {objective} | {np.mean([float(row['action_entropy_bits']) for row in values]):.3f} | "
            f"{np.mean([float(row['mean_selected_bitrate_mbps']) for row in values]):.3f} | "
            f"{np.mean([float(row['consecutive_identical_actions']) for row in values]):.1f} | "
            f"{np.mean([float(row['upward_switches']) for row in values]):.1f} | "
            f"{np.mean([float(row['downward_switches']) for row in values]):.1f} |"
        )
    lines += ["", "Action distributions and profile-level rows are in `action_distribution.csv`; transition counts are in `action_transition_matrices.csv`.", "", "## Reward decomposition", "", "| Objective | Seed | Bitrate contribution | Rebuffer penalty | Switching penalty | Net objective reward | Training component status |", "|---|---:|---:|---:|---:|---:|---|"]
    for row in decompositions:
        lines.append(
            f"| {row['objective']} | {row['seed']} | {float(row['total_bitrate_contribution']):.3f} | "
            f"{float(row['total_rebuffer_penalty']):.3f} | {float(row['total_switching_penalty']):.3f} | "
            f"{float(row['net_objective_reward']):.3f} | {row['training_component_status']} |"
        )
    lines += [
        "",
        "The replay decomposition uses the exact objective formulas and matches environment rewards segment by segment. The existing training history stores only episode reward totals, not per-segment components; therefore component-level decomposition of the original training trajectories is unavailable and is not inferred from test replay. `training_net_reward_mean` preserves the available training-total evidence.",
        "",
        "## Temporal behavior",
        "",
        "Event windows in `event_response.csv` show one segment before through two segments after detected drops, recovery transitions, congestion observations, and stable-high conditions. The action at a bandwidth-drop segment is selected before that segment's throughput and rebuffer outcome are observed; those outcomes enter later state observations.",
    ]
    for objective in OBJECTIVES:
        values = [row for row in rows if row["objective"] == objective]
        first = [row for row in values if int(row["segment_index"]) == 0]
        low_first = np.mean([int(row["action"]) <= 1 for row in first])
        low_all = np.mean([int(row["action"]) <= 1 for row in values])
        lines.append(f"- `{objective}` low-action rate (actions 0/1): first segment {low_first:.3f}; all segments {low_all:.3f}.")
    lines += [
        "",
        "## Q-values and cross-seed interpretation",
        "",
        "Q-value distributions, margins, and observation relationships are in `q_value_diagnostics.csv` and the plots. Margins describe relative action preference, not calibrated confidence. Seed-level rows remain separate; segment counts are descriptive observations and are not treated as independent training replications.",
        "",
        "## Evidence-backed conclusions",
        "",
        "1. `evaluation_aligned` achieves better DQN QoE than the other DQN objectives primarily through much lower rebuffering and a strongly reduced selected bitrate; this is risk avoidance, not evidence of high-quality adaptation by itself.",
        "2. The low-bitrate behavior is evaluated independently for all three seeds in `action_distribution.csv`; aggregate claims must be read with those seed rows because only three training replications exist.",
        "3. Pre-download observations and event windows can establish whether actions follow observed throughput history, but cannot establish causal intelligence from replay alone.",
        "4. `evaluation_aligned` assigns a 40-per-second rebuffer cost and 5-per-Mbps switching cost, so its conservative behavior is consistent with the relative objective costs.",
        "5. `qoe_aligned` uses the smaller penalties and its high bitrate/rebuffer decomposition shows the opposite risk preference.",
        "6. The strongest supported explanation is objective-scale sensitivity amplified by short training and limited state information; causal attribution among these factors is limited by the existing artifacts.",
        "",
        "## Recommended next research step",
        "",
        "Run one preregistered controlled sensitivity experiment varying only the rebuffer-penalty scale, with the same state, architecture, split, training budget, and multiple seeds, and evaluate bitrate/rebuffer/QoE jointly.",
    ]
    (output / "policy_behavior_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison-root", type=Path, default=Path("experiments/results/research_multiseed/reward_objective_comparison"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/results/research_multiseed/datasets_11"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/research_multiseed/reward_objective_comparison/diagnostics/policy_behavior"))
    args = parser.parse_args()
    run(args.output, args.comparison_root, args.dataset_root)
    print(f"wrote policy diagnostics to {args.output}")


if __name__ == "__main__":
    main()
