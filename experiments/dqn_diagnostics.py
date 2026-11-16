"""Decision-level diagnostics for an existing DQN checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from network_lab.traces.models import NetworkTrace
from rl_engine.agents.dqn import DQNAgent
from rl_engine.environment.abr_env import ABREnvironment
from rl_engine.evaluation.baselines import buffer_based, fixed_bitrate, throughput_based
from rl_engine.environment.reward import RewardConfig, calculate_reward


POLICIES = ("dqn", "fixed", "throughput", "buffer")
POLICY_LABELS = {
    "dqn": "DQN",
    "fixed": "Fixed 240p",
    "throughput": "Throughput",
    "buffer": "Buffer",
}


def reward_components(
    action: int,
    rebuffer_seconds: float,
    previous_bitrate_mbps: float | None,
    bitrate_mbps: float,
    quality_weight: float = 1.0,
    rebuffer_penalty: float = 4.0,
    switch_penalty: float = 0.5,
) -> dict[str, float]:
    """Return the unchanged environment reward terms for one decision."""
    quality = quality_weight * action
    rebuffer = -rebuffer_penalty * rebuffer_seconds
    switching = (
        0.0
        if previous_bitrate_mbps is None
        else -switch_penalty * abs(bitrate_mbps - previous_bitrate_mbps)
    )
    total = calculate_reward(
        action,
        rebuffer_seconds,
        previous_bitrate_mbps,
        bitrate_mbps,
        RewardConfig(quality_weight, rebuffer_penalty, switch_penalty),
    )
    if not np.isclose(total, quality + rebuffer + switching):
        raise AssertionError("diagnostic reward terms do not match environment reward")
    return {
        "quality_reward": quality,
        "rebuffer_penalty": rebuffer,
        "switching_penalty": switching,
        "reward": total,
    }


def _record_episode(
    trace: NetworkTrace,
    policy_name: str,
    policy: Callable,
) -> list[dict[str, object]]:
    environment = ABREnvironment(trace)
    state, _ = environment.reset(seed=trace.seed)
    rows: list[dict[str, object]] = []
    done = False
    while not done:
        previous_bitrate = float(state[6]) if len(state) == 9 and state[6] > 0 else None
        action = int(policy(state, environment))
        segment_index = len(environment.metrics.segments)
        next_state, reward, terminated, truncated, _ = environment.step(action)
        metric = environment.metrics.segments[-1]
        components = reward_components(
            action,
            metric.rebuffer_seconds,
            previous_bitrate,
            metric.bitrate_mbps,
        )
        rows.append(
            {
                "trace_id": trace.trace_id,
                "profile": trace.profile,
                "policy": policy_name,
                "segment_index": segment_index,
                "bandwidth_mbps": metric.bandwidth_mbps,
                "throughput_mbps": metric.throughput_mbps,
                "throughput_history_mbps": json.dumps([float(value) for value in state[1:6]]),
                "buffer_occupancy_seconds": float(state[0]),
                "previous_bitrate_mbps": previous_bitrate if previous_bitrate is not None else "",
                "action": action,
                "selected_bitrate_mbps": metric.bitrate_mbps,
                "rebuffer_seconds": metric.rebuffer_seconds,
                "quality_reward": components["quality_reward"],
                "rebuffer_penalty": components["rebuffer_penalty"],
                "switching_penalty": components["switching_penalty"],
                "reward": float(reward),
                "download_duration_seconds": metric.download_duration_seconds,
            }
        )
        state = next_state
        done = terminated or truncated
    return rows


def _summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[
            (str(row["policy"]), str(row["profile"]), str(row["trace_id"]))
        ].append(row)
    episodes = []
    for (policy, profile, _trace_id), values in grouped.items():
        actions = [int(row["action"]) for row in values]
        bitrate_changes = [
            abs(float(current["selected_bitrate_mbps"]) - float(previous["selected_bitrate_mbps"]))
            for previous, current in zip(values, values[1:])
        ]
        average_bitrate = float(np.mean([float(row["selected_bitrate_mbps"]) for row in values]))
        rebuffer_seconds = sum(float(row["rebuffer_seconds"]) for row in values)
        variation = sum(bitrate_changes)
        episodes.append(
            {
                "policy": policy,
                "profile": profile,
                "qoe": average_bitrate - 4.0 * rebuffer_seconds - 0.5 * variation,
                "average_bitrate_mbps": average_bitrate,
                "rebuffer_seconds": rebuffer_seconds,
                "switches": sum(
                    current != previous for previous, current in zip(actions, actions[1:])
                ),
                "quality_reward": sum(float(row["quality_reward"]) for row in values),
                "rebuffer_penalty": sum(float(row["rebuffer_penalty"]) for row in values),
                "switching_penalty": sum(float(row["switching_penalty"]) for row in values),
                **{
                    f"action_{action}_frequency": actions.count(action) / len(actions)
                    for action in range(5)
                },
            }
        )
    grouped_profiles: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for episode in episodes:
        grouped_profiles[(str(episode["policy"]), str(episode["profile"]))].append(episode)
    summaries = []
    for (policy, profile), values in sorted(grouped_profiles.items()):
        summaries.append(
            {
                "policy": policy,
                "profile": profile,
                "segments": len(values),
                "qoe": np.mean([float(value["qoe"]) for value in values]),
                "average_bitrate_mbps": np.mean(
                    [float(value["average_bitrate_mbps"]) for value in values]
                ),
                "rebuffer_seconds": np.mean(
                    [float(value["rebuffer_seconds"]) for value in values]
                ),
                "switches": np.mean([float(value["switches"]) for value in values]),
                **{
                    f"action_{action}_frequency": np.mean(
                        [float(value[f"action_{action}_frequency"]) for value in values]
                    )
                    for action in range(5)
                },
                "quality_reward": np.mean(
                    [float(value["quality_reward"]) for value in values]
                ),
                "rebuffer_penalty": np.mean(
                    [float(value["rebuffer_penalty"]) for value in values]
                ),
                "switching_penalty": np.mean(
                    [float(value["switching_penalty"]) for value in values]
                ),
            }
        )
    return summaries


def _plot(output: Path, rows: list[dict[str, object]], summaries: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    profiles = sorted({str(row["profile"]) for row in rows})
    dqn = [row for row in summaries if row["policy"] == "dqn"]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    actions = np.arange(5)
    for profile in profiles:
        values = next(row for row in dqn if row["profile"] == profile)
        axes[0].plot(
            actions,
            [values[f"action_{action}_frequency"] for action in actions],
            marker="o",
            label=profile,
        )
        axes[1].bar(
            profile,
            values["qoe"],
            label=profile,
        )
    axes[0].set_title("DQN action frequency by network profile")
    axes[0].set_xlabel("Action / quality index")
    axes[0].set_ylabel("Decision frequency")
    axes[0].set_xticks(actions)
    axes[0].legend(fontsize=8)
    axes[1].set_title("DQN QoE by network profile")
    axes[1].set_ylabel("Accumulated QoE")
    axes[1].tick_params(axis="x", rotation=45)
    figure.tight_layout()
    figure.savefig(output / "dqn_profile_actions_qoe.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for policy in POLICIES:
        values = [row for row in summaries if row["policy"] == policy]
        axes[0].bar(
            POLICY_LABELS[policy],
            np.mean([float(row["qoe"]) for row in values]),
        )
        axes[1].bar(
            POLICY_LABELS[policy],
            np.mean([float(row["rebuffer_seconds"]) for row in values]),
        )
    axes[0].set_title("Policy QoE on identical unseen traces")
    axes[0].set_ylabel("Mean profile QoE")
    axes[1].set_title("Policy rebuffering on identical unseen traces")
    axes[1].set_ylabel("Mean rebuffering (seconds)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output / "policy_diagnostic_comparison.png", dpi=220)
    plt.close(figure)

    dqn_rows = [row for row in rows if row["policy"] == "dqn"]
    figure, axis = plt.subplots(figsize=(11, 5))
    x = np.arange(len(dqn_rows))
    for field, label in (
        ("quality_reward", "Quality reward"),
        ("rebuffer_penalty", "Rebuffer penalty"),
        ("switching_penalty", "Switching penalty"),
    ):
        axis.plot(x, [float(row[field]) for row in dqn_rows], label=label)
    axis.set_title("DQN reward components by decision")
    axis.set_xlabel("Decision record")
    axis.set_ylabel("Reward contribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "dqn_reward_components.png", dpi=220)
    plt.close(figure)


def run(
    output: Path,
    dataset_root: Path,
    checkpoint: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    traces = [
        NetworkTrace.load(path)
        for path in sorted((dataset_root / "test").glob("*.json"))
    ]
    if not traces:
        raise FileNotFoundError(f"no unseen test traces found in {dataset_root / 'test'}")
    agent = DQNAgent(9, seed=11, epsilon_start=0.0, epsilon_end=0.0)
    agent.load(checkpoint)
    policies = {
        "dqn": lambda state, _env: agent.select_action(state, explore=False),
        "fixed": fixed_bitrate(0),
        "throughput": throughput_based,
        "buffer": buffer_based,
    }
    rows = [
        row
        for policy_name, policy in policies.items()
        for trace in traces
        for row in _record_episode(trace, policy_name, policy)
    ]
    summaries = _summarize(rows)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "decisions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "profile_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    _plot(output, rows, summaries)
    report_lines = [
        "# DQN Decision Diagnostics",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Unseen test dataset: `{dataset_root}`",
        f"- Traces evaluated: {len(traces)}",
        f"- Policies: {', '.join(POLICY_LABELS.values())}",
        "",
        "## Findings by profile",
        "",
        "| Policy | Profile | QoE | Average bitrate (Mbps) | Rebuffer (s) | Switches | Quality reward | Rebuffer penalty | Switching penalty |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        report_lines.append(
            f"| {POLICY_LABELS[str(summary['policy'])]} | {summary['profile']} | "
            f"{float(summary['qoe']):.3f} | {float(summary['average_bitrate_mbps']):.3f} | "
            f"{float(summary['rebuffer_seconds']):.3f} | {summary['switches']} | "
            f"{float(summary['quality_reward']):.3f} | {float(summary['rebuffer_penalty']):.3f} | "
            f"{float(summary['switching_penalty']):.3f} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Decision records contain only information available before each segment download. "
            "DQN is evaluated greedily from the existing checkpoint; no training or measurement "
            "files are modified. Reward components are recomputed from the environment's existing "
            "reward definition and checked against the returned reward.",
            "",
            "The diagnostics identify poor decisions by profile, action frequency, rebuffering, "
            "and reward contribution. They are descriptive and do not establish causality.",
        ]
    )
    (output / "diagnostic_report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return rows, summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/results/research_multiseed/diagnostics"))
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("experiments/results/research_multiseed/datasets_11"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/results/research_multiseed/training/11/full/model_selected.pt"),
    )
    args = parser.parse_args()
    rows, summaries = run(args.output, args.dataset_root, args.checkpoint)
    print(f"wrote {len(rows)} decision records and {len(summaries)} profile summaries")


if __name__ == "__main__":
    main()
