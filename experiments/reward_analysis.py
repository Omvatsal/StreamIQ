"""Read-only analysis of StreamIQ reward components and evaluation QoE."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def enrich(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    enriched = []
    previous_bandwidth: dict[tuple[str, str], float] = {}
    for row in rows:
        item = dict(row)
        quality = float(row["quality_reward"])
        rebuffer = float(row["rebuffer_penalty"])
        switching = float(row["switching_penalty"])
        reward = float(row["reward"])
        rebuffer_seconds = float(row["rebuffer_seconds"])
        key = (row["trace_id"], row["policy"])
        current_bandwidth = (
            float(row["bandwidth_mbps"]) if row.get("bandwidth_mbps") else None
        )
        bandwidth_drop = (
            current_bandwidth is not None
            and key in previous_bandwidth
            and current_bandwidth < previous_bandwidth[key]
        )
        if current_bandwidth is not None:
            previous_bandwidth[key] = current_bandwidth
        item.update(
            quality_reward_value=quality,
            rebuffer_penalty_value=rebuffer,
            switching_penalty_value=switching,
            reward_value=reward,
            reward_recomputed=quality + rebuffer + switching,
            undesirable_high_quality=quality >= 3 and rebuffer_seconds > 0 and reward > 0,
            drop_decision=(
                row.get("bandwidth_drop", "").lower() == "true"
                or bandwidth_drop
            ),
        )
        enriched.append(item)
    return enriched


def _episode_groups(rows: list[dict[str, object]]):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["trace_id"], row["policy"])].append(row)
    return groups


def episode_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    results = []
    for (trace_id, policy), values in _episode_groups(rows).items():
        values.sort(key=lambda row: int(row["segment_index"]))
        bitrates = np.asarray([float(row["selected_bitrate_mbps"]) for row in values])
        changes = np.diff(bitrates)
        qoe = float(np.mean(bitrates) - 4 * sum(float(row["rebuffer_seconds"]) for row in values) - 0.5 * np.abs(changes).sum())
        results.append(
            {
                "trace_id": trace_id,
                "profile": values[0]["profile"],
                "policy": policy,
                "accumulated_reward": float(sum(row["reward_value"] for row in values)),
                "quality_reward_total": float(sum(row["quality_reward_value"] for row in values)),
                "rebuffer_penalty_total": float(sum(row["rebuffer_penalty_value"] for row in values)),
                "switching_penalty_total": float(sum(row["switching_penalty_value"] for row in values)),
                "independent_qoe": qoe,
                "average_bitrate_mbps": float(np.mean(bitrates)),
                "rebuffer_seconds": float(sum(float(row["rebuffer_seconds"]) for row in values)),
                "bitrate_variation_mbps": float(np.abs(changes).sum()),
            }
        )
    return results


def summary(rows: list[dict[str, object]]) -> dict[str, object]:
    fields = (
        "accumulated_reward",
        "quality_reward_total",
        "rebuffer_penalty_total",
        "switching_penalty_total",
        "independent_qoe",
        "average_bitrate_mbps",
        "rebuffer_seconds",
        "bitrate_variation_mbps",
    )
    result = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["policy"], row["profile"])].append(row)
    for key, values in sorted(grouped.items()):
        result[f"{key[0]}:{key[1]}"] = {
            field: {
                "mean": float(np.mean([float(row[field]) for row in values])),
                "sd": float(np.std([float(row[field]) for row in values], ddof=1)),
            }
            for field in fields
        }
    return result


def _plot(output: Path, episode_rows: list[dict[str, object]], segment_rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    values = defaultdict(list)
    for row in episode_rows:
        values[row["policy"]].append(row)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for policy, rows in sorted(values.items()):
        axes[0].scatter(
            [float(row["accumulated_reward"]) for row in rows],
            [float(row["independent_qoe"]) for row in rows],
            alpha=0.65,
            label=policy,
        )
        axes[1].scatter(
            [float(row["quality_reward_total"]) for row in rows],
            [float(row["independent_qoe"]) for row in rows],
            alpha=0.65,
            label=policy,
        )
    axes[0].set_xlabel("Accumulated training reward")
    axes[0].set_ylabel("Independent evaluation QoE")
    axes[0].set_title("Reward versus independent QoE")
    axes[1].set_xlabel("Quality reward total")
    axes[1].set_ylabel("Independent evaluation QoE")
    axes[1].set_title("Quality-index contribution versus QoE")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output / "reward_vs_independent_qoe.png", dpi=220)
    plt.close(figure)

    component_fields = (
        ("quality_reward_value", "Quality reward"),
        ("rebuffer_penalty_value", "Rebuffer penalty"),
        ("switching_penalty_value", "Switching penalty"),
        ("reward_value", "Total reward"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for axis, (field, title) in zip(axes.flat, component_fields):
        for policy in sorted({str(row["policy"]) for row in segment_rows}):
            values_for_policy = [float(row[field]) for row in segment_rows if row["policy"] == policy]
            axis.hist(values_for_policy, bins=25, alpha=0.45, label=policy)
        axis.set_title(title)
        axis.set_xlabel("Contribution")
        axis.set_ylabel("Segment count")
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(output / "reward_component_distributions.png", dpi=220)
    plt.close(figure)

    profiles = sorted({str(row["profile"]) for row in segment_rows})
    policies = sorted({str(row["policy"]) for row in segment_rows})
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, field, title in (
        (axes[0], "reward_value", "Mean reward"),
        (axes[1], "rebuffer_penalty_value", "Mean rebuffer penalty"),
        (axes[2], "quality_reward_value", "Mean quality reward"),
    ):
        matrix = np.asarray([
            [
                np.mean([
                    float(row[field])
                    for row in segment_rows
                    if row["profile"] == profile and row["policy"] == policy
                ])
                for policy in policies
            ]
            for profile in profiles
        ])
        image = axis.imshow(matrix, aspect="auto")
        axis.set_xticks(range(len(policies)), policies, rotation=35)
        axis.set_yticks(range(len(profiles)), profiles)
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output / "per_profile_reward_breakdown.png", dpi=220)
    plt.close(figure)


def write_report(output: Path, source: Path, segment_rows, episodes) -> None:
    rewards = [row["accumulated_reward"] for row in episodes]
    qoes = [row["independent_qoe"] for row in episodes]
    correlation = float(np.corrcoef(rewards, qoes)[0, 1])
    high_quality = [row for row in segment_rows if row["undesirable_high_quality"]]
    drop_high_quality = [row for row in high_quality if row["drop_decision"]]
    lines = [
        "# Reward Function Analysis",
        "",
        f"- Source: `{source}`",
        f"- Segment records: {len(segment_rows)}",
        f"- Episode-policy records: {len(episodes)}",
        "",
        "## Reward definition",
        "",
        "The implemented reward is `quality_index - 4 × rebuffer_seconds - 0.5 × "
        "absolute_bitrate_change_mbps`. Quality is the discrete action index (0–4), "
        "not the selected bitrate in Mbps. Evaluation QoE instead uses average bitrate "
        "in Mbps with the same rebuffer and switching coefficients.",
        "",
        "## Overall component means per segment",
        "",
        "| Policy | Quality reward | Rebuffer penalty | Switching penalty | Total reward | Independent QoE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in sorted({str(row["policy"]) for row in episodes}):
        values = [row for row in episodes if row["policy"] == policy]
        mean = lambda field: np.mean([float(row[field]) for row in values]) / 10
        lines.append(
            f"| {policy} | {mean('quality_reward_total'):.3f} | {-mean('rebuffer_penalty_total'):.3f} | "
            f"{-mean('switching_penalty_total'):.3f} | {mean('accumulated_reward'):.3f} | "
            f"{np.mean([float(row['independent_qoe']) for row in values]):.3f} |"
        )
    lines += [
        "",
        f"Across all episode-policy records, accumulated reward and independent QoE correlation was "
        f"`{correlation:.3f}`. This is descriptive and does not establish causality.",
        "",
        "## Potentially undesirable high-quality decisions",
        "",
        f"{len(high_quality)} segments selected quality index ≥ 3, incurred rebuffering, "
        f"and still received positive immediate reward.",
        f"{len(drop_high_quality)} of those occurred on detected bandwidth-drop segments.",
        "",
        "| Trace | Profile | Segment | Bandwidth | Bitrate | Rebuffer (s) | Quality reward | Rebuffer penalty | Switching penalty | Total reward |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(high_quality, key=lambda value: float(value["reward_value"]), reverse=True)[:12]:
        lines.append(
            f"| {row['trace_id']} | {row['profile']} | {row['segment_index']} | "
            f"{float(row['bandwidth_mbps']):.3f} | {float(row['selected_bitrate_mbps']):.3f} | "
            f"{float(row['rebuffer_seconds']):.3f} | {float(row['quality_reward_value']):.1f} | "
            f"{float(row['rebuffer_penalty_value']):.3f} | {float(row['switching_penalty_value']):.3f} | "
            f"{float(row['reward_value']):.3f} |"
        )
    lines += [
        "",
        "## Evidence-backed assessment",
        "",
        "- **Reward magnitude balance:** the quality term is bounded at 0–4 per segment, while rebuffering is unbounded in seconds and multiplied by 4; switching is proportional to Mbps change and multiplied by 0.5. The relative contribution is therefore state-dependent, not uniformly balanced.",
        "- **High bitrate incentive:** the reward directly rewards quality index, so higher actions receive more immediate reward even when their bitrate is not sustainable. Because the quality term is an index rather than Mbps, it does not align linearly with evaluation quality.",
        "- **Evaluation alignment:** shared rebuffer and switching penalties make the objectives partially aligned, but quality-index reward and bitrate-based QoE differ. The correlation and component tables quantify the mismatch.",
        "- **Drops and congestion:** during a drop, an action can receive a positive quality reward while rebuffer penalty is delayed until the download completes. This creates a temporal credit-assignment conflict for high actions; sustained congestion increases the rebuffer term and can dominate quality.",
        "",
        "Reward misalignment is therefore a plausible contributing cause of poor DQN performance, especially for bitrate selection and temporal credit assignment. This analysis does not justify changing coefficients or claim that reward misalignment is the sole cause.",
        "",
        "No reward function, model, configuration, or existing result was modified.",
    ]
    (output / "reward_analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(source: Path, output: Path) -> None:
    segment_rows = enrich(load_rows(source))
    episodes = episode_metrics(segment_rows)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "reward_components.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = list(segment_rows[0])
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(segment_rows)
    with (output / "episode_reward_qoe.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    (output / "per_profile_summary.json").write_text(json.dumps(summary(episodes), indent=2) + "\n", encoding="utf-8")
    _plot(output, episodes, segment_rows)
    write_report(output, source, segment_rows, episodes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("experiments/results/research_multiseed/diagnostics/decisions.csv"))
    parser.add_argument("--output", type=Path, default=Path("experiments/results/research_multiseed/reward_analysis"))
    args = parser.parse_args()
    run(args.source, args.output)
    print(f"wrote reward analysis to {args.output}")


if __name__ == "__main__":
    main()
