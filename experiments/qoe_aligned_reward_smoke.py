"""Small identical-trace smoke comparison for the two reward objectives."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from network_lab.traces.generator import generate_trace
from network_lab.traces.models import TraceConfig
from rl_engine.environment.abr_env import ABREnvironment
from rl_engine.environment.reward import RewardConfig
from rl_engine.evaluation.metrics import summarize
from rl_engine.environment.video import VideoModel


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    episode_rows = []
    actions = (0, 2, 4, 3, 1, 4, 2, 3, 4, 1)
    for profile, seed in (("stable", 301), ("sudden_drop", 304), ("congestion", 306)):
        trace = generate_trace(profile, f"qoe_smoke_{profile}_{seed}", TraceConfig(60, 1, seed=seed))
        environments = {
            objective: ABREnvironment(
                trace,
                video=VideoModel(total_segments=10),
                reward_config=RewardConfig(objective=objective),
                state_representation="without_buffer",
            )
            for objective in ("legacy", "qoe_aligned")
        }
        for objective, environment in environments.items():
            environment.reset(seed=seed)
            total_reward = 0.0
            for segment_index, action in enumerate(actions):
                _, reward, terminated, _, _ = environment.step(action)
                metric = environment.metrics.segments[-1]
                total_reward += reward
                rows.append({
                    "profile": profile,
                    "seed": seed,
                    "objective": objective,
                    "segment_index": segment_index,
                    "action": action,
                    "bitrate_mbps": metric.bitrate_mbps,
                    "rebuffer_seconds": metric.rebuffer_seconds,
                    "switch_mbps": (
                        0.0
                        if segment_index == 0
                        else abs(metric.bitrate_mbps - environment.metrics.segments[-2].bitrate_mbps)
                    ),
                    "segment_reward": reward,
                })
                if terminated:
                    break
            evaluation = summarize(environment.metrics)
            undiscounted_difference = total_reward - 10.0 * evaluation.qoe
            episode_rows.append({
                "profile": profile,
                "seed": seed,
                "objective": objective,
                "segment_reward_sum": total_reward,
                "evaluation_qoe": evaluation.qoe,
                "ten_times_evaluation_qoe": 10.0 * evaluation.qoe,
                "sum_minus_ten_qoe": undiscounted_difference,
                "average_bitrate_mbps": evaluation.average_bitrate_mbps,
                "rebuffer_seconds": evaluation.rebuffer_seconds,
                "total_bitrate_variation_mbps": evaluation.total_bitrate_variation_mbps,
            })
    with (output / "segment_smoke_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "episode_smoke_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    (output / "smoke_summary.json").write_text(json.dumps({
        "profiles": ["stable", "sudden_drop", "congestion"],
        "actions": actions,
        "objectives": ["legacy", "qoe_aligned"],
        "episodes": len(episode_rows),
        "aggregation_identity": (
            "The requested sum == 10 * evaluation QoE identity does not hold for the existing "
            "evaluation implementation because it averages bitrate but sums penalties."
        ),
        "max_absolute_sum_minus_ten_qoe": max(abs(float(row["sum_minus_ten_qoe"])) for row in episode_rows),
    }, indent=2) + "\n", encoding="utf-8")
    (output / "smoke_report.md").write_text(
        "# QoE-Aligned Reward Smoke Test\n\n"
        "Both objectives used identical traces, state representation, and action sequences. "
        "The legacy objective remains the default. The qoe_aligned objective uses "
        "`bitrate_mbps - 4 * rebuffer_seconds - 0.5 * switch_mbps`.\n\n"
        "The requested identity that the undiscounted reward sum equals ten times existing "
        "episode QoE is inconsistent with `evaluation.metrics.summarize`: evaluation QoE "
        "uses average bitrate but does not average the penalty sums. For ten segments, the "
        "new reward sum differs from ten times QoE by `36 * total_rebuffer_seconds + "
        "4.5 * total_bitrate_variation_mbps`. The CSV records the exact discrepancy.\n\n"
        "A discounted DQN return with gamma < 1 is additionally not equal to either "
        "undiscounted aggregate.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/research_multiseed/qoe_aligned_reward"),
    )
    args = parser.parse_args()
    run(args.output)
    print(f"wrote smoke results to {args.output}")


if __name__ == "__main__":
    main()
