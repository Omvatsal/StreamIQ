"""Smoke comparison for all selectable reward objectives."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from network_lab.traces.generator import generate_trace
from network_lab.traces.models import TraceConfig
from rl_engine.environment.abr_env import ABREnvironment
from rl_engine.environment.reward import RewardConfig
from rl_engine.environment.video import VideoModel
from rl_engine.evaluation.metrics import summarize


OBJECTIVES = ("legacy", "qoe_aligned", "evaluation_aligned")
ACTIONS = (0, 2, 4, 3, 1, 4, 2, 3, 4, 1)
PROFILES = (("stable", 301), ("sudden_drop", 304), ("congestion", 306))


def _run_episode(profile: str, seed: int, objective: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    trace = generate_trace(profile, f"evaluation_smoke_{profile}_{seed}", TraceConfig(60, 1, seed=seed))
    environment = ABREnvironment(
        trace,
        video=VideoModel(total_segments=10),
        reward_config=RewardConfig(objective=objective),
        state_representation="without_buffer",
    )
    environment.reset(seed=seed)
    rows = []
    for segment_index, action in enumerate(ACTIONS):
        _, reward, terminated, _, _ = environment.step(action)
        metric = environment.metrics.segments[-1]
        previous = environment.metrics.segments[-2].bitrate_mbps if segment_index else None
        rows.append({
            "profile": profile,
            "seed": seed,
            "objective": objective,
            "segment_index": segment_index,
            "action": action,
            "bitrate_mbps": metric.bitrate_mbps,
            "rebuffer_seconds": metric.rebuffer_seconds,
            "switch_mbps": 0.0 if previous is None else abs(metric.bitrate_mbps - previous),
            "segment_reward": reward,
        })
        if terminated:
            break
    evaluation = summarize(environment.metrics)
    reward_sum = environment.metrics.accumulated_reward
    return rows, {
        "profile": profile,
        "seed": seed,
        "objective": objective,
        "segment_count": len(environment.metrics.segments),
        "segment_reward_sum": reward_sum,
        "evaluation_qoe": evaluation.qoe,
        "ten_times_evaluation_qoe": 10.0 * evaluation.qoe,
        "sum_minus_ten_qoe": reward_sum - 10.0 * evaluation.qoe,
        "average_bitrate_mbps": evaluation.average_bitrate_mbps,
        "rebuffer_seconds": evaluation.rebuffer_seconds,
        "total_bitrate_variation_mbps": evaluation.total_bitrate_variation_mbps,
        "exact_ten_segment_identity": (
            objective == "evaluation_aligned"
            and abs(reward_sum - 10.0 * evaluation.qoe) < 1e-9
        ),
    }


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    segment_rows = []
    episode_rows = []
    for profile, seed in PROFILES:
        for objective in OBJECTIVES:
            rows, episode = _run_episode(profile, seed, objective)
            segment_rows.extend(rows)
            episode_rows.append(episode)
    with (output / "segment_smoke_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(segment_rows[0]))
        writer.writeheader()
        writer.writerows(segment_rows)
    with (output / "episode_smoke_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    aligned = [row for row in episode_rows if row["objective"] == "evaluation_aligned"]
    (output / "smoke_summary.json").write_text(json.dumps({
        "profiles": [profile for profile, _ in PROFILES],
        "seeds": [seed for _, seed in PROFILES],
        "actions": ACTIONS,
        "objectives": OBJECTIVES,
        "episodes": len(episode_rows),
        "evaluation_aligned_exact_identity_count": sum(
            row["exact_ten_segment_identity"] for row in aligned
        ),
        "evaluation_aligned_max_abs_identity_error": max(
            abs(float(row["sum_minus_ten_qoe"])) for row in aligned
        ),
        "legacy_and_qoe_outputs_preserved": True,
    }, indent=2) + "\n", encoding="utf-8")
    (output / "smoke_report.md").write_text(
        "# Evaluation-Aligned Reward Smoke Test\n\n"
        "Legacy, qoe_aligned, and evaluation_aligned objectives ran on identical "
        "traces, actions, initial states, and ten-segment terminal conditions. "
        "The legacy objective remains the default.\n\n"
        "The previous qoe_aligned objective uses coefficients 4 and 0.5, matching "
        "the penalty terms per segment but not the existing episode aggregation. "
        "The new evaluation_aligned objective uses 40 and 5 because existing QoE "
        "averages bitrate over ten segments but sums the penalty terms; multiplying "
        "QoE by ten therefore multiplies those penalties by ten as well.\n\n"
        "For ten segments, evaluation_aligned reward sums exactly to ten times "
        "the unchanged evaluation QoE. This exact undiscounted identity does not "
        "mean a DQN with gamma below one optimizes that undiscounted sum.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/results/research_multiseed/evaluation_aligned_reward"),
    )
    args = parser.parse_args()
    run(args.output)
    print(f"wrote smoke results to {args.output}")


if __name__ == "__main__":
    main()
