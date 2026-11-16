"""Controlled multi-seed comparison of the three StreamIQ reward objectives."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from experiments.configurations.datasets import DatasetConfig
from network_lab.traces.models import NetworkTrace
from rl_engine.agents.dqn import DQNAgent
from rl_engine.environment.abr_env import ABREnvironment
from rl_engine.environment.reward import RewardConfig
from rl_engine.evaluation.baselines import buffer_based, fixed_bitrate, throughput_based
from rl_engine.evaluation.metrics import summarize
from rl_engine.evaluation.statistics import paired_statistics
from rl_engine.training.config import TrainingConfig
from rl_engine.training.trainer import DQNTrainer


OBJECTIVES = ("legacy", "qoe_aligned", "evaluation_aligned")
SEEDS = (11, 12, 13)
EPISODES = 20


def load_split(root: Path, split: str) -> list[NetworkTrace]:
    return [NetworkTrace.load(path) for path in sorted((root / split).glob("*.json"))]


def make_envs(traces: list[NetworkTrace], objective: str) -> list[ABREnvironment]:
    return [
        ABREnvironment(
            trace,
            reward_config=RewardConfig(objective=objective),
            state_representation="without_buffer",
        )
        for trace in traces
    ]


def evaluate_trace(trace: NetworkTrace, agent: DQNAgent, objective: str) -> tuple[dict[str, object], list[dict[str, object]]]:
    environment = ABREnvironment(
        trace,
        reward_config=RewardConfig(objective=objective),
        state_representation="without_buffer",
    )
    state, _ = environment.reset(seed=trace.seed)
    rows = []
    previous_bandwidth = None
    done = False
    while not done:
        action = agent.select_action(state, explore=False)
        state, _, terminated, truncated, _ = environment.step(action)
        metric = environment.metrics.segments[-1]
        rows.append({
            "trace_id": trace.trace_id,
            "profile": trace.profile,
            "segment_index": metric.segment_index,
            "action": action,
            "bandwidth_drop": previous_bandwidth is not None and metric.bandwidth_mbps < previous_bandwidth,
            "bandwidth_mbps": metric.bandwidth_mbps,
            "throughput_mbps": metric.throughput_mbps,
            "bitrate_mbps": metric.bitrate_mbps,
            "buffer_occupancy_seconds": metric.buffer_occupancy_seconds,
            "rebuffer_seconds": metric.rebuffer_seconds,
            "reward": metric.reward,
        })
        previous_bandwidth = metric.bandwidth_mbps
        done = terminated or truncated
    metrics = summarize(environment.metrics)
    result = {
        "trace_id": trace.trace_id,
        "profile": trace.profile,
        "objective": objective,
        "accumulated_reward": environment.metrics.accumulated_reward,
        **asdict(metrics),
    }
    return result, rows


def evaluate_baseline(trace: NetworkTrace, policy_name: str) -> dict[str, object]:
    environment = ABREnvironment(
        trace,
        reward_config=RewardConfig(),
        state_representation="full",
    )
    state, _ = environment.reset(seed=trace.seed)
    done = False
    policy = {
        "fixed": fixed_bitrate(0),
        "throughput": throughput_based,
        "buffer": buffer_based,
    }[policy_name]
    while not done:
        action = policy(state, environment)
        state, _, terminated, truncated, _ = environment.step(action)
        done = terminated or truncated
    return {"trace_id": trace.trace_id, "profile": trace.profile, "policy": policy_name, **asdict(summarize(environment.metrics))}


def _paired(first: list[float], second: list[float]) -> dict[str, object]:
    return asdict(paired_statistics(first, second))


def _summary(values: list[float]) -> dict[str, float]:
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    try:
        from scipy.stats import t
        critical = float(t.ppf(0.975, len(values) - 1))
    except (ImportError, ValueError):
        critical = 1.96
    margin = critical * sd / np.sqrt(len(values))
    return {"mean": mean, "sd": sd, "ci_low": mean - margin, "ci_high": mean + margin}


def _holm(rows: list[dict[str, object]]) -> None:
    ordered = sorted(
        [row for row in rows if row["p_value"] is not None],
        key=lambda row: float(row["p_value"]),
    )
    count = len(ordered)
    previous = 0.0
    for index, row in enumerate(ordered):
        adjusted = min(1.0, max(previous, float(row["p_value"]) * (count - index)))
        row["p_value_holm"] = adjusted
        previous = adjusted


def _plot(output: Path, histories: list[dict[str, object]], validations: list[dict[str, object]], profile_rows: list[dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for field, title, filename in (
        ("reward", "Training reward", "training_reward.png"),
        ("loss", "Training loss", "training_loss.png"),
        ("epsilon", "Epsilon", "training_epsilon.png"),
    ):
        figure, axis = plt.subplots(figsize=(10, 5))
        for objective in OBJECTIVES:
            values = [row for row in histories if row["objective"] == objective]
            grouped = {}
            for row in values:
                grouped.setdefault(int(row["episode"]), []).append(float(row[field]))
            x = sorted(grouped)
            mean = np.asarray([np.mean(grouped[index]) for index in x])
            sd = np.asarray([np.std(grouped[index], ddof=1) for index in x])
            axis.plot(x, mean, label=objective)
            axis.fill_between(x, mean - sd, mean + sd, alpha=0.15)
        axis.set(title=title, xlabel="Episode", ylabel=field)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=220)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    for objective in OBJECTIVES:
        values = [row for row in validations if row["objective"] == objective]
        axis.plot(
            [int(row["episode"]) for row in values],
            [float(row["score"]) for row in values],
            marker="o",
            label=objective,
        )
    axis.set(title="Validation score progression", xlabel="Episode", ylabel="Validation score")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "validation_score.png", dpi=220)
    plt.close(figure)

    profiles = sorted({str(row["profile"]) for row in profile_rows})
    figure, axis = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(OBJECTIVES)
    x = np.arange(len(profiles))
    for index, objective in enumerate(OBJECTIVES):
        values = [
            np.mean([
                float(row["qoe"])
                for row in profile_rows
                if row["objective"] == objective and row["profile"] == profile
            ])
            for profile in profiles
        ]
        axis.bar(x + index * width, values, width, label=objective)
    axis.set(title="Test QoE by network profile", xlabel="Profile", ylabel="Independent QoE")
    axis.set_xticks(x + width, profiles, rotation=25)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "test_qoe_by_profile.png", dpi=220)
    plt.close(figure)


def run(output: Path, dataset_root: Path) -> None:
    splits = {split: load_split(dataset_root, split) for split in ("training", "validation", "test")}
    if any(not traces for traces in splits.values()):
        raise FileNotFoundError(f"missing dataset split under {dataset_root}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps({
        "objectives": OBJECTIVES,
        "seeds": SEEDS,
        "episodes": EPISODES,
        "state_representation": "without_buffer",
        "dataset_root": str(dataset_root),
        "dataset_config": asdict(DatasetConfig()),
        "selection": "best_validation.pt selected by existing validation score; test is never used for selection",
        "unchanged": ["DQN architecture", "optimizer", "learning rate", "gamma", "exploration schedule", "replay buffer", "target update", "evaluation QoE"],
    }, indent=2) + "\n", encoding="utf-8")

    histories = []
    validations = []
    test_trace_rows = []
    segment_rows = []
    run_records = []
    for objective in OBJECTIVES:
        for seed in SEEDS:
            torch.manual_seed(seed)
            train_envs = make_envs(splits["training"], objective)
            validation_envs = make_envs(splits["validation"], objective)
            agent = DQNAgent(train_envs[0].observation_space.shape[0], batch_size=32, seed=seed)
            training_config = TrainingConfig(
                episodes=EPISODES,
                seed=seed,
                validation_every=4,
            )
            checkpoint_dir = output / "training" / objective / str(seed)
            history = DQNTrainer(agent, training_config).train(
                train_envs,
                checkpoint_dir,
                validation_environments=validation_envs,
            )
            selected = checkpoint_dir / "best_validation.pt"
            selected_episode = None
            if selected.exists():
                checkpoint = torch.load(selected, map_location="cpu", weights_only=False)
                selected_episode = int(checkpoint.get("episode", 0))
                agent.load(selected)
            agent.save(checkpoint_dir / "model_selected.pt", EPISODES)
            for episode, reward in enumerate(history.rewards):
                histories.append({
                    "objective": objective,
                    "seed": seed,
                    "episode": episode,
                    "reward": reward,
                    "loss": history.losses[episode],
                    "epsilon": history.epsilons[episode],
                })
            for index, score in enumerate(history.validation_scores):
                validations.append({
                    "objective": objective,
                    "seed": seed,
                    "episode": (index + 1) * training_config.validation_every,
                    "score": score,
                })
            test_results = []
            for trace in splits["test"]:
                result, rows = evaluate_trace(trace, agent, objective)
                test_results.append(result)
                test_trace_rows.append({"objective": objective, "seed": seed, **result})
                segment_rows.extend({"objective": objective, "seed": seed, **row} for row in rows)
            validation_scores = [row["score"] for row in validations if row["objective"] == objective and row["seed"] == seed]
            run_records.append({
                "objective": objective,
                "seed": seed,
                "selected_checkpoint": str(selected),
                "selected_checkpoint_episode": selected_episode,
                "validation_scores": validation_scores,
                "test_mean": {
                    field: float(np.mean([float(row[field]) for row in test_results]))
                    for field in ("qoe", "average_bitrate_mbps", "rebuffer_seconds", "rebuffer_ratio", "bitrate_switches", "average_bitrate_change_mbps", "total_bitrate_variation_mbps", "average_download_duration_seconds")
                },
            })

    baselines = []
    for policy in ("fixed", "throughput", "buffer"):
        for trace in splits["test"]:
            baselines.append(evaluate_baseline(trace, policy))
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    (metrics_dir / "run_records.json").write_text(json.dumps(run_records, indent=2) + "\n", encoding="utf-8")
    (metrics_dir / "baselines_per_trace.json").write_text(json.dumps(baselines, indent=2) + "\n", encoding="utf-8")
    for name, rows in (("training_history.csv", histories), ("validation_curves.csv", validations), ("test_per_trace.csv", test_trace_rows), ("test_segments.csv", segment_rows)):
        with (metrics_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    metric_names = ("qoe", "average_bitrate_mbps", "rebuffer_seconds", "rebuffer_ratio", "bitrate_switches", "average_bitrate_change_mbps", "total_bitrate_variation_mbps", "average_download_duration_seconds")
    for objective in OBJECTIVES:
        values = [record["test_mean"] for record in run_records if record["objective"] == objective]
        summary[objective] = {
            metric: {
                **_summary([row[metric] for row in values]),
            }
            for metric in metric_names
        }
    (metrics_dir / "summary_mean_sd.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    profile_summary = {}
    for objective in OBJECTIVES:
        profile_summary[objective] = {}
        for profile in sorted({row["profile"] for row in test_trace_rows}):
            profile_summary[objective][profile] = {}
            rows = [
                row for row in test_trace_rows
                if row["objective"] == objective and row["profile"] == profile
            ]
            for metric in metric_names:
                profile_summary[objective][profile][metric] = _summary([float(row[metric]) for row in rows])
    (metrics_dir / "profile_summary.json").write_text(json.dumps(profile_summary, indent=2) + "\n", encoding="utf-8")
    baseline_summary = {}
    for policy in ("fixed", "throughput", "buffer"):
        rows = [row for row in baselines if row["policy"] == policy]
        baseline_summary[policy] = {
            metric: _summary([float(row[metric]) for row in rows])
            for metric in metric_names
        }
    (metrics_dir / "baseline_summary.json").write_text(json.dumps(baseline_summary, indent=2) + "\n", encoding="utf-8")

    paired_rows = []
    for first_index, first in enumerate(OBJECTIVES):
        for second in OBJECTIVES[first_index + 1:]:
            first_records = {record["seed"]: record for record in run_records if record["objective"] == first}
            second_records = {record["seed"]: record for record in run_records if record["objective"] == second}
            for metric in metric_names:
                stats = _paired(
                    [first_records[seed]["test_mean"][metric] for seed in SEEDS],
                    [second_records[seed]["test_mean"][metric] for seed in SEEDS],
                )
                paired_rows.append({"unit": "training_seed", "first": first, "second": second, "metric": metric, **stats})
            first_trace = {(record["seed"], record["trace_id"]): record for record in test_trace_rows if record["objective"] == first}
            second_trace = {(record["seed"], record["trace_id"]): record for record in test_trace_rows if record["objective"] == second}
            for metric in ("qoe", "rebuffer_seconds", "average_bitrate_mbps"):
                stats = _paired(
                    [first_trace[key][metric] for key in sorted(first_trace)],
                    [second_trace[key][metric] for key in sorted(first_trace)],
                )
                paired_rows.append({"unit": "trace_within_seed", "first": first, "second": second, "metric": metric, **stats})
    _holm(paired_rows)
    with (metrics_dir / "paired_comparisons.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    profile_rows = test_trace_rows
    _plot(output, histories, validations, profile_rows)
    _write_report(output, summary, run_records, baselines, paired_rows)


def _write_report(output, summary, records, baselines, paired_rows):
    lines = [
        "# Reward Objective Comparison",
        "",
        "Three independently trained objectives used the unchanged without-buffer state, DQN architecture, hyperparameters, 20 episodes, seeds 11/12/13, and existing train/validation/test splits. Checkpoints were selected using validation only; test traces were evaluated once after selection.",
        "",
        "## Test metrics (mean ± sample SD across training seeds)",
        "",
        "| Objective | QoE (mean ± SD; 95% CI) | Bitrate (Mbps) | Rebuffer (s) | Rebuffer ratio | Switches | Variation (Mbps) | Download (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for objective in OBJECTIVES:
        value = summary[objective]
        lines.append(
            f"| {objective} | {value['qoe']['mean']:.3f} ± {value['qoe']['sd']:.3f}; "
            f"[{value['qoe']['ci_low']:.3f}, {value['qoe']['ci_high']:.3f}] | "
            f"{value['average_bitrate_mbps']['mean']:.3f} ± {value['average_bitrate_mbps']['sd']:.3f} | "
            f"{value['rebuffer_seconds']['mean']:.3f} ± {value['rebuffer_seconds']['sd']:.3f} | "
            f"{value['rebuffer_ratio']['mean']:.3f} ± {value['rebuffer_ratio']['sd']:.3f} | "
            f"{value['bitrate_switches']['mean']:.3f} ± {value['bitrate_switches']['sd']:.3f} | "
            f"{value['total_bitrate_variation_mbps']['mean']:.3f} ± {value['total_bitrate_variation_mbps']['sd']:.3f} | "
            f"{value['average_download_duration_seconds']['mean']:.3f} ± {value['average_download_duration_seconds']['sd']:.3f} |"
        )
    lines += ["", "## Checkpoint selection", "", "| Objective | Seed | Selected checkpoint episode | Validation scores |", "|---|---:|---:|---|"]
    for record in records:
        lines.append(f"| {record['objective']} | {record['seed']} | {record['selected_checkpoint_episode']} | {', '.join(f'{score:.3f}' for score in record['validation_scores'])} |")
    lines += ["", "## Baseline context", "", "Baselines were evaluated on the same 36 unseen traces. Their full aggregate metrics are in `metrics/baseline_summary.json`; per-trace metrics are in `metrics/baselines_per_trace.json`; the evaluation QoE implementation was unchanged.", "", "| Policy | QoE (mean ± SD) | Bitrate (Mbps) | Rebuffer (s) |", "|---|---:|---:|---:|"]
    for policy in ("fixed", "throughput", "buffer"):
        values = [row for row in baselines if row["policy"] == policy]
        lines.append(
            f"| {policy} | {np.mean([row['qoe'] for row in values]):.3f} ± {np.std([row['qoe'] for row in values], ddof=1):.3f} | "
            f"{np.mean([row['average_bitrate_mbps'] for row in values]):.3f} | "
            f"{np.mean([row['rebuffer_seconds'] for row in values]):.3f} |"
        )
    lines += ["", "## Statistical analysis", "", "Seed-level paired comparisons use three independent training seeds. Trace-within-seed comparisons use identical traces but are not independent training replications. Holm-adjusted p-values are included in `metrics/paired_comparisons.csv`; no significance claim is made from trace-level rows.", "", "## Research questions", ""]
    best = max(OBJECTIVES, key=lambda objective: summary[objective]["qoe"]["mean"])
    lines += [
        f"- Best mean test QoE: `{best}`. This is descriptive; the three-seed confidence is limited.",
        "- Reward alignment should be judged against both QoE and bitrate/rebuffer trade-offs; higher reward alone is not evidence.",
        "- `qoe_aligned` and `evaluation_aligned` differ in penalty scale because the latter exactly matches the existing ten-segment QoE aggregation.",
        "- Improvements are not considered consistent unless the per-seed records and profile results support the same direction.",
        "- No objective or seed was discarded, and no test result was used for selection.",
        "",
        "## Diagnostics and limitations",
        "",
        "Training histories record reward, loss, and epsilon; validation curves record every validation checkpoint. A decreasing loss is not treated as convergence. Only three seeds and 20 episodes were run, so uncertainty is substantial.",
    ]
    (output / "reward_objective_comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/results/research_multiseed/reward_objective_comparison"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/results/research_multiseed/datasets_11"))
    args = parser.parse_args()
    run(args.output, args.dataset_root)
    print(f"wrote reward-objective comparison to {args.output}")


if __name__ == "__main__":
    main()
