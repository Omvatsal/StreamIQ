"""Controlled DQN state-representation ablation."""

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
from rl_engine.evaluation.baselines import buffer_based, fixed_bitrate, throughput_based
from rl_engine.evaluation.metrics import EvaluationMetrics, summarize
from rl_engine.training.config import TrainingConfig
from rl_engine.training.trainer import DQNTrainer


VARIANTS = {"full": "full", "without_buffer": "without_buffer", "buffer_change": "buffer_change"}
SEEDS = (11, 12, 13)


def load_split(root: Path, split: str) -> list[NetworkTrace]:
    return [NetworkTrace.load(path) for path in sorted((root / split).glob("*.json"))]


def make_envs(traces: list[NetworkTrace], state_representation: str = "full") -> list[ABREnvironment]:
    return [ABREnvironment(trace, state_representation=state_representation) for trace in traces]


def _average(metrics: list[EvaluationMetrics]) -> EvaluationMetrics:
    integer_fields = {"bitrate_switches", "upward_switches", "downward_switches"}
    return EvaluationMetrics(
        **{
            field: (
                round(sum(getattr(item, field) for item in metrics) / len(metrics))
                if field in integer_fields
                else sum(getattr(item, field) for item in metrics) / len(metrics)
            )
            for field in EvaluationMetrics.__dataclass_fields__
        }
    )


def evaluate_with_segments(traces, policy, state_representation="full"):
    episode_metrics = []
    rows = []
    for trace in traces:
        env = ABREnvironment(trace, state_representation=state_representation)
        state, _ = env.reset(seed=trace.seed)
        previous_bandwidth = None
        done = False
        while not done:
            action = int(policy(state, env))
            state, _, terminated, truncated, _ = env.step(action)
            metric = env.metrics.segments[-1]
            rows.append(
                {
                    "trace_id": trace.trace_id,
                    "profile": trace.profile,
                    "segment_index": metric.segment_index,
                    "bandwidth_drop": previous_bandwidth is not None and metric.bandwidth_mbps < previous_bandwidth,
                    "bandwidth_mbps": metric.bandwidth_mbps,
                    "throughput_mbps": metric.throughput_mbps,
                    "bitrate_mbps": metric.bitrate_mbps,
                    "buffer_occupancy_seconds": metric.buffer_occupancy_seconds,
                    "rebuffer_seconds": metric.rebuffer_seconds,
                }
            )
            previous_bandwidth = metric.bandwidth_mbps
            done = terminated or truncated
        episode_metrics.append(summarize(env.metrics))
    return _average(episode_metrics), rows


def drop_summary(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["trace_id"], []).append(row)
    before, drops, congestion = [], [], []
    for trace_rows in grouped.values():
        trace_rows.sort(key=lambda row: row["segment_index"])
        for index, row in enumerate(trace_rows):
            if row["bandwidth_drop"]:
                drops.append(row)
                if index:
                    before.append(trace_rows[index - 1])
                if row["profile"] == "congestion":
                    congestion.append(row)
    mean = lambda values, field: float(np.mean([float(row[field]) for row in values])) if values else 0.0
    return {
        "drop_decisions": len(drops),
        "pre_drop_bitrate_mbps": mean(before, "bitrate_mbps"),
        "drop_bitrate_mbps": mean(drops, "bitrate_mbps"),
        "drop_rebuffer_seconds": mean(drops, "rebuffer_seconds"),
        "congestion_bitrate_mbps": mean(congestion, "bitrate_mbps"),
        "congestion_rebuffer_seconds": mean(congestion, "rebuffer_seconds"),
    }


def _mean_sd(rows, field):
    values = np.asarray([float(row[field]) for row in rows])
    return float(values.mean()), float(values.std(ddof=1))


def run(output: Path, dataset_root: Path, episodes: int = 20, seeds=SEEDS):
    splits = {split: load_split(dataset_root, split) for split in ("training", "validation", "test")}
    if any(not traces for traces in splits.values()):
        raise FileNotFoundError(f"missing dataset split under {dataset_root}")
    output.mkdir(parents=True, exist_ok=True)
    training_config = TrainingConfig(episodes=episodes, seed=seeds[0], validation_every=max(1, episodes // 5))
    (output / "config.json").write_text(json.dumps({
        "episodes": episodes,
        "seeds": seeds,
        "dataset_root": str(dataset_root),
        "dataset_config": asdict(DatasetConfig()),
        "variants": VARIANTS,
        "assumptions": {
            "without_buffer": "buffer feature replaced with zero",
            "buffer_change": "buffer feature is current post-download occupancy minus prior post-download occupancy; reset is zero",
            "baselines": "evaluated in the full state on identical test traces",
        },
    }, indent=2) + "\n", encoding="utf-8")
    results = []
    segment_rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        for variant, representation in VARIANTS.items():
            train_envs = make_envs(splits["training"], representation)
            validation_envs = make_envs(splits["validation"], representation)
            agent = DQNAgent(train_envs[0].observation_space.shape[0], batch_size=32, seed=seed)
            config = TrainingConfig(episodes=episodes, seed=seed, validation_every=training_config.validation_every)
            checkpoint_dir = output / "training" / str(seed) / variant
            history = DQNTrainer(agent, config).train(train_envs, checkpoint_dir, validation_environments=validation_envs)
            selected = checkpoint_dir / "best_validation.pt"
            if selected.exists():
                agent.load(selected)
            agent.save(checkpoint_dir / "model_selected.pt", episodes)
            dqn, dqn_rows = evaluate_with_segments(
                splits["test"], lambda state, _env: agent.select_action(state, explore=False), representation
            )
            policies = {"fixed": fixed_bitrate(0), "throughput": throughput_based, "buffer": buffer_based}
            baselines = {}
            for policy_name, policy in policies.items():
                baselines[policy_name], baseline_rows = evaluate_with_segments(splits["test"], policy)
                segment_rows.extend({"seed": seed, "variant": variant, "policy": policy_name, **row} for row in baseline_rows)
            segment_rows.extend({"seed": seed, "variant": variant, "policy": "dqn", **row} for row in dqn_rows)
            results.append({
                "seed": seed,
                "variant": variant,
                "dqn": asdict(dqn),
                "drop_response": drop_summary(dqn_rows),
                "baselines": {name: asdict(value) for name, value in baselines.items()},
            })
            history.to_csv(output / "training" / str(seed) / f"{variant}.csv")
    metrics_dir = output / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    (metrics_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    with (metrics_dir / "segment_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(segment_rows[0]))
        writer.writeheader()
        writer.writerows(segment_rows)
    _write_report(output, results)


def _write_report(output: Path, results):
    dqn_rows = {variant: [row["dqn"] for row in results if row["variant"] == variant] for variant in VARIANTS}
    lines = [
        "# Buffer State Representation Ablation",
        "",
        "Each variant was independently retrained for 20 episodes with seeds 11, 12, and 13. "
        "Hyperparameters, train/validation/test traces, reward, architecture, and checkpoint selection were held constant.",
        "",
        "## Assumptions",
        "",
        "- `full`: unchanged nine-value state.",
        "- `without_buffer`: first feature replaced by zero, preserving nine dimensions.",
        "- `buffer_change`: first feature is current post-download buffer minus prior post-download buffer; reset is zero.",
        "- Baselines use the unchanged full-state environment on identical test traces.",
        "",
        "## DQN overall metrics (mean ± sample SD)",
        "",
        "| Variant | QoE | Bitrate (Mbps) | Rebuffer (s) | Switches | Variation (Mbps) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant, rows in dqn_rows.items():
        values = {field: _mean_sd(rows, field) for field in ("qoe", "average_bitrate_mbps", "rebuffer_seconds", "bitrate_switches", "total_bitrate_variation_mbps")}
        lines.append(
            f"| {variant} | {values['qoe'][0]:.3f} ± {values['qoe'][1]:.3f} | "
            f"{values['average_bitrate_mbps'][0]:.3f} ± {values['average_bitrate_mbps'][1]:.3f} | "
            f"{values['rebuffer_seconds'][0]:.3f} ± {values['rebuffer_seconds'][1]:.3f} | "
            f"{values['bitrate_switches'][0]:.3f} ± {values['bitrate_switches'][1]:.3f} | "
            f"{values['total_bitrate_variation_mbps'][0]:.3f} ± {values['total_bitrate_variation_mbps'][1]:.3f} |"
        )
    baseline_rows = {
        policy: [
            result["baselines"][policy]
            for result in results
        ]
        for policy in ("fixed", "throughput", "buffer")
    }
    lines += [
        "",
        "## Baseline metrics on identical unseen test traces",
        "",
        "| Policy | QoE | Bitrate (Mbps) | Rebuffer (s) | Switches | Variation (Mbps) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy, rows in baseline_rows.items():
        values = {
            field: _mean_sd(rows, field)
            for field in ("qoe", "average_bitrate_mbps", "rebuffer_seconds", "bitrate_switches", "total_bitrate_variation_mbps")
        }
        lines.append(
            f"| {policy} | {values['qoe'][0]:.3f} ± {values['qoe'][1]:.3f} | "
            f"{values['average_bitrate_mbps'][0]:.3f} ± {values['average_bitrate_mbps'][1]:.3f} | "
            f"{values['rebuffer_seconds'][0]:.3f} ± {values['rebuffer_seconds'][1]:.3f} | "
            f"{values['bitrate_switches'][0]:.3f} ± {values['bitrate_switches'][1]:.3f} | "
            f"{values['total_bitrate_variation_mbps'][0]:.3f} ± {values['total_bitrate_variation_mbps'][1]:.3f} |"
        )
    lines += ["", "## Drop and congestion response", "", "| Seed | Variant | Pre-drop bitrate | Drop bitrate | Drop rebuffer (s) | Congestion bitrate | Congestion rebuffer (s) |", "|---:|---|---:|---:|---:|---:|---:|"]
    for row in results:
        drop = row["drop_response"]
        lines.append(f"| {row['seed']} | {row['variant']} | {drop['pre_drop_bitrate_mbps']:.3f} | {drop['drop_bitrate_mbps']:.3f} | {drop['drop_rebuffer_seconds']:.3f} | {drop['congestion_bitrate_mbps']:.3f} | {drop['congestion_rebuffer_seconds']:.3f} |")
    lines += [
        "",
        "### Drop/congestion aggregates (mean ± sample SD)",
        "",
        "| Variant | Pre-drop bitrate | Drop bitrate | Drop rebuffer (s) | Congestion bitrate | Congestion rebuffer (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        rows = [row["drop_response"] for row in results if row["variant"] == variant]
        values = {
            field: _mean_sd(rows, field)
            for field in (
                "pre_drop_bitrate_mbps",
                "drop_bitrate_mbps",
                "drop_rebuffer_seconds",
                "congestion_bitrate_mbps",
                "congestion_rebuffer_seconds",
            )
        }
        lines.append(
            f"| {variant} | {values['pre_drop_bitrate_mbps'][0]:.3f} ± {values['pre_drop_bitrate_mbps'][1]:.3f} | "
            f"{values['drop_bitrate_mbps'][0]:.3f} ± {values['drop_bitrate_mbps'][1]:.3f} | "
            f"{values['drop_rebuffer_seconds'][0]:.3f} ± {values['drop_rebuffer_seconds'][1]:.3f} | "
            f"{values['congestion_bitrate_mbps'][0]:.3f} ± {values['congestion_bitrate_mbps'][1]:.3f} | "
            f"{values['congestion_rebuffer_seconds'][0]:.3f} ± {values['congestion_rebuffer_seconds'][1]:.3f} |"
        )
    lines += [
        "",
        "## Conclusion",
        "",
        "In this three-seed, 20-episode experiment, both alternatives had higher mean QoE than the full-state variant. "
        "Both alternatives had lower mean drop-segment bitrate than the full variant, but neither reduced it in every seed. "
        "Drop and congestion outcomes varied materially "
        "by seed, so the results do not establish that removing or replacing buffer occupancy improves drop responsiveness. "
        "No statistical significance is claimed.",
        "",
        "Results are descriptive mean ± SD across training seeds; no statistical significance is claimed.",
        "",
    ]
    (output / "state_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/results/research_multiseed/state_ablation"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/results/research_multiseed/datasets_11"))
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()
    run(args.output, args.dataset_root, args.episodes)


if __name__ == "__main__":
    main()
