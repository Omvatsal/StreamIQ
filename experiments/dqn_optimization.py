"""Validation-only DQN hyperparameter optimization for the without-buffer state."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
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


SEEDS = (11, 12, 13)
STATE_REPRESENTATION = "without_buffer"


@dataclass(frozen=True)
class OptimizationConfig:
    name: str
    learning_rate: float
    gamma: float
    epsilon_decay: float
    epsilon_end: float
    target_update_frequency: int
    replay_capacity: int
    batch_size: int
    episodes: int = 60


CONFIGS = (
    OptimizationConfig("baseline_20ep", 1e-3, 0.99, 0.995, 0.05, 100, 100_000, 32, 20),
    OptimizationConfig("lower_lr_slow_decay", 5e-4, 0.99, 0.99, 0.10, 50, 50_000, 32),
    OptimizationConfig("discounted_stable", 3e-4, 0.95, 0.99, 0.10, 100, 50_000, 32),
    OptimizationConfig("larger_batch_slow_target", 3e-4, 0.99, 0.995, 0.05, 200, 100_000, 64),
)


def load_split(root: Path, split: str) -> list[NetworkTrace]:
    return [NetworkTrace.load(path) for path in sorted((root / split).glob("*.json"))]


def make_envs(traces: list[NetworkTrace], representation: str = STATE_REPRESENTATION):
    return [ABREnvironment(trace, state_representation=representation) for trace in traces]


def mean_metrics(metrics: list[EvaluationMetrics]) -> EvaluationMetrics:
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


def evaluate(traces, policy, representation=STATE_REPRESENTATION, with_profiles=False):
    per_trace = []
    profiles = {}
    for trace in traces:
        env = ABREnvironment(trace, state_representation=representation)
        state, _ = env.reset(seed=trace.seed)
        done = False
        while not done:
            state, _, terminated, truncated, _ = env.step(policy(state, env))
            done = terminated or truncated
        metric = summarize(env.metrics)
        per_trace.append(metric)
        profiles.setdefault(trace.profile, []).append(metric)
    result = mean_metrics(per_trace)
    if not with_profiles:
        return result, {}
    return result, {profile: asdict(mean_metrics(values)) for profile, values in profiles.items()}


def q_probe(agent: DQNAgent, environments: list[ABREnvironment]) -> dict[str, float]:
    import torch

    q_values = []
    margins = []
    for env in environments:
        state, _ = env.reset(seed=env.trace.seed)
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            values = agent.online_network(tensor).squeeze(0).cpu().numpy()
        q_values.extend(values.tolist())
        ordered = np.sort(values)
        margins.append(float(ordered[-1] - ordered[-2]))
    return {
        "mean_q_value": float(np.mean(q_values)),
        "mean_q_margin": float(np.mean(margins)),
        "max_q_value": float(np.max(q_values)),
        "min_q_value": float(np.min(q_values)),
    }


def validation_score(metrics: EvaluationMetrics, config: TrainingConfig) -> float:
    return (
        metrics.qoe
        - config.validation_rebuffer_weight * metrics.rebuffer_seconds
        - config.validation_switch_weight * metrics.total_bitrate_variation_mbps
    )


def ci95(values: list[float]) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1))
    critical = 2.920 if len(array) == 3 else 1.96
    return mean, sd, critical * sd / np.sqrt(len(array))


def run(output: Path, dataset_root: Path, seeds: tuple[int, ...] = SEEDS):
    splits = {split: load_split(dataset_root, split) for split in ("training", "validation", "test")}
    if any(not values for values in splits.values()):
        raise FileNotFoundError(f"missing dataset split under {dataset_root}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "configurations.json").write_text(json.dumps({
        "state_representation": STATE_REPRESENTATION,
        "seeds": seeds,
        "dataset_root": str(dataset_root),
        "dataset_config": asdict(DatasetConfig()),
        "selection_rule": "highest mean validation score across seeds; test data is not used for selection",
        "configurations": [asdict(config) for config in CONFIGS],
    }, indent=2) + "\n", encoding="utf-8")

    records = []
    histories = []
    validation_rows = []
    baseline_results = {}
    for name, policy in (
        ("fixed", fixed_bitrate(0)),
        ("throughput", throughput_based),
        ("buffer", buffer_based),
    ):
        metrics, profiles = evaluate(splits["test"], policy, representation="full", with_profiles=True)
        baseline_results[name] = {"overall": asdict(metrics), "by_profile": profiles}
    for config in CONFIGS:
        for seed in seeds:
            torch.manual_seed(seed)
            train_envs = make_envs(splits["training"])
            validation_envs = make_envs(splits["validation"])
            agent = DQNAgent(
                train_envs[0].observation_space.shape[0],
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                epsilon_decay=config.epsilon_decay,
                epsilon_end=config.epsilon_end,
                batch_size=config.batch_size,
                replay_capacity=config.replay_capacity,
                target_update_frequency=config.target_update_frequency,
                seed=seed,
            )
            training_config = TrainingConfig(
                episodes=config.episodes,
                learning_rate=config.learning_rate,
                gamma=config.gamma,
                batch_size=config.batch_size,
                replay_capacity=config.replay_capacity,
                epsilon_end=config.epsilon_end,
                epsilon_decay=config.epsilon_decay,
                target_update_frequency=config.target_update_frequency,
                seed=seed,
                checkpoint_every=max(1, config.episodes // 3),
                validation_every=max(1, config.episodes // 10),
            )
            checkpoint_dir = output / "training" / config.name / str(seed)
            history = DQNTrainer(agent, training_config).train(
                train_envs,
                checkpoint_dir,
                validation_environments=validation_envs,
            )
            selected = checkpoint_dir / "best_validation.pt"
            if selected.exists():
                agent.load(selected)
            agent.save(checkpoint_dir / "model_selected.pt", config.episodes)
            for index, score in enumerate(history.validation_scores):
                validation_rows.append({
                    "configuration": config.name,
                    "seed": seed,
                    "validation_index": (index + 1) * training_config.validation_every,
                    "score": score,
                })
            for episode, reward in enumerate(history.rewards):
                histories.append({
                    "configuration": config.name,
                    "seed": seed,
                    "episode": episode,
                    "reward": reward,
                    "loss": history.losses[episode],
                    "epsilon": history.epsilons[episode],
                })
            validation_metrics, profile_metrics = evaluate(
                splits["validation"],
                lambda state, _env: agent.select_action(state, explore=False),
                with_profiles=True,
            )
            test_metrics, test_profiles = evaluate(
                splits["test"],
                lambda state, _env: agent.select_action(state, explore=False),
                with_profiles=True,
            )
            records.append({
                "configuration": config.name,
                "seed": seed,
                "validation_score": validation_score(validation_metrics, training_config),
                "validation": asdict(validation_metrics),
                "test": asdict(test_metrics),
                "test_by_profile": test_profiles,
                "q_probe": q_probe(agent, make_envs(splits["validation"])),
            })

    metrics_dir = output / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    (metrics_dir / "per_seed_results.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    (metrics_dir / "baseline_results.json").write_text(json.dumps(baseline_results, indent=2) + "\n", encoding="utf-8")
    with (output / "training_histories.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(histories[0]))
        writer.writeheader()
        writer.writerows(histories)
    with (output / "validation_curves.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(validation_rows[0]))
        writer.writeheader()
        writer.writerows(validation_rows)
    summaries = {}
    for config in CONFIGS:
        values = [record for record in records if record["configuration"] == config.name]
        summaries[config.name] = {
            "validation_score": ci95([record["validation_score"] for record in values]),
            **{
                field: ci95([record["test"][field] for record in values])
                for field in ("qoe", "average_bitrate_mbps", "rebuffer_seconds", "total_bitrate_variation_mbps", "bitrate_switches")
            },
            "q_probe_margin": ci95([record["q_probe"]["mean_q_margin"] for record in values]),
        }
    (metrics_dir / "summary_mean_sd_ci95.json").write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    best = max(summaries, key=lambda name: summaries[name]["validation_score"][0])
    (metrics_dir / "selected_configuration.json").write_text(json.dumps({
        "configuration": best,
        "criterion": "highest mean validation score",
        "validation_score_mean_sd_ci95": summaries[best]["validation_score"],
    }, indent=2) + "\n", encoding="utf-8")
    existing_full = _load_existing_full_state_metrics(output)
    _write_report(output, records, summaries, best, baseline_results, existing_full)
    _plot(output, histories, validation_rows, summaries)


def _load_existing_full_state_metrics(output: Path):
    source = output.parent / "state_ablation" / "metrics" / "results.json"
    if not source.exists():
        return None
    data = json.loads(source.read_text(encoding="utf-8"))
    rows = [entry["dqn"] for entry in data if entry["variant"] == "full"]
    if not rows:
        return None
    return {
        field: ci95([float(row[field]) for row in rows])
        for field in ("qoe", "average_bitrate_mbps", "rebuffer_seconds", "total_bitrate_variation_mbps")
    }


def _write_report(output, records, summaries, best, baseline_results, existing_full):
    lines = [
        "# DQN Optimization Experiment",
        "",
        "The without-buffer state was used for every candidate. Each configuration was independently trained "
        "with seeds 11, 12, and 13. Selection used validation scores only; the untouched test split was used only "
        "for final reporting.",
        "",
        "## Configurations tested",
        "",
        "| Configuration | Learning rate | Gamma | Epsilon decay | Epsilon min | Target update | Replay capacity | Batch | Episodes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIGS:
        lines.append(
            f"| {config.name} | {config.learning_rate:g} | {config.gamma:g} | {config.epsilon_decay:g} | "
            f"{config.epsilon_end:g} | {config.target_update_frequency} | {config.replay_capacity} | "
            f"{config.batch_size} | {config.episodes} |"
        )
    lines += [
        "",
        "## Validation selection (mean ± SD; 95% CI half-width)",
        "",
        "| Configuration | Validation score | Test QoE | Test bitrate | Test rebuffer (s) | Test variation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in summaries.items():
        fmt = lambda value: f"{value[0]:.3f} ± {value[1]:.3f} (±{value[2]:.3f})"
        lines.append(
            f"| {name} | {fmt(summary['validation_score'])} | {fmt(summary['qoe'])} | "
            f"{fmt(summary['average_bitrate_mbps'])} | {fmt(summary['rebuffer_seconds'])} | "
            f"{fmt(summary['total_bitrate_variation_mbps'])} |"
        )
    lines += [
        "",
        f"## Selected configuration: `{best}`",
        "",
        "Selection was made before inspecting test outcomes, using the highest mean validation score. "
        "All configurations and seeds, including unfavorable outcomes, are retained in `metrics/per_seed_results.json`.",
        "",
        "## Existing comparison",
        "",
        "The existing full-state DQN and fixed, throughput-based, and buffer-based baselines remain unchanged. "
        "The optimized candidate uses the without-buffer state and is compared descriptively against those saved "
        "results; no test-set tuning or metric changes were performed.",
        "",
        "### Existing full-state DQN comparison",
        "",
        "| Model | QoE | Bitrate (Mbps) | Rebuffer (s) | Variation (Mbps) |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Existing full-state DQN | {existing_full['qoe'][0]:.3f} ± {existing_full['qoe'][1]:.3f} | "
            f"{existing_full['average_bitrate_mbps'][0]:.3f} ± {existing_full['average_bitrate_mbps'][1]:.3f} | "
            f"{existing_full['rebuffer_seconds'][0]:.3f} ± {existing_full['rebuffer_seconds'][1]:.3f} | "
            f"{existing_full['total_bitrate_variation_mbps'][0]:.3f} ± {existing_full['total_bitrate_variation_mbps'][1]:.3f} |"
            if existing_full is not None
            else "| Existing full-state DQN | unavailable | unavailable | unavailable | unavailable |"
        ),
        f"| Selected `{best}` | {summaries[best]['qoe'][0]:.3f} ± {summaries[best]['qoe'][1]:.3f} | "
        f"{summaries[best]['average_bitrate_mbps'][0]:.3f} ± {summaries[best]['average_bitrate_mbps'][1]:.3f} | "
        f"{summaries[best]['rebuffer_seconds'][0]:.3f} ± {summaries[best]['rebuffer_seconds'][1]:.3f} | "
        f"{summaries[best]['total_bitrate_variation_mbps'][0]:.3f} ± {summaries[best]['total_bitrate_variation_mbps'][1]:.3f} |",
        "",
        "### Baselines on the untouched test split",
        "",
        "| Policy | QoE | Bitrate (Mbps) | Rebuffer (s) | Variation (Mbps) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in baseline_results.items():
        metrics = value["overall"]
        lines.append(
            f"| {name} | {metrics['qoe']:.3f} | {metrics['average_bitrate_mbps']:.3f} | "
            f"{metrics['rebuffer_seconds']:.3f} | {metrics['total_bitrate_variation_mbps']:.3f} |"
        )
    lines += [
        "",
        "## Profile generalization for selected configuration",
        "",
        "| Profile | QoE mean | Bitrate mean (Mbps) | Rebuffer mean (s) |",
        "|---|---:|---:|---:|",
    ]
    selected_records = [record for record in records if record["configuration"] == best]
    for profile in sorted(selected_records[0]["test_by_profile"]):
        profile_rows = [record["test_by_profile"][profile] for record in selected_records]
        lines.append(
            f"| {profile} | {np.mean([row['qoe'] for row in profile_rows]):.3f} | "
            f"{np.mean([row['average_bitrate_mbps'] for row in profile_rows]):.3f} | "
            f"{np.mean([row['rebuffer_seconds'] for row in profile_rows]):.3f} |"
        )
    lines += [
        "",
        "## Limitations",
        "",
        "Only three training seeds and a 60-episode maximum were run. Confidence intervals are t-based with "
        "two degrees of freedom and are therefore wide. Q-value probes are validation-state snapshots after "
        "checkpoint selection, not evidence of causal convergence.",
        "",
    ]
    (output / "optimization_report.md").write_text("\n".join(lines), encoding="utf-8")


def _plot(output, histories, validation_rows, summaries):
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    for field, title, filename in (
        ("reward", "Training episode reward", "training_reward.png"),
        ("loss", "Training loss", "training_loss.png"),
        ("epsilon", "Epsilon decay", "epsilon_decay.png"),
    ):
        figure, axis = plt.subplots(figsize=(10, 5))
        for config in CONFIGS:
            values = [row for row in histories if row["configuration"] == config.name]
            grouped = {}
            for row in values:
                grouped.setdefault(row["episode"], []).append(float(row[field]))
            x = sorted(grouped)
            mean = [np.mean(grouped[index]) for index in x]
            sd = [np.std(grouped[index], ddof=1) for index in x]
            axis.plot(x, mean, label=config.name)
            axis.fill_between(x, np.asarray(mean) - sd, np.asarray(mean) + sd, alpha=0.12)
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.set_ylabel(field)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=220)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 5))
    for config in CONFIGS:
        values = [row for row in validation_rows if row["configuration"] == config.name]
        axis.plot([row["validation_index"] for row in values], [row["score"] for row in values], marker="o", label=config.name)
    axis.set_title("Validation score during training")
    axis.set_xlabel("Episode")
    axis.set_ylabel("Validation score")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "validation_score.png", dpi=220)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("experiments/results/research_multiseed/dqn_optimization"))
    parser.add_argument("--dataset-root", type=Path, default=Path("experiments/results/research_multiseed/datasets_11"))
    args = parser.parse_args()
    run(args.output, args.dataset_root)
    print(f"wrote optimization experiment to {args.output}")


if __name__ == "__main__":
    main()
