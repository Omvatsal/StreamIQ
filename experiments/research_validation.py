"""Reproducible research validation runner.

This runner generates disjoint datasets, retrains each ablation, evaluates
policies on identical unseen traces, and writes descriptive/paired statistics.
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from experiments.configurations.datasets import DatasetConfig, generate_dataset
from network_lab.traces.models import NetworkTrace
from rl_engine.agents.dqn import DQNAgent
from rl_engine.environment.abr_env import ABREnvironment
from rl_engine.evaluation.baselines import buffer_based, fixed_bitrate, throughput_based
from rl_engine.evaluation.evaluator import evaluate_policy, evaluate_policy_per_trace
from rl_engine.evaluation.research_plots import plot_policy_metric
from rl_engine.evaluation.report import write_research_report
from rl_engine.evaluation.statistics import paired_statistics
from rl_engine.training.ablations import ABLATIONS, AblationConfig
from rl_engine.training.config import TrainingConfig
from rl_engine.training.trainer import DQNTrainer
from rl_engine.evaluation.visualize import plot_training


def make_envs(traces: list[NetworkTrace], ablation: AblationConfig) -> list[ABREnvironment]:
    return [
        ABREnvironment(
            trace,
            reward_config=ablation.reward_config,
            throughput_history_length=ablation.throughput_history_length,
            include_previous_bitrate=ablation.include_previous_bitrate,
        )
        for trace in traces
    ]


def run(output: Path, episodes: int, seeds: tuple[int, ...]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(
        json.dumps(
            {
                "episodes": episodes,
                "seeds": seeds,
                "dataset_config": asdict(DatasetConfig()),
                "ablations": [ablation.name for ablation in ABLATIONS],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    all_results = {}
    for seed in seeds:
        torch.manual_seed(seed)
        datasets = generate_dataset(output / f"datasets_{seed}", DatasetConfig())
        train_traces = datasets["training"]
        validation_traces = datasets["validation"]
        test_traces = datasets["test"]
        for ablation in ABLATIONS:
            train_envs = make_envs(train_traces, ablation)
            validation_envs = make_envs(validation_traces, ablation)
            agent = DQNAgent(
                train_envs[0].observation_space.shape[0],
                batch_size=32,
                seed=seed,
            )
            config = TrainingConfig(episodes=episodes, seed=seed, validation_every=max(1, episodes // 5))
            history = DQNTrainer(agent, config).train(
                train_envs,
                output / "training" / str(seed) / ablation.name,
                validation_environments=validation_envs,
            )
            best_checkpoint = (
                output / "training" / str(seed) / ablation.name / "best_validation.pt"
            )
            if best_checkpoint.exists():
                agent.load(best_checkpoint)
            agent.save(
                output / "training" / str(seed) / ablation.name / "model_selected.pt",
                episodes,
            )
            test_envs = make_envs(test_traces, ablation)
            policies = {
                "dqn": lambda state, _env: agent.select_action(state, explore=False),
                "fixed": fixed_bitrate(0),
                "throughput": throughput_based,
                "buffer": buffer_based,
            }
            policy_results = {
                name: evaluate_policy(
                    test_envs if name == "dqn" else make_envs(test_traces, ablation),
                    policy,
                ).__dict__
                for name, policy in policies.items()
            }
            paired = {}
            dqn_per_trace = evaluate_policy_per_trace(
                make_envs(test_traces, ablation), policies["dqn"]
            )
            for baseline_name in ("fixed", "throughput", "buffer"):
                baseline_per_trace = evaluate_policy_per_trace(
                    make_envs(test_traces, ablation), policies[baseline_name]
                )
                paired[baseline_name] = paired_statistics(
                    [item.qoe for item in dqn_per_trace],
                    [item.qoe for item in baseline_per_trace],
                ).__dict__
            all_results[f"{seed}:{ablation.name}"] = {
                "policies": policy_results,
                "paired_dqn_qoe": paired,
            }
            history.to_csv(output / "training" / str(seed) / f"{ablation.name}.csv")
            plot_training(
                history,
                output / "plots" / "training" / str(seed) / f"{ablation.name}.png",
            )
    (output / "metrics").mkdir(exist_ok=True)
    (output / "metrics" / "results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    summary = {}
    for key, policies in all_results.items():
        summary[key] = policies["policies"]["dqn"]
    for metric in ("average_bitrate_mbps", "rebuffer_seconds", "total_bitrate_variation_mbps"):
        plot_policy_metric(
            {key: type("Metrics", (), value)() for key, value in summary.items()},
            metric,
            output / "plots" / f"dqn_{metric}.png",
        )
    write_research_report(
        output / "reports" / "research_report.md",
        str(output / "datasets_*/manifest.json"),
        summary,
        [
            "Results are descriptive unless paired statistical tests are explicitly computed.",
            "Test traces are generated from seeds disjoint from training and validation seeds.",
            "DQN ablations retrain from fresh random initialization.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--output", type=Path, default=Path("experiments/results"))
    args = parser.parse_args()
    run(args.output, args.episodes, tuple(args.seeds))


if __name__ == "__main__":
    main()
