"""Train and compare a DQN on the saved network profiles."""

import argparse
from pathlib import Path

import torch

from network_lab.traces.generator import PROFILE_NAMES, generate_trace
from network_lab.traces.models import TraceConfig
from rl_engine.agents.dqn import DQNAgent
from rl_engine.environment.abr_env import ABREnvironment
from rl_engine.evaluation.baselines import buffer_based, fixed_bitrate, throughput_based
from rl_engine.evaluation.evaluator import evaluate_policy
from rl_engine.evaluation.visualize import plot_comparison, plot_training
from .config import TrainingConfig
from .trainer import DQNTrainer


def build_environments(seed: int, duration: float = 60) -> list[ABREnvironment]:
    return [
        ABREnvironment(generate_trace(profile, f"{profile}_{seed}", TraceConfig(duration, seed=seed)))
        for profile in PROFILE_NAMES
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--duration", type=float, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/results/dqn"))
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    environments = build_environments(args.seed, args.duration)
    state_size = environments[0].observation_space.shape[0]
    config = TrainingConfig(episodes=args.episodes, seed=args.seed)
    agent = DQNAgent(state_size, learning_rate=config.learning_rate, gamma=config.gamma,
                     batch_size=config.batch_size, replay_capacity=config.replay_capacity,
                     epsilon_start=config.epsilon_start, epsilon_end=config.epsilon_end,
                     epsilon_decay=config.epsilon_decay, target_update_frequency=config.target_update_frequency,
                     gradient_clip=config.gradient_clip, seed=config.seed)
    start_episode = agent.load(args.resume) if args.resume else 0
    history = DQNTrainer(agent, config).train(environments, args.output_dir / "checkpoints")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agent.save(args.output_dir / "model_final.pt", config.episodes)
    history.to_csv(args.output_dir / "training.csv")
    plot_training(history, args.output_dir / "training.png")

    policies = {
        "dqn": lambda state, env: agent.select_action(state, explore=False),
        "fixed_240p": fixed_bitrate(0),
        "throughput": throughput_based,
        "buffer": buffer_based,
    }
    results = {name: evaluate_policy(build_environments(args.seed, args.duration), policy)
               for name, policy in policies.items()}
    plot_comparison(results, args.output_dir / "comparison.png")
    with (args.output_dir / "evaluation.txt").open("w", encoding="utf-8") as stream:
        for name, metrics in results.items():
            stream.write(f"{name}: {metrics}\n")
    print(f"trained from episode {start_episode}; results written to {args.output_dir}")


if __name__ == "__main__":
    main()
