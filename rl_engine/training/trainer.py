"""DQN training loop."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from rl_engine.agents.dqn import DQNAgent
from rl_engine.environment.abr_env import ABREnvironment
from rl_engine.evaluation.evaluator import evaluate_policy
from rl_engine.evaluation.metrics import EvaluationMetrics
from .config import TrainingConfig


@dataclass
class TrainingHistory:
    rewards: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    epsilons: list[float] = field(default_factory=list)
    validation_scores: list[float] = field(default_factory=list)

    def to_csv(self, path: str | Path) -> None:
        import csv
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("episode", "reward", "mean_loss", "epsilon"))
            for index, reward in enumerate(self.rewards):
                writer.writerow((index, reward, self.losses[index] if index < len(self.losses) else "", self.epsilons[index]))


class DQNTrainer:
    def __init__(self, agent: DQNAgent, config: TrainingConfig):
        self.agent, self.config = agent, config

    def train(
        self,
        environments: Sequence[ABREnvironment] | Callable[[], Sequence[ABREnvironment]],
        checkpoint_dir: str | Path | None = None,
        validation_environments: Sequence[ABREnvironment] | Callable[[], Sequence[ABREnvironment]] | None = None,
    ) -> TrainingHistory:
        history = TrainingHistory()
        best_validation_score = float("-inf")
        rng = np.random.default_rng(self.config.seed)
        for episode in range(self.config.episodes):
            envs = environments() if callable(environments) else environments
            env = envs[int(rng.integers(len(envs)))]
            state, _ = env.reset(seed=self.config.seed + episode)
            done = False
            reward_total, losses = 0.0, []
            while not done:
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                self.agent.remember(state, action, reward, next_state, terminated or truncated)
                loss = self.agent.learn()
                if loss is not None:
                    losses.append(loss)
                state, done = next_state, terminated or truncated
                reward_total += reward
            history.rewards.append(reward_total)
            history.losses.append(float(np.mean(losses)) if losses else 0.0)
            history.epsilons.append(self.agent.epsilon)
            if validation_environments is not None and (episode + 1) % self.config.validation_every == 0:
                validation_envs = (
                    validation_environments()
                    if callable(validation_environments)
                    else validation_environments
                )
                metrics = evaluate_policy(
                    validation_envs,
                    lambda state, _env: self.agent.select_action(state, explore=False),
                )
                score = (
                    metrics.qoe
                    - self.config.validation_rebuffer_weight * metrics.rebuffer_seconds
                    - self.config.validation_switch_weight * metrics.total_bitrate_variation_mbps
                )
                history.validation_scores.append(score)
                if checkpoint_dir and score > best_validation_score:
                    best_validation_score = score
                    self.agent.save(Path(checkpoint_dir) / "best_validation.pt", episode + 1)
            if checkpoint_dir and (episode + 1) % self.config.checkpoint_every == 0:
                self.agent.save(Path(checkpoint_dir) / f"checkpoint_{episode + 1}.pt", episode + 1)
        return history
