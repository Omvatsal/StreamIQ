"""Configurable DQN training hyperparameters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 100
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    replay_capacity: int = 100_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    target_update_frequency: int = 100
    gradient_clip: float = 1.0
    seed: int = 42
    checkpoint_every: int = 25
