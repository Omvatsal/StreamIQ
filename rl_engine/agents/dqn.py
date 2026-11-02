"""Deep Q-learning agent with a target network."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn, optim

from .network import QNetwork
from .replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int = 5,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        replay_capacity: int = 100_000,
        target_update_frequency: int = 100,
        gradient_clip: float = 1.0,
        seed: Optional[int] = None,
        device: str = "cpu",
    ):
        if not 0 <= epsilon_end <= epsilon_start <= 1:
            raise ValueError("epsilon values must satisfy 0 <= end <= start <= 1")
        if not 0 < gamma <= 1 or epsilon_decay <= 0:
            raise ValueError("gamma and epsilon_decay are invalid")
        self.state_size, self.action_size = state_size, action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.gradient_clip = gradient_clip
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            torch.manual_seed(seed)
        self.online_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = ReplayBuffer(replay_capacity, seed)
        self.train_steps = 0

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(self.online_network(tensor).argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.add(state, action, reward, next_state, done)

    def learn(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)
        current = self.online_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            target = rewards + self.gamma * (1 - dones) * self.target_network(next_states).max(dim=1).values
        loss = self.loss_fn(current, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), self.gradient_clip)
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())

    def save(self, path: str | Path, episode: int = 0) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"online": self.online_network.state_dict(), "target": self.target_network.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "epsilon": self.epsilon,
                    "train_steps": self.train_steps, "episode": episode}, destination)

    def load(self, path: str | Path) -> int:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.online_network.load_state_dict(checkpoint["online"])
        self.target_network.load_state_dict(checkpoint["target"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = float(checkpoint["epsilon"])
        self.train_steps = int(checkpoint["train_steps"])
        return int(checkpoint.get("episode", 0))
