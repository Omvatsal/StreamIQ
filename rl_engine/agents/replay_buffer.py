"""Bounded experience replay."""

from collections import deque
from dataclasses import dataclass
import random
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, seed: Optional[int] = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._data: deque[Transition] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self._data.append(
            Transition(np.asarray(state, dtype=np.float32), int(action), float(reward),
                       np.asarray(next_state, dtype=np.float32), bool(done))
        )

    def sample(self, batch_size: int) -> list[Transition]:
        if not 0 < batch_size <= len(self._data):
            raise ValueError("batch_size must not exceed the number of stored transitions")
        return self._rng.sample(list(self._data), batch_size)

    def __len__(self) -> int:
        return len(self._data)
