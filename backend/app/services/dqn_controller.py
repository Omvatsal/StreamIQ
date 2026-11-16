"""Inference-only adapter from browser metrics to the trained ABR DQN."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from threading import Lock

import numpy as np

from rl_engine.agents.dqn import DQNAgent


BITRATES_MBPS = (0.3, 0.7, 1.2, 2.5, 5.0)
QUALITY_NAMES = ("240p", "360p", "480p", "720p", "1080p")


@dataclass(frozen=True)
class DecisionRequest:
    buffer_seconds: float
    throughput_history_mbps: tuple[float, ...]
    previous_bitrate_mbps: float
    previous_download_duration_seconds: float
    remaining_segments: float

    def state(self) -> np.ndarray:
        if len(self.throughput_history_mbps) != 5:
            raise ValueError("throughput_history_mbps must contain exactly five values")
        values = (
            self.buffer_seconds,
            *self.throughput_history_mbps,
            self.previous_bitrate_mbps,
            self.previous_download_duration_seconds,
            self.remaining_segments,
        )
        state = np.asarray(values, dtype=np.float32)
        if not np.isfinite(state).all() or (state < 0).any():
            raise ValueError("decision state values must be finite and non-negative")
        return state


class DQNController:
    def __init__(
        self,
        checkpoint_path: str | Path,
        state_size: int = 9,
        fallback_action: int = 0,
    ):
        if not 0 <= fallback_action < len(BITRATES_MBPS):
            raise ValueError("fallback_action is outside the bitrate action range")
        self.checkpoint_path = Path(checkpoint_path)
        self.fallback_action = fallback_action
        self._agent: Optional[DQNAgent] = None
        self._load_lock = Lock()
        self._inference_lock = Lock()
        self.state_size = state_size

    def _load(self) -> DQNAgent:
        if self._agent is None:
            with self._load_lock:
                if self._agent is None:
                    if not self.checkpoint_path.is_file():
                        raise FileNotFoundError(self.checkpoint_path)
                    agent = DQNAgent(self.state_size, action_size=5, epsilon_start=0.0, epsilon_end=0.0)
                    agent.load(self.checkpoint_path)
                    agent.online_network.eval()
                    self._agent = agent
        return self._agent

    def readiness(self) -> bool:
        try:
            self._load()
            return True
        except (FileNotFoundError, RuntimeError, OSError):
            return False

    def decide(self, request: DecisionRequest) -> dict[str, object]:
        error: Optional[str] = None
        try:
            state = request.state()
            with self._inference_lock:
                action = self._load().select_action(state, explore=False)
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exception:
            error = str(exception)
            action = self.fallback_action
        return {
            "action": action,
            "bitrate_mbps": BITRATES_MBPS[action],
            "representation": QUALITY_NAMES[action],
            "fallback": error is not None,
            "error": error,
        }
