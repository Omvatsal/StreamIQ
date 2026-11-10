"""PyTorch DQN model inference service."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch

from backend.app.core.config import settings
from rl_engine.agents.dqn import DQNAgent
from rl_engine.environment.video import VideoModel


class RLInferenceService:
    """Provides low-latency ABR decisions using the trained DQN policy."""

    def __init__(self, checkpoint_path: Optional[Path] = None) -> None:
        self.checkpoint_path = checkpoint_path or settings.model_checkpoint
        self.video = VideoModel()
        self.state_size = 9
        self.action_size = 5
        self.agent: Optional[DQNAgent] = None
        self._load_agent()

    def _load_agent(self) -> None:
        self.agent = DQNAgent(state_size=self.state_size, action_size=self.action_size, seed=42)
        if self.checkpoint_path.exists():
            self.agent.load(self.checkpoint_path)

    def predict(
        self,
        buffer_occupancy_seconds: float,
        throughput_history: List[float],
        previous_bitrate_mbps: Optional[float] = None,
        previous_download_duration_seconds: float = 0.0,
        remaining_segments: int = 10,
    ) -> Dict[str, Any]:
        """Compute the recommended bitrate tier for the next segment."""
        # Pad or truncate throughput history to length 5
        history = list(throughput_history)
        if len(history) < 5:
            history = [0.0] * (5 - len(history)) + history
        else:
            history = history[-5:]

        prev_bitrate = previous_bitrate_mbps if previous_bitrate_mbps is not None else 0.0

        observation = np.asarray(
            [
                float(buffer_occupancy_seconds),
                *history,
                float(prev_bitrate),
                float(previous_download_duration_seconds),
                float(remaining_segments),
            ],
            dtype=np.float32,
        )

        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.agent.device).unsqueeze(0)
            q_values_tensor = self.agent.online_network(tensor)[0]
            action = int(q_values_tensor.argmax().item())
            q_values = [round(float(q), 4) for q in q_values_tensor.tolist()]

        return {
            "selected_action": action,
            "quality_name": self.video.quality_names[action],
            "bitrate_mbps": self.video.bitrates_mbps[action],
            "q_values": q_values,
            "observation": [round(float(x), 4) for x in observation.tolist()],
        }


# Global singleton
rl_service = RLInferenceService()

