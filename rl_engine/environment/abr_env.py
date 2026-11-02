"""Gymnasium-compatible adaptive bitrate environment."""

from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from network_lab.traces.models import NetworkTrace

from .buffer import BufferState
from .metrics import EpisodeMetrics, SegmentMetric
from .network import TraceNetworkAdapter
from .reward import RewardConfig, calculate_reward
from .video import VideoModel


class ABREnvironment(gym.Env):
    """One action downloads one four-second video segment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        trace: NetworkTrace,
        video: Optional[VideoModel] = None,
        buffer_capacity_seconds: float = 60.0,
        throughput_history_length: int = 5,
        reward_config: Optional[RewardConfig] = None,
        include_previous_bitrate: bool = True,
        network_packet_loss_retries: bool = False,
    ):
        super().__init__()
        if throughput_history_length < 0:
            raise ValueError("throughput_history_length cannot be negative")
        self.trace = trace
        self.video = video or VideoModel()
        self.buffer_capacity_seconds = buffer_capacity_seconds
        self.history_length = throughput_history_length
        self.include_previous_bitrate = include_previous_bitrate
        self.reward_config = reward_config or RewardConfig()
        self.action_space = spaces.Discrete(5)
        state_size = 1 + throughput_history_length + int(include_previous_bitrate) + 2
        self.observation_space = spaces.Box(
            low=np.zeros(state_size, dtype=np.float32),
            high=np.array(
                [buffer_capacity_seconds]
                + [np.finfo(np.float32).max] * throughput_history_length
                + ([max(self.video.bitrates_mbps)] if include_previous_bitrate else [])
                + [np.finfo(np.float32).max, np.inf],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )
        self.network = TraceNetworkAdapter(
            trace, packet_loss_retries=network_packet_loss_retries
        )
        self.metrics = EpisodeMetrics()
        self.buffer = BufferState(capacity_seconds=buffer_capacity_seconds)
        self._time = 0.0
        self._segment_index = 0
        self._previous_quality: Optional[int] = None
        self._previous_download = 0.0
        self._throughputs = [0.0] * throughput_history_length

    def _observation(self) -> np.ndarray:
        return np.asarray(
            [
                self.buffer.occupancy_seconds,
                *self._throughputs,
                *(
                    [0.0 if self._previous_quality is None
                     else self.video.bitrates_mbps[self._previous_quality]]
                    if self.include_previous_bitrate else []
                ),
                self._previous_download,
                float(self.video.segment_count - self._segment_index),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        self.network = TraceNetworkAdapter(
            self.trace, packet_loss_retries=self.network.packet_loss_retries
        )
        self.metrics = EpisodeMetrics()
        self.buffer = BufferState(capacity_seconds=self.buffer_capacity_seconds)
        self._time = 0.0
        self._segment_index = 0
        self._previous_quality = None
        self._previous_download = 0.0
        self._throughputs = [0.0] * self.history_length
        return self._observation(), {"trace_id": self.trace.trace_id}

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError("action must be an integer from 0 through 4")
        if self._segment_index >= self.video.segment_count:
            raise RuntimeError("episode has terminated; call reset()")
        quality = int(action)
        result = self.network.download(
            self.video.segment_size_megabytes(quality), self._time
        )
        rebuffer = self.buffer.download_and_play(
            result.duration_seconds, self.video.segment_duration_seconds
        )
        previous_bitrate = (
            None if self._previous_quality is None
            else self.video.bitrates_mbps[self._previous_quality]
        )
        reward = calculate_reward(
            quality,
            rebuffer,
            previous_bitrate,
            self.video.bitrates_mbps[quality],
            self.reward_config,
        )
        self._time = result.end_time_seconds
        self._previous_download = result.duration_seconds
        if self.history_length:
            self._throughputs = (
                self._throughputs + [result.throughput_mbps]
            )[-self.history_length:]
        else:
            self._throughputs = []
        self._segment_index += 1
        self._previous_quality = quality
        self.metrics.add(
            SegmentMetric(
                self._segment_index - 1,
                quality,
                self.video.quality_names[quality],
                self.video.bitrates_mbps[quality],
                result.throughput_mbps,
                result.duration_seconds,
                self.buffer.occupancy_seconds,
                rebuffer,
                reward,
                result.condition.bandwidth_mbps,
            )
        )
        terminated = self._segment_index >= self.video.segment_count
        info = {"segment_index": self._segment_index - 1, "time_seconds": self._time}
        return self._observation(), reward, terminated, False, info
