import csv

import numpy as np
import pytest

from network_lab.traces.generator import generate_trace
from network_lab.traces.models import NetworkCondition, NetworkTrace, TraceConfig
from rl_engine.environment.abr_env import ABREnvironment
from rl_engine.environment.buffer import BufferState
from rl_engine.environment.metrics import EpisodeMetrics
from rl_engine.environment.network import TraceNetworkAdapter
from rl_engine.environment.reward import RewardConfig, calculate_reward
from rl_engine.environment.video import VideoModel


def trace(profile="stable", seed=42):
    return generate_trace(profile, profile, TraceConfig(20, 1, seed=seed))


def test_video_segment_size_and_buffer_stall():
    video = VideoModel(total_segments=2)
    assert video.segment_size_megabytes(4) == 2.5
    buffer = BufferState(occupancy_seconds=1, capacity_seconds=10)
    assert buffer.download_and_play(3, 4) == 2
    assert buffer.occupancy_seconds == 4


def test_download_uses_trace_bandwidth():
    network = TraceNetworkAdapter(
        NetworkTrace("x", "stable", 1, 10, 1, (NetworkCondition(0, 2),))
    )
    result = network.download(1, 0)
    assert result.duration_seconds == pytest.approx(4.005)
    assert result.throughput_mbps == pytest.approx(2, rel=0.01)


def test_reward_accounts_for_switch_magnitude():
    reward = calculate_reward(4, 0, 0.3, 5.0, RewardConfig())
    assert reward == pytest.approx(4 - 2.35)


def test_observation_options_do_not_leak_future_state():
    environment = ABREnvironment(
        trace(), throughput_history_length=0, include_previous_bitrate=False
    )
    observation, _ = environment.reset()
    assert observation.shape == (3,)
    assert environment.observation_space.contains(observation)


def test_delay_is_included_once_per_segment():
    low_delay = NetworkTrace("low", "stable", 1, 10, 1, (NetworkCondition(0, 2, 0),))
    high_delay = NetworkTrace("high", "stable", 1, 10, 1, (NetworkCondition(0, 2, 100),))
    low = TraceNetworkAdapter(low_delay).download(1, 0)
    high = TraceNetworkAdapter(high_delay).download(1, 0)
    assert high.duration_seconds - low.duration_seconds == pytest.approx(0.1)


def test_environment_reset_step_and_termination():
    environment = ABREnvironment(trace(), video=VideoModel(total_segments=3))
    observation, info = environment.reset(seed=7)
    assert observation.shape == environment.observation_space.shape
    assert info["trace_id"] == "stable"
    for index in range(3):
        observation, reward, terminated, truncated, _ = environment.step(index)
    assert terminated is True
    assert truncated is False
    assert len(environment.metrics.segments) == 3


def test_reproducibility_and_csv_export(tmp_path):
    first = ABREnvironment(trace("oscillating"), video=VideoModel(total_segments=3))
    second = ABREnvironment(trace("oscillating"), video=VideoModel(total_segments=3))
    first.reset(seed=1)
    second.reset(seed=1)
    for action in (0, 2, 4):
        first.step(action)
        second.step(action)
    assert first.metrics.segments == second.metrics.segments
    path = tmp_path / "metrics.csv"
    first.metrics.to_csv(path)
    with path.open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 3


def test_sudden_drop_increases_download_time():
    environment = ABREnvironment(trace("sudden_drop"), video=VideoModel(total_segments=5))
    environment.reset()
    durations = []
    for _ in range(5):
        _, _, terminated, _, _ = environment.step(4)
        durations.append(environment.metrics.segments[-1].download_duration_seconds)
        if terminated:
            break
    assert max(durations) > min(durations)
