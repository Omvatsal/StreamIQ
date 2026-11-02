import numpy as np
import pytest

torch = pytest.importorskip("torch")

from network_lab.traces.generator import generate_trace
from network_lab.traces.models import TraceConfig
from rl_engine.agents.dqn import DQNAgent
from rl_engine.agents.network import QNetwork
from rl_engine.agents.replay_buffer import ReplayBuffer
from rl_engine.environment.abr_env import ABREnvironment
from rl_engine.evaluation.baselines import buffer_based, fixed_bitrate, throughput_based
from rl_engine.evaluation.evaluator import evaluate_policy
from rl_engine.evaluation.metrics import summarize
from rl_engine.environment.video import VideoModel


def make_env():
    return ABREnvironment(
        generate_trace("stable", "test", TraceConfig(20, seed=1)),
        video=VideoModel(total_segments=3),
    )


def test_network_and_replay_buffer():
    network = QNetwork(9, 5)
    assert network(torch.zeros((2, 9))).shape == (2, 5)
    buffer = ReplayBuffer(3, seed=1)
    state = np.zeros(9, dtype=np.float32)
    for _ in range(3):
        buffer.add(state, 1, 1.0, state, False)
    assert len(buffer.sample(2)) == 2


def test_agent_learning_action_and_checkpoint(tmp_path):
    agent = DQNAgent(9, batch_size=2, replay_capacity=10, seed=2)
    state = np.zeros(9, dtype=np.float32)
    for _ in range(2):
        agent.remember(state, 0, 1, state, False)
    assert agent.learn() is not None
    assert 0 <= agent.select_action(state) < 5
    path = tmp_path / "model.pt"
    agent.save(path, episode=3)
    restored = DQNAgent(9, batch_size=2, seed=2)
    assert restored.load(path) == 3


def test_baselines_and_metrics_on_same_trace():
    results = {
        "fixed": evaluate_policy([make_env()], fixed_bitrate(1)),
        "throughput": evaluate_policy([make_env()], throughput_based),
        "buffer": evaluate_policy([make_env()], buffer_based),
    }
    assert set(results) == {"fixed", "throughput", "buffer"}
    assert results["fixed"].average_bitrate_mbps == pytest.approx(0.7)
    assert summarize(make_env().metrics).qoe == 0
