import numpy as np

from experiments.state_action_diagnostics import (
    STATE_FEATURES,
    extract_q_values,
    named_state,
)


def test_named_state_preserves_all_model_features():
    state = np.arange(9, dtype=np.float32)

    named = named_state(state)

    assert tuple(named) == STATE_FEATURES
    assert named["buffer_occupancy_seconds"] == 0.0
    assert named["remaining_segments"] == 8.0


def test_extract_q_values_returns_argmax_and_margin():
    class Network:
        def __call__(self, tensor):
            import torch

            return torch.tensor([[1.0, 3.0, 2.0, -1.0, 0.0]])

    class Agent:
        device = "cpu"
        online_network = Network()

    values, action, margin = extract_q_values(Agent(), np.zeros(9, dtype=np.float32))

    assert values.tolist() == [1.0, 3.0, 2.0, -1.0, 0.0]
    assert action == 1
    assert margin == 1.0
