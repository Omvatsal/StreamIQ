import numpy as np
import pytest

from experiments.state_ablation import make_envs
from network_lab.traces.generator import generate_trace
from network_lab.traces.models import TraceConfig


def test_state_variants_preserve_dimension_and_semantics():
    trace = generate_trace("sudden_drop", "test", TraceConfig(20.0, 1.0, seed=7))
    full, without, change = [make_envs([trace], mode)[0] for mode in ("full", "without_buffer", "buffer_change")]
    full_state, _ = full.reset(seed=7)
    without_state, _ = without.reset(seed=7)
    change_state, _ = change.reset(seed=7)
    assert full_state.shape == without_state.shape == change_state.shape == (9,)
    assert without_state[0] == 0
    assert change_state[0] == 0
    full.step(2)
    without.step(2)
    change.step(2)
    assert without._observation()[0] == 0
    assert np.isclose(change._observation()[0], change.buffer.occupancy_seconds)
    assert full._observation()[0] == change.buffer.occupancy_seconds


def test_invalid_state_representation_is_rejected():
    trace = generate_trace("stable", "test", TraceConfig(10.0, 1.0, seed=2))
    with pytest.raises(ValueError, match="unsupported state representation"):
        make_envs([trace], "invalid")
