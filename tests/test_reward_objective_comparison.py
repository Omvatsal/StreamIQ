from experiments.reward_objective_comparison import OBJECTIVES, make_envs
from network_lab.traces.generator import generate_trace
from network_lab.traces.models import TraceConfig


def test_comparison_declares_all_objectives_and_same_state():
    assert OBJECTIVES == ("legacy", "qoe_aligned", "evaluation_aligned")
    trace = generate_trace("stable", "comparison", TraceConfig(20, 1, seed=11))
    environments = [make_envs([trace], objective)[0] for objective in OBJECTIVES]
    assert {environment.reward_config.objective for environment in environments} == set(OBJECTIVES)
    assert {environment.observation_space.shape for environment in environments} == {(9,)}
    assert all(environment.state_representation == "without_buffer" for environment in environments)
