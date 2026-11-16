from experiments.reward_objective_policy_diagnostics import (
    action_entropy,
    consecutive_identical_actions,
    reward_components,
    transition_counts,
)


def test_action_statistics_are_deterministic():
    actions = [0, 0, 1, 1, 1, 2]
    assert consecutive_identical_actions(actions) == 3
    transitions = transition_counts(actions)
    assert transitions[(0, 0)] == 1
    assert transitions[(0, 1)] == 1
    assert transitions[(1, 1)] == 2
    assert action_entropy(actions) > 0


def test_reward_components_match_each_objective_and_first_segment():
    legacy = reward_components("legacy", 3, 2.5, None, 0.25)
    assert legacy == {
        "quality_contribution": 3.0,
        "rebuffer_penalty": -1.0,
        "switching_penalty": 0.0,
        "net_reward": 2.0,
        "switch_magnitude_mbps": 0.0,
    }
    aligned = reward_components("evaluation_aligned", 3, 2.5, 0.3, 0.25)
    assert aligned["quality_contribution"] == 2.5
    assert aligned["rebuffer_penalty"] == -10.0
    assert aligned["switching_penalty"] == -11.0
    assert aligned["net_reward"] == -18.5
