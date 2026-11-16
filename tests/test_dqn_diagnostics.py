from experiments.dqn_diagnostics import _summarize, reward_components


def test_reward_components_match_environment_formula():
    components = reward_components(3, 2.0, 0.7, 2.5)

    assert components["quality_reward"] == 3.0
    assert components["rebuffer_penalty"] == -8.0
    assert components["switching_penalty"] == -0.9
    assert components["reward"] == -5.9


def test_profile_summary_counts_actions_and_components():
    rows = [
        {
            "trace_id": "trace-1",
            "policy": "dqn",
            "profile": "stable",
            "action": 0,
            "selected_bitrate_mbps": 0.3,
            "rebuffer_seconds": 0.0,
            "reward": 0.0,
            "quality_reward": 0.0,
            "rebuffer_penalty": 0.0,
            "switching_penalty": 0.0,
        },
        {
            "trace_id": "trace-1",
            "policy": "dqn",
            "profile": "stable",
            "action": 2,
            "selected_bitrate_mbps": 1.2,
            "rebuffer_seconds": 1.0,
            "reward": -3.4,
            "quality_reward": 2.0,
            "rebuffer_penalty": -4.0,
            "switching_penalty": -1.4,
        },
    ]

    summary = _summarize(rows)[0]

    assert summary["segments"] == 1
    assert summary["qoe"] == -3.7
    assert summary["switches"] == 1
    assert summary["action_0_frequency"] == 0.5
    assert summary["action_2_frequency"] == 0.5
    assert summary["rebuffer_penalty"] == -4.0
