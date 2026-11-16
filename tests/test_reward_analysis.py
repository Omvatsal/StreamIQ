from experiments.reward_analysis import enrich, episode_metrics


def test_reward_components_recompute_exactly():
    rows = [
        {
            "trace_id": "t",
            "profile": "stable",
            "policy": "dqn",
            "segment_index": "0",
            "quality_reward": "4",
            "rebuffer_penalty": "-2",
            "switching_penalty": "-0.5",
            "reward": "1.5",
            "rebuffer_seconds": "0.5",
            "selected_bitrate_mbps": "5",
        }
    ]
    enriched = enrich(rows)
    assert enriched[0]["reward_recomputed"] == 1.5


def test_episode_qoe_uses_bitrate_not_quality_index():
    rows = [
        {
            "trace_id": "t",
            "profile": "stable",
            "policy": "dqn",
            "segment_index": str(index),
            "quality_reward": "4",
            "rebuffer_penalty": "0",
            "switching_penalty": "0",
            "reward": "4",
            "rebuffer_seconds": "0",
            "selected_bitrate_mbps": "5",
        }
        for index in range(2)
    ]
    result = episode_metrics(enrich(rows))[0]
    assert result["accumulated_reward"] == 8
    assert result["independent_qoe"] == 5
