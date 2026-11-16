import pytest

from network_lab.traces.generator import generate_trace
from network_lab.traces.models import TraceConfig
from rl_engine.environment.abr_env import ABREnvironment
from rl_engine.environment.reward import RewardConfig, calculate_reward
from rl_engine.environment.video import VideoModel
from rl_engine.evaluation.metrics import summarize


def test_legacy_objective_remains_unchanged():
    assert calculate_reward(4, 0.0, 0.3, 5.0, RewardConfig()) == pytest.approx(1.65)


def test_qoe_aligned_objective_uses_bitrate_units():
    reward = calculate_reward(
        4, 0.0, 0.3, 5.0, RewardConfig(objective="qoe_aligned")
    )
    assert reward == pytest.approx(2.65)


def test_all_objectives_are_selectable():
    assert RewardConfig().objective == "legacy"
    assert RewardConfig(objective="qoe_aligned").objective == "qoe_aligned"
    assert RewardConfig(objective="evaluation_aligned").objective == "evaluation_aligned"


def test_evaluation_aligned_objective_uses_evaluation_scale():
    reward = calculate_reward(
        4, 0.0, 0.3, 5.0, RewardConfig(objective="evaluation_aligned")
    )
    assert reward == pytest.approx(5.0 - 5.0 * 4.7)


def test_first_segment_has_no_switching_penalty():
    assert calculate_reward(
        4, 0.0, None, 5.0, RewardConfig(objective="qoe_aligned")
    ) == pytest.approx(5.0)
    assert calculate_reward(
        4, 0.0, None, 5.0, RewardConfig(objective="evaluation_aligned")
    ) == pytest.approx(5.0)


def test_ten_segment_terminal_aggregation_reports_existing_qoe_inconsistency():
    trace = generate_trace("stable", "aggregation", TraceConfig(60, 1, seed=301))
    environment = ABREnvironment(
        trace,
        video=VideoModel(total_segments=10),
        reward_config=RewardConfig(objective="qoe_aligned"),
    )
    environment.reset(seed=301)
    reward_sum = 0.0
    for action in (0, 2, 4, 3, 1, 4, 2, 3, 4, 1):
        _, reward, terminated, _, _ = environment.step(action)
        reward_sum += reward
    assert terminated is True
    evaluation_qoe = summarize(environment.metrics).qoe
    expected_difference = (
        36.0 * environment.metrics.total_rebuffer_seconds
        + 4.5 * sum(
            abs(current.bitrate_mbps - previous.bitrate_mbps)
            for previous, current in zip(
                environment.metrics.segments, environment.metrics.segments[1:]
            )
        )
    )
    assert reward_sum - 10 * evaluation_qoe == pytest.approx(expected_difference)


def test_evaluation_aligned_terminal_sum_exactly_matches_ten_times_qoe():
    trace = generate_trace("stable", "evaluation-aligned", TraceConfig(60, 1, seed=301))
    environment = ABREnvironment(
        trace,
        video=VideoModel(total_segments=10),
        reward_config=RewardConfig(objective="evaluation_aligned"),
    )
    environment.reset(seed=301)
    for action in (0, 2, 4, 3, 1, 4, 2, 3, 4, 1):
        _, _, terminated, _, _ = environment.step(action)
    assert terminated is True
    assert environment.metrics.accumulated_reward == pytest.approx(
        10 * summarize(environment.metrics).qoe
    )


def test_non_ten_episode_does_not_assume_exact_identity():
    trace = generate_trace("stable", "non-ten", TraceConfig(60, 1, seed=301))
    environment = ABREnvironment(
        trace,
        video=VideoModel(total_segments=3),
        reward_config=RewardConfig(objective="evaluation_aligned"),
    )
    environment.reset(seed=301)
    for action in (0, 2, 4):
        environment.step(action)
    evaluation = summarize(environment.metrics)
    variation = sum(
        abs(current.bitrate_mbps - previous.bitrate_mbps)
        for previous, current in zip(
            environment.metrics.segments, environment.metrics.segments[1:]
        )
    )
    expected_difference = (
        (4 * 3 - 40) * environment.metrics.total_rebuffer_seconds
        + (0.5 * 3 - 5) * variation
    )
    assert environment.metrics.accumulated_reward - 3 * evaluation.qoe == pytest.approx(
        expected_difference
    )
    assert abs(expected_difference) > 0


def test_zero_rebuffer_and_switch_terms_are_exact_for_any_segment_count():
    trace = generate_trace("stable", "zero-terms", TraceConfig(60, 1, seed=301))
    environment = ABREnvironment(
        trace,
        video=VideoModel(total_segments=1),
        reward_config=RewardConfig(objective="evaluation_aligned"),
    )
    environment.reset(seed=301)
    environment.buffer.occupancy_seconds = 4.0
    for _ in range(1):
        environment.step(0)
    assert environment.metrics.total_rebuffer_seconds == pytest.approx(0)
    assert environment.metrics.accumulated_reward == pytest.approx(
        summarize(environment.metrics).qoe
    )


def test_invalid_objective_is_rejected():
    with pytest.raises(ValueError, match="unsupported reward objective"):
        RewardConfig(objective="invalid")
