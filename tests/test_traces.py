from network_lab.traces.generator import PROFILE_NAMES, generate_trace
from network_lab.traces.models import NetworkCondition, NetworkTrace, TraceConfig
from network_lab.traces.replay import ReplayController


def config(seed=42):
    return TraceConfig(duration_seconds=20, interval_seconds=1, seed=seed)


def test_all_profiles_generate_valid_ordered_traces():
    traces = [generate_trace(profile, profile, config()) for profile in PROFILE_NAMES]
    assert all(trace.conditions[0].timestamp_seconds == 0 for trace in traces)
    for trace in traces:
        timestamps = [condition.timestamp_seconds for condition in trace.conditions]
        assert timestamps == sorted(timestamps)
        assert all(0 <= condition.packet_loss_percent <= 100 for condition in trace.conditions)


def test_seed_reproducibility_and_random_variation():
    first = generate_trace("random", "one", config(42))
    second = generate_trace("random", "two", config(42))
    different = generate_trace("random", "three", config(43))
    assert first.conditions == second.conditions
    assert first.conditions != different.conditions


def test_profile_specific_behavior():
    sudden = generate_trace("sudden_drop", "drop", config())
    recovery = generate_trace("recovery", "recovery", config())
    congestion = generate_trace("congestion", "congestion", config())
    assert min(c.bandwidth_mbps for c in sudden.conditions) < sudden.conditions[0].bandwidth_mbps
    assert recovery.conditions[-1].bandwidth_mbps > min(c.bandwidth_mbps for c in recovery.conditions)
    assert max(c.delay_ms for c in congestion.conditions) > congestion.conditions[0].delay_ms
    assert max(c.packet_loss_percent for c in congestion.conditions) > 0


def test_replay_applies_in_order_and_closes():
    applied = []

    class Applier:
        def apply(self, condition):
            applied.append(condition)

        def close(self):
            applied.append("closed")

    trace = NetworkTrace(
        "trace",
        "stable",
        None,
        3,
        1,
        (NetworkCondition(0, 10), NetworkCondition(1, 5), NetworkCondition(2, 10)),
    )
    clock_values = iter([0, 0, 1, 1, 2, 2])
    ReplayController(Applier(), clock=lambda: next(clock_values), sleeper=lambda _: None).run(trace)
    assert [condition.bandwidth_mbps for condition in applied[:-1]] == [10, 5, 10]
    assert applied[-1] == "closed"
