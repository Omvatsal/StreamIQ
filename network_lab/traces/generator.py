"""Trace generation with reproducible profile implementations."""

import random
from typing import Callable

from .models import NetworkCondition, NetworkTrace, TraceConfig

PROFILE_NAMES = ("stable", "random", "oscillating", "sudden_drop", "recovery", "congestion")
ProfileGenerator = Callable[[TraceConfig, random.Random], list[NetworkCondition]]


def _times(config: TraceConfig) -> list[float]:
    count = int((config.duration_seconds - 1e-12) // config.interval_seconds) + 1
    return [round(index * config.interval_seconds, 10) for index in range(count)]


def _stable(config: TraceConfig, rng: random.Random) -> list[NetworkCondition]:
    return [
        NetworkCondition(t, max(0.01, config.bandwidth_mbps + rng.uniform(-0.02, 0.02) * config.bandwidth_mbps), config.delay_ms)
        for t in _times(config)
    ]


def _random(config: TraceConfig, rng: random.Random) -> list[NetworkCondition]:
    conditions = []
    times = _times(config)
    index = 0
    while index < len(times):
        bandwidth = rng.uniform(config.min_bandwidth_mbps, config.max_bandwidth_mbps)
        hold = rng.randint(1, max(1, min(5, len(times) - index)))
        conditions.extend(
            NetworkCondition(times[index + offset], bandwidth, config.delay_ms)
            for offset in range(hold)
        )
        index += hold
    return conditions


def _oscillating(config: TraceConfig, rng: random.Random) -> list[NetworkCondition]:
    import math
    period = max(config.interval_seconds * 2, min(config.duration_seconds, 10.0))
    amplitude = (config.max_bandwidth_mbps - config.min_bandwidth_mbps) / 2
    midpoint = (config.max_bandwidth_mbps + config.min_bandwidth_mbps) / 2
    return [
        NetworkCondition(t, max(0.01, midpoint + amplitude * math.sin(2 * math.pi * t / period) + rng.uniform(-0.02, 0.02) * amplitude), config.delay_ms)
        for t in _times(config)
    ]


def _sudden_drop(config: TraceConfig, rng: random.Random) -> list[NetworkCondition]:
    drop_at = config.duration_seconds * 0.35
    restore_at = config.duration_seconds * 0.75
    degraded = max(config.min_bandwidth_mbps, config.bandwidth_mbps * 0.2)
    return [
        NetworkCondition(t, degraded if drop_at <= t < restore_at else config.bandwidth_mbps, config.delay_ms)
        for t in _times(config)
    ]


def _recovery(config: TraceConfig, rng: random.Random) -> list[NetworkCondition]:
    start = config.duration_seconds * 0.3
    degraded = max(config.min_bandwidth_mbps, config.bandwidth_mbps * 0.2)
    recovery_duration = max(config.interval_seconds, config.duration_seconds * 0.5)
    conditions = []
    for t in _times(config):
        if t < start:
            bandwidth = config.bandwidth_mbps
        else:
            progress = min(1.0, (t - start) / recovery_duration)
            bandwidth = degraded + (config.bandwidth_mbps - degraded) * progress
        conditions.append(NetworkCondition(t, bandwidth, config.delay_ms))
    return conditions


def _congestion(config: TraceConfig, rng: random.Random) -> list[NetworkCondition]:
    return [
        NetworkCondition(
            t,
            max(config.min_bandwidth_mbps, config.bandwidth_mbps * (0.35 + rng.uniform(-0.05, 0.05))),
            config.delay_ms + 35 + rng.uniform(-5, 5),
            min(100, 5 + rng.uniform(-1, 1)),
        )
        for t in _times(config)
    ]


_GENERATORS: dict[str, ProfileGenerator] = {
    "stable": _stable,
    "random": _random,
    "oscillating": _oscillating,
    "sudden_drop": _sudden_drop,
    "recovery": _recovery,
    "congestion": _congestion,
}


def generate_trace(profile: str, trace_id: str, config: TraceConfig) -> NetworkTrace:
    try:
        generator = _GENERATORS[profile]
    except KeyError as error:
        raise ValueError(f"unknown profile {profile!r}; choose from {PROFILE_NAMES}") from error
    conditions = tuple(generator(config, random.Random(config.seed)))
    return NetworkTrace(trace_id, profile, config.seed, config.duration_seconds, config.interval_seconds, conditions)
