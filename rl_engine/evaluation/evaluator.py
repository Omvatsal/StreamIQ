"""Evaluate policies on identical traces."""

from collections.abc import Callable, Sequence
from typing import Any

from rl_engine.environment.abr_env import ABREnvironment
from .metrics import EvaluationMetrics, summarize


def evaluate_policy(
    environments: Sequence[ABREnvironment],
    policy: Callable[[Any, ABREnvironment], int],
) -> EvaluationMetrics:
    episode_metrics = []
    for environment in environments:
        state, _ = environment.reset(seed=environment.trace.seed)
        done = False
        while not done:
            state, _, terminated, truncated, _ = environment.step(policy(state, environment))
            done = terminated or truncated
        episode_metrics.append(environment.metrics)
    combined = episode_metrics[0]
    for metrics in episode_metrics[1:]:
        combined.segments.extend(metrics.segments)
    return summarize(combined)
