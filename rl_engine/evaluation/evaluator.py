"""Evaluate policies on identical traces."""

from collections.abc import Callable, Sequence
from typing import Any

from rl_engine.environment.abr_env import ABREnvironment
from .metrics import EvaluationMetrics, summarize


def evaluate_policy_per_trace(
    environments: Sequence[ABREnvironment],
    policy: Callable[[Any, ABREnvironment], int],
) -> list[EvaluationMetrics]:
    results = []
    for environment in environments:
        state, _ = environment.reset(seed=environment.trace.seed)
        done = False
        while not done:
            state, _, terminated, truncated, _ = environment.step(
                policy(state, environment)
            )
            done = terminated or truncated
        results.append(summarize(environment.metrics))
    return results


def evaluate_policy(
    environments: Sequence[ABREnvironment],
    policy: Callable[[Any, ABREnvironment], int],
) -> EvaluationMetrics:
    per_trace = evaluate_policy_per_trace(environments, policy)
    if not per_trace:
        raise ValueError("at least one environment is required")
    return EvaluationMetrics(
        **{
            field: sum(getattr(item, field) for item in per_trace) / len(per_trace)
            for field in EvaluationMetrics.__dataclass_fields__
            if field not in {"bitrate_switches", "upward_switches", "downward_switches"}
        },
        bitrate_switches=round(sum(item.bitrate_switches for item in per_trace) / len(per_trace)),
        upward_switches=round(sum(item.upward_switches for item in per_trace) / len(per_trace)),
        downward_switches=round(sum(item.downward_switches for item in per_trace) / len(per_trace)),
    )
