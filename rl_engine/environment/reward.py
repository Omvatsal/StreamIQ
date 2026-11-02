"""Configurable ABR reward calculation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardConfig:
    quality_weight: float = 1.0
    rebuffer_penalty: float = 4.0
    switch_penalty: float = 0.5


def calculate_reward(
    quality_index: int,
    rebuffer_seconds: float,
    previous_quality_index: int | None,
    config: RewardConfig,
) -> float:
    switch = (
        0.0
        if previous_quality_index is None
        else abs(quality_index - previous_quality_index)
    )
    return (
        config.quality_weight * quality_index
        - config.rebuffer_penalty * rebuffer_seconds
        - config.switch_penalty * switch
    )
