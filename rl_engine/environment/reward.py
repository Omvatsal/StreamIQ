"""Configurable ABR reward calculation."""

from dataclasses import dataclass
from typing import Literal


RewardObjective = Literal["legacy", "qoe_aligned", "evaluation_aligned"]


@dataclass(frozen=True)
class RewardConfig:
    quality_weight: float = 1.0
    rebuffer_penalty: float = 4.0
    switch_penalty: float = 0.5
    objective: RewardObjective = "legacy"

    def __post_init__(self) -> None:
        if self.objective not in {"legacy", "qoe_aligned", "evaluation_aligned"}:
            raise ValueError(f"unsupported reward objective: {self.objective}")


def calculate_reward(
    quality_index: int,
    rebuffer_seconds: float,
    previous_bitrate_mbps: float | None,
    current_bitrate_mbps: float,
    config: RewardConfig,
) -> float:
    switch = (
        0.0
        if previous_bitrate_mbps is None
        else abs(current_bitrate_mbps - previous_bitrate_mbps)
    )
    if config.objective == "qoe_aligned":
        return current_bitrate_mbps - 4.0 * rebuffer_seconds - 0.5 * switch
    if config.objective == "evaluation_aligned":
        return current_bitrate_mbps - 40.0 * rebuffer_seconds - 5.0 * switch
    return (
        config.quality_weight * quality_index
        - config.rebuffer_penalty * rebuffer_seconds
        - config.switch_penalty * switch
    )
