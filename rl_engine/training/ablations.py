"""Retrained, controlled ablation configurations."""

from dataclasses import dataclass

from rl_engine.environment.reward import RewardConfig


@dataclass(frozen=True)
class AblationConfig:
    name: str
    reward_config: RewardConfig = RewardConfig()
    throughput_history_length: int = 5
    include_previous_bitrate: bool = True


ABLATIONS = (
    AblationConfig("full"),
    AblationConfig("without_switch_penalty", RewardConfig(switch_penalty=0.0)),
    AblationConfig("without_rebuffer_penalty", RewardConfig(rebuffer_penalty=0.0)),
    AblationConfig("without_throughput_history", throughput_history_length=0),
    AblationConfig("without_previous_bitrate", include_previous_bitrate=False),
    AblationConfig("high_switch_penalty", RewardConfig(switch_penalty=1.0)),
)
