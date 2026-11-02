"""Non-learning ABR policies."""

from typing import Callable


def fixed_bitrate(index: int) -> Callable:
    return lambda state, _env: index


def throughput_based(state, env) -> int:
    throughput = state[1:].max() if len(state) > 1 else 0
    eligible = [index for index, bitrate in enumerate(env.video.bitrates_mbps) if bitrate <= throughput * 0.85]
    return eligible[-1] if eligible else 0


def buffer_based(state, env) -> int:
    buffer_seconds = float(state[0])
    if buffer_seconds < 8:
        return 0
    if buffer_seconds < 16:
        return 1
    if buffer_seconds < 28:
        return 2
    if buffer_seconds < 44:
        return 3
    return 4
