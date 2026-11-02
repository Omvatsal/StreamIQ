"""Monotonic trace replay with an injectable network-environment adapter."""

from dataclasses import dataclass
import time
from typing import Callable, Optional, Protocol

from .models import NetworkCondition, NetworkTrace


class ConditionApplier(Protocol):
    def apply(self, condition: NetworkCondition) -> None: ...

    def close(self) -> None: ...


@dataclass
class ReplayController:
    applier: ConditionApplier
    clock: Callable[[], float] = time.monotonic
    sleeper: Callable[[float], None] = time.sleep

    def run(self, trace: NetworkTrace, stop: Optional[Callable[[], bool]] = None) -> None:
        started = self.clock()
        try:
            for condition in trace.conditions:
                if stop and stop():
                    break
                target = started + condition.timestamp_seconds
                remaining = target - self.clock()
                if remaining > 0:
                    self.sleeper(remaining)
                self.applier.apply(condition)
        finally:
            self.applier.close()


def replay_trace(trace: NetworkTrace, applier: ConditionApplier) -> None:
    ReplayController(applier).run(trace)
