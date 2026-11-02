"""Periodic collection independent of Mininet and its interactive CLI."""

from dataclasses import dataclass
import threading
import time
from datetime import datetime
from typing import Callable, Optional, Protocol

from .metrics import Measurement, utc_now
from .trace_writer import CsvTraceWriter


class NetworkSampler(Protocol):
    def __call__(self, timestamp: datetime) -> Measurement: ...


@dataclass
class CollectorError:
    timestamp: datetime
    error: Exception


class MeasurementCollector:
    """Run a sampler on a dedicated timer thread and write each completed sample.

    The sampler must use its own command/process interface. It never accesses a
    Mininet CLI object, which keeps CLI input and host command execution separate.
    """

    def __init__(
        self,
        experiment_id: str,
        interval_seconds: float,
        sampler: Callable[[datetime], Measurement],
        writer: CsvTraceWriter,
    ):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._experiment_id = experiment_id
        self._interval = interval_seconds
        self._sampler = sampler
        self._writer = writer
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.errors: list[CollectorError] = []

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("collector is already running")
        self._writer.open()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"telemetry-{self._experiment_id}", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError("telemetry collector did not stop in time")
        self._writer.close()

    def _run(self) -> None:
        next_sample = time.monotonic()
        while not self._stop.is_set():
            timestamp = utc_now()
            try:
                measurement = self._sampler(timestamp)
                if measurement.experiment_id != self._experiment_id:
                    raise ValueError("sampler returned a different experiment_id")
                self._writer.write(measurement)
            except Exception as error:
                self.errors.append(CollectorError(timestamp, error))
            next_sample += self._interval
            self._stop.wait(max(0, next_sample - time.monotonic()))
