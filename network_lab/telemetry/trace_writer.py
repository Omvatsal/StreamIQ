"""Incremental CSV trace output."""

import csv
from pathlib import Path
from typing import TextIO

from .metrics import Measurement


TRACE_FIELDS = (
    "experiment_id",
    "trace_id",
    "timestamp",
    "configured_bandwidth_mbps",
    "configured_delay_ms",
    "configured_packet_loss_percent",
    "throughput_mbps",
    "rtt_ms",
    "observed_packet_loss_percent",
    "download_duration_seconds",
)


class CsvTraceWriter:
    """Write validated measurements immediately so partial traces survive errors."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._file: TextIO | None = None
        self._writer: csv.DictWriter | None = None

    def open(self) -> None:
        if self._file is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=TRACE_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def write(self, measurement: Measurement) -> None:
        if self._writer is None:
            self.open()
        assert self._writer is not None
        assert self._file is not None
        self._writer.writerow(measurement.as_csv_row())
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self) -> "CsvTraceWriter":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
