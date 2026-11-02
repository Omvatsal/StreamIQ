import csv
import time
from datetime import timezone

import pytest

from network_lab.telemetry.collector import MeasurementCollector
from network_lab.telemetry.metrics import Measurement, utc_now
from network_lab.telemetry.probes import MininetSampler
from network_lab.telemetry.trace_writer import CsvTraceWriter


def test_measurement_rejects_invalid_values():
    with pytest.raises(ValueError):
        Measurement("x", utc_now(), packet_loss_percent=101)
    with pytest.raises(ValueError):
        Measurement("x", utc_now().replace(tzinfo=None))


def test_writer_preserves_missing_metrics(tmp_path):
    path = tmp_path / "trace.csv"
    with CsvTraceWriter(path) as writer:
        writer.write(Measurement("exp", utc_now(), rtt_ms=2.5))
    with path.open(newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["experiment_id"] == "exp"
    assert row["throughput_mbps"] == ""
    assert row["rtt_ms"] == "2.5"


def test_collector_writes_ordered_samples_and_stops(tmp_path):
    samples = []

    def sampler(timestamp):
        samples.append(timestamp)
        return Measurement("exp", timestamp, throughput_mbps=1)

    collector = MeasurementCollector(
        "exp", 0.02, sampler, CsvTraceWriter(tmp_path / "trace.csv")
    )
    collector.start()
    time.sleep(0.08)
    collector.stop(timeout=1)
    assert len(samples) >= 2
    assert samples == sorted(samples)
    assert not collector.errors


def test_sampler_keeps_failed_probe_metrics_missing():
    def failing_runner(command, timeout):
        raise RuntimeError("probe unavailable")

    sample = MininetSampler(
        "exp", "10.0.0.1", failing_runner, lambda: None
    )(utc_now())
    assert sample.configured_bandwidth_mbps is None
    assert sample.throughput_mbps is None
    assert sample.rtt_ms is None
    assert sample.packet_loss_percent is None
