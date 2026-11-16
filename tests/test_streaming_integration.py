from pathlib import Path

import pytest

from backend.app.services.dqn_controller import (
    BITRATES_MBPS,
    DQNController,
    DecisionRequest,
)
from backend.app.services.playback_metrics import SegmentPlayback, summarize_playback
from backend.app.services.streaming_service import StreamingService
from backend.app.services.video_processor import VideoProcessor


def test_decision_state_matches_training_shape_and_fallback(tmp_path):
    request = DecisionRequest(4, (1, 2, 3, 4, 5), 0.7, 0.5, 8)
    assert request.state().shape == (9,)
    decision = DQNController(tmp_path / "missing.pt").decide(request)
    assert decision["fallback"] is True
    assert decision["bitrate_mbps"] == BITRATES_MBPS[0]


def test_controller_loads_existing_checkpoint():
    checkpoint = Path("experiments/results/dqn_sample/model_final.pt")
    if not checkpoint.is_file():
        pytest.skip("sample checkpoint is unavailable")
    decision = DQNController(checkpoint).decide(
        DecisionRequest(4, (1, 2, 3, 4, 5), 0.7, 1, 8)
    )
    assert decision["fallback"] is False
    assert 0 <= decision["action"] < 5


def test_processor_builds_aligned_dash_command(tmp_path):
    command = VideoProcessor().build_command("input.mp4", tmp_path, 4)
    assert "-f" in command
    assert "dash" in command
    assert "-seg_duration" in command
    assert "4" in command
    assert "manifest.mpd" in command[-1]


def test_streaming_service_rejects_path_traversal(tmp_path):
    service = StreamingService(tmp_path)
    with pytest.raises(ValueError):
        service.manifest_path("../secret")


def test_playback_metrics_report_switch_magnitude():
    metrics = summarize_playback(
        [
            SegmentPlayback(0, 0.3, 1, 1, 4),
            SegmentPlayback(1, 2.5, 2, 1, 7),
            SegmentPlayback(2, 0.7, 2, 1, 9),
        ]
    )
    assert metrics["bitrate_switches"] == 2
    assert metrics["switching_magnitude_mbps"] == pytest.approx(4.0)
