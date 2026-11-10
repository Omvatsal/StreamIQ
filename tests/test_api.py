"""Tests for the StreamIQ FastAPI backend."""

import pytest
from starlette.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_get_profiles():
    response = client.get("/api/traces/profiles")
    assert response.status_code == 200
    data = response.json()
    assert "profiles" in data
    profile_ids = [p["id"] for p in data["profiles"]]
    assert "sudden_drop" in profile_ids
    assert "oscillating" in profile_ids
    assert "congestion" in profile_ids


def test_generate_trace():
    payload = {
        "profile": "recovery",
        "duration_seconds": 20.0,
        "interval_seconds": 1.0,
        "seed": 42,
    }
    response = client.post("/api/traces/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["profile"] == "recovery"
    assert len(data["conditions"]) > 0


def test_abr_predict():
    payload = {
        "buffer_occupancy_seconds": 15.2,
        "throughput_history_mbps": [2.1, 2.4, 3.0, 3.1, 2.8],
        "previous_bitrate_mbps": 1.2,
        "previous_download_duration_seconds": 1.8,
        "remaining_segments": 8,
    }
    response = client.post("/api/abr/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["selected_action"] <= 4
    assert data["quality_name"] in ["240p", "360p", "480p", "720p", "1080p"]
    assert len(data["q_values"]) == 5


def test_simulation_batch():
    payload = {
        "profile": "sudden_drop",
        "duration_seconds": 20.0,
        "interval_seconds": 1.0,
        "seed": 42,
        "policies": ["dqn", "buffer_based", "fixed_240p"],
    }
    response = client.post("/api/simulation/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "timeline" in data
    assert "summaries" in data
    assert "dqn" in data["summaries"]
    assert "buffer_based" in data["summaries"]
    assert "fixed_240p" in data["summaries"]
    assert data["winner_policy"] != ""


def test_dash_manifest():
    response = client.get("/api/video/manifest.mpd?duration=40")
    assert response.status_code == 200
    assert "MPD" in response.text
    assert "Representation" in response.text


def test_video_segment():
    response = client.get("/api/video/segments/0/1")
    assert response.status_code == 200
    assert response.headers["X-StreamIQ-Quality"] == "240p"
    assert len(response.content) > 0


def test_ui_and_assets():
    import re
    ui_resp = client.get("/ui/")
    assert ui_resp.status_code == 200
    assert "StreamIQ" in ui_resp.text
    css_match = re.search(r'href="\./(assets/[^"]+\.css)"', ui_resp.text)
    js_match = re.search(r'src="\./(assets/[^"]+\.js)"', ui_resp.text)
    assert css_match and js_match
    assert client.get(f"/{css_match.group(1)}").status_code == 200
    assert client.get(f"/{js_match.group(1)}").status_code == 200
    assert client.get(f"/ui/{css_match.group(1)}").status_code == 200
    assert client.get(f"/ui/{js_match.group(1)}").status_code == 200


