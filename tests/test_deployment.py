import concurrent.futures

from backend.app.main import app
from backend.app.services.dqn_controller import DQNController, DecisionRequest
from backend.app.services.playback_metrics import SegmentPlayback
from backend.app.services.session_service import SessionService


def test_deployment_routes_are_exposed():
    paths = set(app.openapi()["paths"])
    assert {"/health", "/ready", "/api/sessions", "/api/abr/decision"} <= paths


def test_session_json_and_csv_lifecycle(tmp_path):
    service = SessionService(tmp_path)
    session = service.create("demo", "manual", "checkpoint.pt")
    updated = service.add_segment(
        session.session_id,
        SegmentPlayback(0, 0.3, 1.0, 0.5, 0.1),
    )

    assert updated.as_dict()["summary"]["average_bitrate_mbps"] == 0.3
    assert service.get(session.session_id).segments[0].segment_index == 0
    assert service.export_csv(session.session_id).is_file()


def test_controller_fallback_is_safe_for_concurrent_requests(tmp_path):
    controller = DQNController(tmp_path / "missing.pt")
    request = DecisionRequest(4, (1, 2, 3, 4, 5), 0.7, 1, 8)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(controller.decide, [request] * 8))

    assert all(result["fallback"] for result in results)
    assert all(result["action"] == 0 for result in results)
