"""The monitoring loop, alert history and HTTP API, driven by fake cameras."""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from collections.abc import Callable, Iterator

import cv2 as cv
import numpy as np
import pytest
from flask.testing import FlaskClient

from conftest import ScriptedDetector, synthetic_face
from dss.config import DetectionThresholds, Settings
from dss.detection.pipeline import FatiguePipeline
from dss.domain import (
    AlertLevel,
    AlertReason,
    Assessment,
    Frame,
    FrameAnalysis,
    FrameMetrics,
    analysis_to_dict,
)
from dss.service.alerts import AlertLog
from dss.service.api import create_app
from dss.service.monitor import LatestValue, MonitorService

FRAME_INTERVAL_S = 0.002


class FakeCamera:
    """Produces blank frames, optionally failing to open or dying after some reads."""

    def __init__(self, can_open: bool = True, frames: int | None = None) -> None:
        self.can_open = can_open
        self.remaining = frames
        self.closed = False

    def open(self) -> bool:
        return self.can_open

    def read(self) -> Frame | None:
        if self.remaining is not None:
            if self.remaining == 0:
                return None
            self.remaining -= 1
        time.sleep(FRAME_INTERVAL_S)
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def wait_until(condition: Callable[[], bool], timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met in time")


def assessment(level: AlertLevel, *reasons: AlertReason) -> Assessment:
    return Assessment(level, reasons, None, 0.0, 0, True)


def pipeline_factory(
    detector: ScriptedDetector | None = None,
) -> Callable[[], FatiguePipeline]:
    thresholds = DetectionThresholds(_env_file=None, pose_calibration_s=0.0)  # type: ignore[call-arg]
    return lambda: FatiguePipeline(detector or ScriptedDetector(synthetic_face()), thresholds)


def settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type,call-arg]


@pytest.fixture
def monitor() -> Iterator[MonitorService]:
    service = MonitorService(FakeCamera, pipeline_factory(), retry_s=0.01)
    service.start()
    wait_until(lambda: service.healthy)
    yield service
    service.stop()


@pytest.fixture
def client(monitor: MonitorService) -> FlaskClient:
    return create_app(settings(), monitor).test_client()


# ---- LatestValue ---------------------------------------------------------------------------


def test_every_reader_sees_the_latest_value() -> None:
    """Regression: readers used to compete for frames on a shared queue."""
    value: LatestValue[int] = LatestValue()
    value.publish(1)
    first = value.wait_newer(0, timeout=0.1)
    second = value.wait_newer(0, timeout=0.1)
    assert first == (1, 1)
    assert second == (1, 1)


def test_readers_skip_stale_values() -> None:
    value: LatestValue[int] = LatestValue()
    for number in range(5):
        value.publish(number)
    assert value.wait_newer(0, timeout=0.1) == (5, 4)


def test_wait_times_out_without_a_new_value() -> None:
    value: LatestValue[int] = LatestValue()
    value.publish(1)
    assert value.wait_newer(1, timeout=0.01) is None


def test_waiting_reader_is_woken_by_publish() -> None:
    value: LatestValue[int] = LatestValue()
    results: list[tuple[int, int] | None] = []
    reader = threading.Thread(target=lambda: results.append(value.wait_newer(0, timeout=2)))
    reader.start()
    value.publish(7)
    reader.join(timeout=2)
    assert results == [(1, 7)]


# ---- MonitorService ------------------------------------------------------------------------


def test_frames_keep_flowing_with_no_face() -> None:
    """Regression: no-face frames used to be dropped, freezing every stream."""
    service = MonitorService(FakeCamera, pipeline_factory(ScriptedDetector(None)))
    service.start()
    try:
        wait_until(lambda: service.healthy)
        latest = service.frames.latest()
        assert latest is not None
        assert not latest.analysis.metrics.face_detected
    finally:
        service.stop()


def test_camera_failure_is_retried() -> None:
    cameras = iter([FakeCamera(can_open=False), FakeCamera(frames=2), FakeCamera()])
    service = MonitorService(lambda: next(cameras), pipeline_factory(), retry_s=0.01)
    service.start()
    try:
        wait_until(lambda: service.healthy)
        assert service.camera_connected
    finally:
        service.stop()


def test_one_bad_frame_does_not_stop_the_loop() -> None:
    class Flaky(ScriptedDetector):
        calls = 0

        def detect(self, frame: Frame) -> np.ndarray | None:
            Flaky.calls += 1
            if Flaky.calls == 3:
                raise RuntimeError("bad frame")
            return synthetic_face()

    service = MonitorService(FakeCamera, pipeline_factory(Flaky()))
    service.start()
    try:
        wait_until(lambda: Flaky.calls > 10)
        assert service.running
    finally:
        service.stop()


def test_file_source_stops_at_the_end() -> None:
    service = MonitorService(lambda: FakeCamera(frames=3), pipeline_factory(), stop_at_end=True)
    service.start()
    wait_until(lambda: not service.running)
    assert service.frames.latest() is not None


def test_frame_rate_survives_a_coarse_clock(frame: Frame) -> None:
    """Regression: Windows' 15 ms tick gave two frames the same time, reporting 9878 fps."""
    # 30 fps in total, delivered in pairs that share a clock tick. Each frame reads the clock twice.
    ticks = [pair * (2 / 30) for pair in range(40) for _ in range(4)]
    clock = iter(ticks)
    service = MonitorService(FakeCamera, pipeline_factory(), clock=lambda: next(clock))
    pipeline = pipeline_factory()()
    for _ in range(len(ticks) // 2):
        service._process(frame, pipeline)
    assert service.fps == pytest.approx(30, abs=2)  # the window can be one tick short


def test_pipeline_is_released_on_stop() -> None:
    detector = ScriptedDetector(synthetic_face())
    service = MonitorService(FakeCamera, pipeline_factory(detector))
    service.start()
    wait_until(lambda: service.healthy)
    service.stop()
    assert detector.closed
    assert not service.running


# ---- AlertLog ------------------------------------------------------------------------------


def test_a_long_alert_is_one_entry() -> None:
    log = AlertLog()
    critical = assessment(AlertLevel.CRITICAL, AlertReason.PROLONGED_EYE_CLOSURE)
    events = [log.observe(critical) for _ in range(100)]
    assert sum(event is not None for event in events) == 1


def test_recovery_is_recorded_newest_first() -> None:
    log = AlertLog()
    log.observe(assessment(AlertLevel.WARNING, AlertReason.LOOKING_AWAY))
    log.observe(assessment(AlertLevel.NORMAL))
    assert [event.level for event in log.recent(10)] == [AlertLevel.NORMAL, AlertLevel.WARNING]


def test_history_is_bounded() -> None:
    log = AlertLog(size=3)
    for index in range(10):
        reason = list(AlertReason)[index % 2]
        log.observe(assessment(AlertLevel.WARNING, reason))
    assert len(log.recent(100)) == 3


# ---- API -----------------------------------------------------------------------------------


def test_health_is_up(client: FlaskClient) -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_ready_reflects_frames_flowing(client: FlaskClient) -> None:
    assert client.get("/api/v1/ready").status_code == 200


def test_not_ready_without_a_camera() -> None:
    service = MonitorService(lambda: FakeCamera(can_open=False), pipeline_factory(), retry_s=0.01)
    service.start()
    try:
        client = create_app(settings(), service).test_client()
        response = client.get("/api/v1/ready")
        assert response.status_code == 503
        assert response.get_json()["checks"]["camera_connected"] is False
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/v1/status").get_json()["level"] == "unknown"
    finally:
        service.stop()


def test_status_reports_the_driver(client: FlaskClient) -> None:
    body = client.get("/api/v1/status").get_json()
    assert body["level"] == "normal"
    assert body["face_detected"] is True
    assert body["metrics"]["ear_left"] == pytest.approx(0.30)
    assert body["fatigue"]["eye_closure_seconds"] == 0
    assert body["service"]["camera_connected"] is True


def test_alerts_endpoint_lists_changes(monitor: MonitorService, client: FlaskClient) -> None:
    monitor.stop()  # otherwise the live loop would record the next normal frame on top
    monitor.alerts.observe(assessment(AlertLevel.CRITICAL, AlertReason.HEAD_NODDING))
    alert = client.get("/api/v1/alerts?limit=1").get_json()["alerts"][0]
    assert alert["level"] == "critical"
    assert alert["reasons"] == ["head_nodding"]


def first_chunk(client: FlaskClient, url: str) -> bytes:
    response = client.get(url, buffered=False)
    assert response.status_code == 200
    assert response.mimetype == "multipart/x-mixed-replace"
    chunk = next(iter(response.response))
    response.close()
    return chunk


@pytest.mark.parametrize(
    "url",
    [
        "/api/v1/streams/annotated",
        "/api/v1/streams/original",
        "/api/v1/streams/head_tilt",
        "/video_feed/yawn",
        "/video_feed/eye_closure",
    ],
)
def test_streams_deliver_decodable_jpeg(client: FlaskClient, url: str) -> None:
    """Regression: a tuple used to be handed to imencode, so no stream ever produced a frame."""
    chunk = first_chunk(client, url)
    jpeg = chunk.split(b"\r\n\r\n", 1)[1]
    image = cv.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv.IMREAD_COLOR)
    assert image.shape == (480, 640, 3)


def test_two_streams_run_at_once(client: FlaskClient) -> None:
    """Regression: concurrent viewers used to steal frames from each other."""
    first = client.get("/api/v1/streams/yawn", buffered=False)
    second = client.get("/api/v1/streams/eye_closure", buffered=False)
    first_frames, second_frames = iter(first.response), iter(second.response)
    for _ in range(3):
        assert next(first_frames).startswith(b"--frame")
        assert next(second_frames).startswith(b"--frame")
    first.close()
    second.close()


def test_unknown_view_is_404(client: FlaskClient) -> None:
    response = client.get("/api/v1/streams/nope")
    assert response.status_code == 404
    assert "annotated" in response.get_json()["message"]


def test_stream_slots_are_limited_and_released(monitor: MonitorService) -> None:
    client = create_app(settings(server_threads=3), monitor).test_client()
    held = client.get("/api/v1/streams/original", buffered=False)
    refused = client.get("/api/v1/streams/original", buffered=False)
    assert refused.status_code == 503
    assert refused.headers["Retry-After"]

    held.close()
    again = client.get("/api/v1/streams/original", buffered=False)
    assert again.status_code == 200
    again.close()


def test_streaming_can_be_switched_off(monitor: MonitorService) -> None:
    client = create_app(settings(stream_enabled=False), monitor).test_client()
    assert client.get("/api/v1/streams/annotated").status_code == 404
    assert client.get("/api/v1/status").status_code == 200
    assert b"disabled by configuration" in client.get("/").data


def test_dashboard_links_only_to_real_routes(client: FlaskClient) -> None:
    """Regression: the old index page linked to three endpoints that did not exist."""
    html = client.get("/").get_data(as_text=True)
    urls = re.findall(r'(?:src|href)="(/[^"]+)"', html)
    assert urls
    for url in urls:
        response = client.get(url, buffered=False)
        assert response.status_code == 200, url
        response.close()


def test_metrics_are_exposed(client: FlaskClient) -> None:
    text = client.get("/metrics").get_data(as_text=True)
    assert "dss_frames_processed_total" in text
    assert "dss_camera_connected" in text


def test_security_headers(client: FlaskClient) -> None:
    headers = client.get("/api/v1/status").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Cache-Control"] == "no-store"
    assert "default-src 'self'" in headers["Content-Security-Policy"]


def test_cors_is_closed_by_default(client: FlaskClient) -> None:
    response = client.get("/api/v1/status", headers={"Origin": "https://evil.example"})
    assert "Access-Control-Allow-Origin" not in response.headers


def test_cors_allows_only_listed_origins(monitor: MonitorService) -> None:
    allowed = "https://console.example"
    client = create_app(settings(cors_origins=[allowed]), monitor).test_client()
    good = client.get("/api/v1/status", headers={"Origin": allowed})
    bad = client.get("/api/v1/status", headers={"Origin": "https://evil.example"})
    assert good.headers["Access-Control-Allow-Origin"] == allowed
    assert "Access-Control-Allow-Origin" not in bad.headers


# ---- Authentication ------------------------------------------------------------------------


@pytest.fixture
def secured(monitor: MonitorService) -> FlaskClient:
    return create_app(settings(api_key="test-key"), monitor).test_client()


@pytest.mark.parametrize("url", ["/", "/api/v1/status", "/api/v1/alerts", "/metrics"])
def test_key_is_required(secured: FlaskClient, url: str) -> None:
    response = secured.get(url)
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"].startswith("Basic")


def test_video_requires_a_key(secured: FlaskClient) -> None:
    assert secured.get("/api/v1/streams/annotated").status_code == 401
    assert secured.get("/video_feed/yawn").status_code == 401


@pytest.mark.parametrize("url", ["/api/v1/health", "/api/v1/ready"])
def test_probes_stay_open(secured: FlaskClient, url: str) -> None:
    assert secured.get(url).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {"X-API-Key": "test-key"},
        {"Authorization": "Bearer test-key"},
        {"Authorization": "Basic " + base64.b64encode(b"driver:test-key").decode()},
    ],
)
def test_accepted_credentials(secured: FlaskClient, headers: dict[str, str]) -> None:
    assert secured.get("/api/v1/status", headers=headers).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {"X-API-Key": "wrong"},
        {"Authorization": "Bearer wrong"},
        {"Authorization": "Basic not-base64!"},
        {"Authorization": "Basic " + base64.b64encode(b"test-key").decode()},
    ],
)
def test_rejected_credentials(secured: FlaskClient, headers: dict[str, str]) -> None:
    assert secured.get("/api/v1/status", headers=headers).status_code == 401


# ---- Serialisation -------------------------------------------------------------------------


def test_status_for_a_frame_without_a_face_is_valid_json() -> None:
    analysis = FrameAnalysis(FrameMetrics(0.0, False), assessment(AlertLevel.NORMAL), None)
    body = json.loads(json.dumps(analysis_to_dict(analysis)))
    assert body["face_detected"] is False
    assert body["metrics"]["head_pose"] is None
    assert body["metrics"]["ear_left"] is None


def test_processed_frames_never_leave_the_process_through_status(client: FlaskClient) -> None:
    assert "frame" not in client.get("/api/v1/status").get_json()
