"""The HTTP API and operator dashboard."""

from __future__ import annotations

import base64
import binascii
import hmac
import logging
import threading
from collections.abc import Iterator

import cv2 as cv
from flask import Blueprint, Flask, Response, jsonify, render_template, request
from flask_cors import CORS
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from werkzeug.exceptions import HTTPException

from dss import __version__
from dss.config import Settings
from dss.domain import analysis_to_dict
from dss.service.monitor import MonitorService
from dss.vision.overlay import View, render

logger = logging.getLogger(__name__)

_BOUNDARY = "frame"
_PUBLIC_ENDPOINTS = {"api.health", "api.ready"}
_MAX_ALERT_LIMIT = 200
_DEFAULT_ALERT_LIMIT = 50
_RESERVED_THREADS = 2
_WAIT_FOR_FRAME_S = 1.0


def _presented_key() -> str | None:
    """The API key from a Bearer token, ``X-API-Key`` header or Basic password.

    Basic is accepted because browsers send it automatically, including for the ``<img>`` tags
    that display MJPEG streams, which cannot carry custom headers.
    """
    header = request.headers.get("Authorization", "")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer":
        return value.strip()
    if scheme.lower() == "basic":
        try:
            return base64.b64decode(value, validate=True).decode().partition(":")[2]
        except (binascii.Error, UnicodeDecodeError):
            return None
    return request.headers.get("X-API-Key")


def _encode_stream(monitor: MonitorService, view: View, jpeg_quality: int) -> Iterator[bytes]:
    version = 0
    params = [cv.IMWRITE_JPEG_QUALITY, jpeg_quality]
    while monitor.running:
        update = monitor.frames.wait_newer(version, timeout=_WAIT_FOR_FRAME_S)
        if update is None:
            continue
        version, processed = update
        encoded, jpeg = cv.imencode(
            ".jpg", render(view, processed.frame, processed.analysis), params
        )
        if encoded:
            yield (
                f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\n\r\n".encode()
                + jpeg.tobytes()
                + b"\r\n"
            )


def create_app(settings: Settings, monitor: MonitorService) -> Flask:
    """Build the Flask app. It reads from ``monitor`` but does not start or stop it."""
    app = Flask(__name__)
    api = Blueprint("api", __name__, url_prefix="/api/v1")

    if settings.cors_origins:
        CORS(app, origins=settings.cors_origins)

    # Each MJPEG stream occupies a server thread for as long as it is open. Capping them keeps
    # threads free for health checks and status polling.
    stream_slots = threading.BoundedSemaphore(max(settings.server_threads - _RESERVED_THREADS, 1))

    @app.before_request
    def require_api_key() -> Response | None:
        if settings.api_key is None or request.endpoint in _PUBLIC_ENDPOINTS:
            return None
        presented = _presented_key()
        if presented is not None and hmac.compare_digest(
            presented.encode(), settings.api_key.encode()
        ):
            return None
        response = jsonify(error="unauthorized", message="A valid API key is required.")
        response.status_code = 401
        response.headers["WWW-Authenticate"] = 'Basic realm="DSS"'
        return response

    @app.after_request
    def add_security_headers(response: Response) -> Response:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        # Status and video describe a person at this moment: nothing should cache them.
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.errorhandler(HTTPException)
    def json_error(error: HTTPException) -> tuple[Response, int]:
        return jsonify(error=error.name, message=error.description), error.code or 500

    def stream(view_name: str) -> Response:
        if not settings.stream_enabled:
            return Response(status=404)
        try:
            view = View(view_name)
        except ValueError:
            valid = [item.value for item in View]
            response = jsonify(error="Not Found", message=f"Unknown view. Choose from {valid}.")
            response.status_code = 404
            return response

        if not stream_slots.acquire(blocking=False):
            response = jsonify(error="Service Unavailable", message="Too many open video streams.")
            response.status_code = 503
            response.headers["Retry-After"] = "5"
            return response

        response = Response(
            _encode_stream(monitor, view, settings.jpeg_quality),
            mimetype=f"multipart/x-mixed-replace; boundary={_BOUNDARY}",
        )
        response.call_on_close(stream_slots.release)
        return response

    @api.get("/health")
    def health() -> Response:
        """Liveness: the process is up. Does not depend on the camera."""
        return jsonify(status="ok", version=__version__)

    @api.get("/ready")
    def ready() -> tuple[Response, int]:
        """Readiness: frames are flowing, so results can be trusted."""
        is_ready = monitor.healthy
        body = jsonify(
            status="ready" if is_ready else "not_ready",
            checks={
                "monitor_running": monitor.running,
                "camera_connected": monitor.camera_connected,
            },
        )
        return body, 200 if is_ready else 503

    @api.get("/status")
    def status() -> tuple[Response, int]:
        """The driver's current state."""
        latest = monitor.frames.latest()
        service = {
            "camera_connected": monitor.camera_connected,
            "healthy": monitor.healthy,
            "fps": round(monitor.fps, 1),
        }
        if latest is None or not monitor.healthy:
            return jsonify(level="unknown", service=service), 200
        return jsonify(**analysis_to_dict(latest.analysis), service=service), 200

    @api.get("/alerts")
    def alerts() -> Response:
        """Recent alert level changes, newest first."""
        limit = request.args.get("limit", default=_DEFAULT_ALERT_LIMIT, type=int) or 0
        limit = min(max(limit, 1), _MAX_ALERT_LIMIT)
        return jsonify(alerts=[event.to_dict() for event in monitor.alerts.recent(limit)])

    @api.get("/streams/<view_name>")
    def video_stream(view_name: str) -> Response:
        """Live MJPEG video."""
        return stream(view_name)

    app.register_blueprint(api)

    @app.get("/video_feed/<view_name>")
    def legacy_video_feed(view_name: str) -> Response:
        """The original video URLs, kept so existing clients continue to work."""
        return stream(view_name)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(), content_type=CONTENT_TYPE_LATEST)

    @app.get("/")
    def dashboard() -> str:
        return render_template(
            "dashboard.html",
            version=__version__,
            views=[view.value for view in View],
            stream_enabled=settings.stream_enabled,
        )

    return app
