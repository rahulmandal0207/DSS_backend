"""Logging setup and Prometheus metrics."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

from prometheus_client import Counter, Gauge, Histogram

FRAMES_PROCESSED = Counter("dss_frames_processed_total", "Frames analysed by the pipeline.")
FRAME_ERRORS = Counter("dss_frame_errors_total", "Frames dropped because analysis raised.")
FRAME_SECONDS = Histogram(
    "dss_frame_processing_seconds",
    "Time spent analysing one frame.",
    buckets=(0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
)
ALERTS_RAISED = Counter("dss_alerts_total", "Alert level changes.", ["level"])
CAMERA_CONNECTED = Gauge("dss_camera_connected", "1 while frames are being read from the camera.")
ALERT_RANK = Gauge("dss_alert_rank", "Current alert level: 0 normal, 1 warning, 2 critical.")


class JsonFormatter(logging.Formatter):
    """One JSON object per line, for log shippers."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry)


def configure_logging(level: str, log_format: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter()
        if log_format == "json"
        else logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    logging.basicConfig(level=level, handlers=[handler], force=True)
