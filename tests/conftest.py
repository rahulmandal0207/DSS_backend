"""Shared test helpers: a synthetic face and a scriptable landmark detector."""

from __future__ import annotations

import numpy as np
import pytest

from dss.config import DetectionThresholds
from dss.domain import Frame, FrameMetrics, HeadPose, Landmarks
from dss.vision.geometry import CHIN, FOREHEAD, INNER_LIPS, LEFT_EYE, MOUTH_CORNERS, RIGHT_EYE

LANDMARK_COUNT = 478
EYE_WIDTH = 60.0


def synthetic_face(
    width: int = 640,
    height: int = 480,
    ear: float = 0.30,
    ear_right: float | None = None,
    mar: float = 0.05,
) -> Landmarks:
    """Landmarks in pixels for a flat face of fixed size, centred and looking at the camera."""
    cx, cy = width / 2.0, height / 2.0
    points = np.full((LANDMARK_COUNT, 3), (cx, cy, 0.0), dtype=np.float64)

    def place_eye(ids: tuple[int, ...], offset: float, ratio: float) -> None:
        corner_a, top_a, top_b, corner_b, bottom_b, bottom_a = ids
        half_gap = ratio * EYE_WIDTH / 2.0
        points[corner_a] = (cx + offset - EYE_WIDTH / 2, cy, 0)
        points[corner_b] = (cx + offset + EYE_WIDTH / 2, cy, 0)
        points[top_a] = (cx + offset - 15, cy - half_gap, 0)
        points[bottom_a] = (cx + offset - 15, cy + half_gap, 0)
        points[top_b] = (cx + offset + 15, cy - half_gap, 0)
        points[bottom_b] = (cx + offset + 15, cy + half_gap, 0)

    place_eye(RIGHT_EYE, -80, ear_right if ear_right is not None else ear)
    place_eye(LEFT_EYE, 80, ear)

    mouth_width = 100.0
    left, right = MOUTH_CORNERS
    upper, lower = INNER_LIPS
    points[left] = (cx - mouth_width / 2, cy + 120, 0)
    points[right] = (cx + mouth_width / 2, cy + 120, 0)
    points[upper] = (cx, cy + 120 - mar * mouth_width / 2, 0)
    points[lower] = (cx, cy + 120 + mar * mouth_width / 2, 0)
    points[FOREHEAD] = (cx, cy - 100, 0)
    points[CHIN] = (cx, cy + 180, 0)
    return points


class ScriptedDetector:
    """Returns queued landmarks, so a test can play out a scenario."""

    def __init__(self, default: Landmarks | None = None) -> None:
        self.default = default
        self.closed = False

    def detect(self, frame: Frame) -> Landmarks | None:
        return self.default

    def close(self) -> None:
        self.closed = True


def metrics_at(
    timestamp: float,
    ear: float = 0.30,
    mar: float = 0.05,
    pose: HeadPose | None = None,
    face: bool = True,
) -> FrameMetrics:
    if not face:
        return FrameMetrics(timestamp=timestamp, face_detected=False)
    return FrameMetrics(
        timestamp=timestamp,
        face_detected=True,
        ear_left=ear,
        ear_right=ear,
        mar=mar,
        pose=pose or HeadPose(0.0, 0.0, 0.0),
    )


def seconds(start: float, end: float, fps: int = 30) -> list[float]:
    """Timestamps for a span of video."""
    return list(np.arange(start, end, 1.0 / fps))


@pytest.fixture
def thresholds() -> DetectionThresholds:
    return DetectionThresholds(_env_file=None, pose_calibration_s=0.0)  # type: ignore[call-arg]


@pytest.fixture
def frame() -> Frame:
    return np.zeros((480, 640, 3), dtype=np.uint8)
