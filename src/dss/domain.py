"""Value objects shared by the vision, decision and service layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

Landmarks = NDArray[np.float64]
"""Face landmarks in pixels, shape ``(n, 3)``. x is right, y is down, z is away from the camera."""

Frame = NDArray[np.uint8]
"""A BGR image, shape ``(height, width, 3)``."""


class AlertLevel(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"

    @property
    def rank(self) -> int:
        return list(AlertLevel).index(self)


class AlertReason(str, Enum):
    PROLONGED_EYE_CLOSURE = "prolonged_eye_closure"
    HIGH_PERCLOS = "high_perclos"
    FREQUENT_YAWNING = "frequent_yawning"
    HEAD_NODDING = "head_nodding"
    HEAD_TILTED = "head_tilted"
    LOOKING_AWAY = "looking_away"
    DRIVER_NOT_VISIBLE = "driver_not_visible"
    MODEL_DROWSY = "model_drowsy"
    CAMERA_FROZEN = "camera_frozen"


REASON_LEVELS: dict[AlertReason, AlertLevel] = {
    AlertReason.PROLONGED_EYE_CLOSURE: AlertLevel.CRITICAL,
    AlertReason.HEAD_NODDING: AlertLevel.CRITICAL,
    AlertReason.HIGH_PERCLOS: AlertLevel.WARNING,
    AlertReason.FREQUENT_YAWNING: AlertLevel.WARNING,
    AlertReason.HEAD_TILTED: AlertLevel.WARNING,
    AlertReason.LOOKING_AWAY: AlertLevel.WARNING,
    AlertReason.DRIVER_NOT_VISIBLE: AlertLevel.WARNING,
    AlertReason.MODEL_DROWSY: AlertLevel.WARNING,
    AlertReason.CAMERA_FROZEN: AlertLevel.WARNING,
}


@dataclass(frozen=True)
class HeadPose:
    """Head orientation in degrees. Positive pitch is chin-down, positive roll is clockwise."""

    pitch: float
    yaw: float
    roll: float


@dataclass(frozen=True)
class FrameMetrics:
    """What one frame measured. Numeric fields are None only when no face was found."""

    timestamp: float
    face_detected: bool
    ear_left: float | None = None
    ear_right: float | None = None
    mar: float | None = None
    pose: HeadPose | None = None
    frame_unchanged: bool = False

    @property
    def ear_mean(self) -> float | None:
        if self.ear_left is None or self.ear_right is None:
            return None
        return (self.ear_left + self.ear_right) / 2.0


@dataclass(frozen=True)
class Assessment:
    """The decision after considering recent history, not just the current frame."""

    level: AlertLevel
    reasons: tuple[AlertReason, ...]
    perclos: float | None
    eye_closure_s: float
    yawns_in_window: int
    calibrated: bool
    model_probability: float | None = None


@dataclass(frozen=True)
class FrameAnalysis:
    """Everything the pipeline produced for one frame."""

    metrics: FrameMetrics
    assessment: Assessment
    landmarks: Landmarks | None


def analysis_to_dict(analysis: FrameAnalysis) -> dict[str, Any]:
    """The JSON form of an analysis, as served by the status endpoint."""
    metrics, assessment = analysis.metrics, analysis.assessment
    pose = metrics.pose
    return {
        "level": assessment.level.value,
        "reasons": [reason.value for reason in assessment.reasons],
        "face_detected": metrics.face_detected,
        "calibrated": assessment.calibrated,
        "metrics": {
            "ear_left": metrics.ear_left,
            "ear_right": metrics.ear_right,
            "mar": metrics.mar,
            "head_pose": None
            if pose is None
            else {"pitch": pose.pitch, "yaw": pose.yaw, "roll": pose.roll},
        },
        "fatigue": {
            "perclos": assessment.perclos,
            "eye_closure_seconds": assessment.eye_closure_s,
            "yawns_in_window": assessment.yawns_in_window,
            "model_probability": assessment.model_probability,
        },
    }
