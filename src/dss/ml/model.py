"""Loading a trained classifier and scoring live frames with it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from dss.domain import FrameMetrics

FEATURE_COLUMNS = ("ear_left", "ear_right", "mar", "pitch", "yaw", "roll")
ARTIFACT_VERSION = 1


def feature_vector(metrics: FrameMetrics) -> tuple[float, ...] | None:
    """The classifier's input for a frame, or None if any measurement is unavailable."""
    pose = metrics.pose
    if metrics.ear_left is None or metrics.ear_right is None or metrics.mar is None or pose is None:
        return None
    return (metrics.ear_left, metrics.ear_right, metrics.mar, pose.pitch, pose.yaw, pose.roll)


class DrowsinessModel:
    """A trained estimator bundled with the feature order it was trained on."""

    def __init__(self, estimator: Any) -> None:
        self._estimator = estimator

    @classmethod
    def load(cls, path: Path) -> DrowsinessModel:
        """Load an artifact written by ``dss train``.

        Artifacts are pickles, so loading one runs code from it. Only load models you trained.
        """
        artifact = joblib.load(path)
        if not isinstance(artifact, dict) or artifact.get("version") != ARTIFACT_VERSION:
            raise ValueError(f"{path} is not a version {ARTIFACT_VERSION} DSS model artifact")
        if tuple(artifact["features"]) != FEATURE_COLUMNS:
            raise ValueError(f"{path} was trained on different features: {artifact['features']}")
        return cls(artifact["estimator"])

    def predict_probability(self, metrics: FrameMetrics) -> float | None:
        """The probability that this frame shows a drowsy driver, or None without a full face."""
        features = feature_vector(metrics)
        if features is None:
            return None
        row = np.array([features], dtype=np.float64)
        return float(self._estimator.predict_proba(row)[0, 1])


def save_artifact(estimator: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"version": ARTIFACT_VERSION, "features": list(FEATURE_COLUMNS), "estimator": estimator},
        path,
    )
