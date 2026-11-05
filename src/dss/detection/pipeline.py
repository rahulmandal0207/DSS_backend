"""The per-frame pipeline: landmarks, then measurements, then a decision."""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from dss.config import DetectionThresholds
from dss.detection.temporal import TemporalAnalyzer, highest_level
from dss.domain import AlertReason, Assessment, Frame, FrameAnalysis, FrameMetrics, Landmarks
from dss.ml.model import DrowsinessModel
from dss.vision.geometry import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio, mouth_aspect_ratio
from dss.vision.head_pose import estimate_head_pose
from dss.vision.landmarks import LandmarkDetector


def measure(
    detector: LandmarkDetector, frame: Frame, timestamp: float
) -> tuple[FrameMetrics, Landmarks | None]:
    """Measure one frame on its own, with no knowledge of earlier frames."""
    landmarks = detector.detect(frame)
    if landmarks is None:
        return FrameMetrics(timestamp=timestamp, face_detected=False), None

    metrics = FrameMetrics(
        timestamp=timestamp,
        face_detected=True,
        ear_left=eye_aspect_ratio(landmarks, LEFT_EYE),
        ear_right=eye_aspect_ratio(landmarks, RIGHT_EYE),
        mar=mouth_aspect_ratio(landmarks),
        pose=estimate_head_pose(landmarks),
    )
    return metrics, landmarks


class FatiguePipeline:
    """Analyses frames in order. Stateful: use one instance per monitored driver."""

    def __init__(
        self,
        detector: LandmarkDetector,
        thresholds: DetectionThresholds | None = None,
        model: DrowsinessModel | None = None,
        model_threshold: float = 0.7,
        model_smoothing_s: float = 3.0,
    ) -> None:
        self._detector = detector
        self._temporal = TemporalAnalyzer(thresholds or DetectionThresholds())
        self._model = model
        self._model_threshold = model_threshold
        self._model_smoothing_s = model_smoothing_s
        self._probabilities: deque[tuple[float, float]] = deque()

    def process(self, frame: Frame, timestamp: float) -> FrameAnalysis:
        metrics, landmarks = measure(self._detector, frame, timestamp)
        assessment = self._temporal.update(metrics)
        if self._model is not None:
            assessment = self._apply_model(self._model, metrics, assessment)
        return FrameAnalysis(metrics, assessment, landmarks)

    def _apply_model(
        self, model: DrowsinessModel, metrics: FrameMetrics, assessment: Assessment
    ) -> Assessment:
        """Blend in the classifier. It can raise a warning, but never clears a rule's alert."""
        now = metrics.timestamp
        probability = model.predict_probability(metrics)
        if probability is not None:
            self._probabilities.append((now, probability))
        while self._probabilities and self._probabilities[0][0] < now - self._model_smoothing_s:
            self._probabilities.popleft()
        if not self._probabilities:
            return assessment

        smoothed = sum(p for _, p in self._probabilities) / len(self._probabilities)
        reasons = assessment.reasons
        if smoothed >= self._model_threshold:
            reasons = (*reasons, AlertReason.MODEL_DROWSY)
        return replace(
            assessment,
            reasons=reasons,
            level=highest_level(reasons),
            model_probability=smoothed,
        )

    def close(self) -> None:
        self._detector.close()
