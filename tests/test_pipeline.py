"""The pipeline must give a complete result for every frame, whatever is in it."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import ScriptedDetector, live_frame, seconds, synthetic_face
from dss.config import DetectionThresholds
from dss.detection.pipeline import FatiguePipeline
from dss.domain import AlertLevel, AlertReason, Frame, FrameMetrics, HeadPose
from dss.ml.model import ARTIFACT_VERSION, FEATURE_COLUMNS, DrowsinessModel, save_artifact
from dss.ml.train import train_model

FIXED_MODEL_COLUMNS = ["subject", "drowsy", *FEATURE_COLUMNS]


def test_no_face_returns_a_complete_result(frame: Frame, thresholds: DetectionThresholds) -> None:
    pipeline = FatiguePipeline(ScriptedDetector(None), thresholds)
    analysis = pipeline.process(frame, 0.0)
    assert not analysis.metrics.face_detected
    assert analysis.landmarks is None
    assert analysis.assessment.level is AlertLevel.NORMAL


def test_alert_driver_has_every_measurement(frame: Frame, thresholds: DetectionThresholds) -> None:
    """Regression: EAR and MAR used to be None unless already over threshold."""
    pipeline = FatiguePipeline(ScriptedDetector(synthetic_face(ear=0.3, mar=0.05)), thresholds)
    metrics = pipeline.process(frame, 0.0).metrics
    assert metrics.ear_left == pytest.approx(0.3)
    assert metrics.ear_right == pytest.approx(0.3)
    assert metrics.mar == pytest.approx(0.05)


def test_scenario_from_landmarks_to_alert(thresholds: DetectionThresholds) -> None:
    detector = ScriptedDetector(synthetic_face(ear=0.3))
    pipeline = FatiguePipeline(detector, thresholds)
    for index, moment in enumerate(seconds(0, 2)):
        analysis = pipeline.process(live_frame(index), moment)
        assert analysis.assessment.level is AlertLevel.NORMAL

    detector.default = synthetic_face(ear=0.05)
    last = None
    for index, moment in enumerate(seconds(2, 5)):
        last = pipeline.process(live_frame(index + 1000), moment)
    assert last is not None
    assert last.assessment.level is AlertLevel.CRITICAL


def test_frozen_picture_of_an_alert_driver_is_not_reported_as_normal(
    frame: Frame, thresholds: DetectionThresholds
) -> None:
    """Regression: a stalled camera showing a healthy face used to read as normal forever."""
    pipeline = FatiguePipeline(ScriptedDetector(synthetic_face(ear=0.3)), thresholds)
    last = None
    for moment in seconds(0, 5):
        last = pipeline.process(frame, moment)
    assert last is not None
    assert last.assessment.level is AlertLevel.WARNING
    assert last.assessment.reasons == (AlertReason.CAMERA_FROZEN,)


def test_a_motionless_driver_on_a_live_camera_is_not_frozen(
    thresholds: DetectionThresholds,
) -> None:
    pipeline = FatiguePipeline(ScriptedDetector(synthetic_face(ear=0.3)), thresholds)
    last = None
    for index, moment in enumerate(seconds(0, 10)):
        last = pipeline.process(live_frame(index), moment)
    assert last is not None
    assert last.assessment.level is AlertLevel.NORMAL


def test_alert_clears_when_the_picture_moves_again(
    frame: Frame, thresholds: DetectionThresholds
) -> None:
    pipeline = FatiguePipeline(ScriptedDetector(synthetic_face(ear=0.3)), thresholds)
    for moment in seconds(0, 5):
        pipeline.process(frame, moment)
    analysis = pipeline.process(live_frame(1), 5.1)
    assert analysis.assessment.reasons == ()


def test_close_releases_the_detector(thresholds: DetectionThresholds) -> None:
    detector = ScriptedDetector()
    FatiguePipeline(detector, thresholds).close()
    assert detector.closed


class FixedModel(DrowsinessModel):
    def __init__(self, probability: float | None) -> None:
        self.probability = probability

    def predict_probability(self, metrics: FrameMetrics) -> float | None:
        return self.probability


def test_model_can_raise_a_warning(frame: Frame, thresholds: DetectionThresholds) -> None:
    pipeline = FatiguePipeline(
        ScriptedDetector(synthetic_face()), thresholds, model=FixedModel(0.9)
    )
    assessment = pipeline.process(frame, 0.0).assessment
    assert assessment.level is AlertLevel.WARNING
    assert assessment.reasons == (AlertReason.MODEL_DROWSY,)
    assert assessment.model_probability == pytest.approx(0.9)


def test_model_cannot_clear_a_rule_alert(frame: Frame, thresholds: DetectionThresholds) -> None:
    pipeline = FatiguePipeline(
        ScriptedDetector(synthetic_face(ear=0.05)), thresholds, model=FixedModel(0.01)
    )
    last = None
    for moment in seconds(0, 2):
        last = pipeline.process(frame, moment)
    assert last is not None
    assert last.assessment.level is AlertLevel.CRITICAL


def test_model_score_is_smoothed_over_time(frame: Frame, thresholds: DetectionThresholds) -> None:
    model = FixedModel(0.0)
    pipeline = FatiguePipeline(
        ScriptedDetector(synthetic_face()), thresholds, model=model, model_smoothing_s=3.0
    )
    for moment in seconds(0, 2):
        pipeline.process(frame, moment)
    model.probability = 1.0
    assert pipeline.process(frame, 2.0).assessment.reasons == ()


def test_model_is_ignored_while_the_picture_is_frozen(
    frame: Frame, thresholds: DetectionThresholds
) -> None:
    pipeline = FatiguePipeline(
        ScriptedDetector(synthetic_face()), thresholds, model=FixedModel(0.9)
    )
    last = None
    for moment in seconds(0, 5):
        last = pipeline.process(frame, moment)
    assert last is not None
    assert AlertReason.MODEL_DROWSY not in last.assessment.reasons


def test_model_gets_no_score_without_a_face(frame: Frame, thresholds: DetectionThresholds) -> None:
    model = DrowsinessModel(estimator=None)
    pipeline = FatiguePipeline(ScriptedDetector(None), thresholds, model=model)
    assert pipeline.process(frame, 0.0).assessment.model_probability is None


def feature_table(subjects: int = 6, rows_each: int = 30) -> pd.DataFrame:
    """Separable synthetic data: drowsy rows have low EAR."""
    rng = np.random.default_rng(1)
    rows = []
    for subject in range(subjects):
        for drowsy in (0, 1):
            for _ in range(rows_each):
                ear = rng.normal(0.14 if drowsy else 0.32, 0.02)
                rows.append(
                    {
                        "subject": f"s{subject}",
                        "drowsy": drowsy,
                        "ear_left": ear,
                        "ear_right": ear,
                        "mar": rng.normal(0.1, 0.03),
                        "pitch": rng.normal(0, 3),
                        "yaw": rng.normal(0, 3),
                        "roll": rng.normal(0, 3),
                    }
                )
    return pd.DataFrame(rows, columns=FIXED_MODEL_COLUMNS)


def test_training_holds_out_whole_subjects() -> None:
    _, report = train_model(feature_table())
    assert report.split == "by_subject"
    assert report.accuracy > 0.95
    assert report.subjects == 6


def test_training_without_subjects_flags_a_random_split() -> None:
    table = feature_table().assign(subject="unknown")
    _, report = train_model(table)
    assert report.split == "random"


def test_training_rejects_a_single_class() -> None:
    with pytest.raises(ValueError, match="both drowsy and alert"):
        train_model(feature_table().assign(drowsy=1))


def test_model_round_trip_scores_a_live_frame(tmp_path: Path) -> None:
    estimator, _ = train_model(feature_table())
    path = tmp_path / "model.joblib"
    save_artifact(estimator, path)
    model = DrowsinessModel.load(path)

    pose = HeadPose(0, 0, 0)
    drowsy = FrameMetrics(0.0, True, 0.12, 0.12, 0.1, pose)
    alert = FrameMetrics(0.0, True, 0.33, 0.33, 0.1, pose)
    drowsy_score = model.predict_probability(drowsy)
    alert_score = model.predict_probability(alert)
    assert drowsy_score is not None
    assert alert_score is not None
    assert drowsy_score > alert_score


def test_loading_rejects_a_foreign_artifact(tmp_path: Path) -> None:
    import joblib

    path = tmp_path / "other.joblib"
    joblib.dump({"version": ARTIFACT_VERSION + 1}, path)
    with pytest.raises(ValueError, match="not a version"):
        DrowsinessModel.load(path)
