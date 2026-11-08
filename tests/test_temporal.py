"""Fatigue is a pattern over time, so these tests play out scenarios rather than single frames."""

import pytest

from conftest import metrics_at, seconds
from dss.config import DetectionThresholds
from dss.detection.temporal import EventCounter, Hold, Perclos, PoseCalibrator, TemporalAnalyzer
from dss.domain import AlertLevel, AlertReason, Assessment, HeadPose

OPEN = 0.30
SHUT = 0.05


def play(analyzer: TemporalAnalyzer, times: list[float], **kwargs: object) -> Assessment:
    result = None
    for moment in times:
        result = analyzer.update(metrics_at(moment, **kwargs))  # type: ignore[arg-type]
    assert result is not None
    return result


def test_alert_driver_stays_normal(thresholds: DetectionThresholds) -> None:
    result = play(TemporalAnalyzer(thresholds), seconds(0, 10))
    assert result.level is AlertLevel.NORMAL
    assert result.reasons == ()


def test_a_blink_is_not_an_alarm(thresholds: DetectionThresholds) -> None:
    analyzer = TemporalAnalyzer(thresholds)
    play(analyzer, seconds(0, 2), ear=OPEN)
    during = play(analyzer, seconds(2, 2.2), ear=SHUT)  # a normal 200 ms blink
    after = play(analyzer, seconds(2.2, 3), ear=OPEN)
    assert during.level is AlertLevel.NORMAL
    assert after.eye_closure_s == 0


def test_prolonged_closure_is_critical(thresholds: DetectionThresholds) -> None:
    result = play(TemporalAnalyzer(thresholds), seconds(0, 2), ear=SHUT)
    assert result.level is AlertLevel.CRITICAL
    assert AlertReason.PROLONGED_EYE_CLOSURE in result.reasons


def test_one_closed_eye_is_not_closure(thresholds: DetectionThresholds) -> None:
    analyzer = TemporalAnalyzer(thresholds)
    result = None
    for moment in seconds(0, 3):
        metrics = metrics_at(moment)
        result = analyzer.update(
            type(metrics)(
                timestamp=moment,
                face_detected=True,
                ear_left=0.32,
                ear_right=0.12,  # a wink: the mean stays above the threshold
                mar=0.05,
                pose=metrics.pose,
            )
        )
    assert result is not None
    assert result.eye_closure_s == 0


def test_repeated_long_blinks_raise_perclos_without_critical(
    thresholds: DetectionThresholds,
) -> None:
    analyzer = TemporalAnalyzer(thresholds)
    result = None
    for cycle in range(40):
        start = cycle * 2.0
        play(analyzer, seconds(start, start + 1.4), ear=OPEN)
        result = play(analyzer, seconds(start + 1.4, start + 2.0), ear=SHUT)  # 0.6 s each
    assert result is not None
    assert result.perclos == pytest.approx(0.3, abs=0.03)
    assert AlertReason.HIGH_PERCLOS in result.reasons
    assert result.level is AlertLevel.WARNING


def test_a_single_yawn_is_counted_once_and_is_not_an_alert(
    thresholds: DetectionThresholds,
) -> None:
    analyzer = TemporalAnalyzer(thresholds)
    result = play(analyzer, seconds(0, 4), mar=0.8)
    assert result.yawns_in_window == 1
    assert result.level is AlertLevel.NORMAL


def test_talking_is_not_a_yawn(thresholds: DetectionThresholds) -> None:
    analyzer = TemporalAnalyzer(thresholds)
    result = None
    for index, moment in enumerate(seconds(0, 10)):
        mar = 0.8 if index % 15 < 5 else 0.1  # brief peaks, far shorter than a yawn
        result = analyzer.update(metrics_at(moment, mar=mar))
    assert result is not None
    assert result.yawns_in_window == 0


def test_frequent_yawning_warns(thresholds: DetectionThresholds) -> None:
    analyzer = TemporalAnalyzer(thresholds)
    result = None
    for start in (0, 30, 60):
        play(analyzer, seconds(start, start + 3), mar=0.8)
        result = play(analyzer, seconds(start + 3, start + 4), mar=0.05)
    assert result is not None
    assert result.yawns_in_window == 3
    assert AlertReason.FREQUENT_YAWNING in result.reasons


def test_old_yawns_age_out(thresholds: DetectionThresholds) -> None:
    analyzer = TemporalAnalyzer(thresholds)
    play(analyzer, seconds(0, 3), mar=0.8)
    result = play(analyzer, seconds(thresholds.yawn_window_s + 10, thresholds.yawn_window_s + 11))
    assert result.yawns_in_window == 0


def test_nodding_off_is_critical(thresholds: DetectionThresholds) -> None:
    pose = HeadPose(pitch=30, yaw=0, roll=0)
    result = play(TemporalAnalyzer(thresholds), seconds(0, 3), pose=pose)
    assert result.level is AlertLevel.CRITICAL
    assert AlertReason.HEAD_NODDING in result.reasons


def test_checking_a_mirror_is_not_an_alert(thresholds: DetectionThresholds) -> None:
    pose = HeadPose(pitch=0, yaw=45, roll=0)
    result = play(TemporalAnalyzer(thresholds), seconds(0, 1), pose=pose)
    assert result.level is AlertLevel.NORMAL


def test_looking_away_warns(thresholds: DetectionThresholds) -> None:
    pose = HeadPose(pitch=0, yaw=45, roll=0)
    result = play(TemporalAnalyzer(thresholds), seconds(0, 3), pose=pose)
    assert result.reasons == (AlertReason.LOOKING_AWAY,)
    assert result.level is AlertLevel.WARNING


def test_sustained_head_tilt_warns(thresholds: DetectionThresholds) -> None:
    pose = HeadPose(pitch=0, yaw=0, roll=-35)
    result = play(TemporalAnalyzer(thresholds), seconds(0, 3), pose=pose)
    assert AlertReason.HEAD_TILTED in result.reasons


def test_missing_driver_is_a_warning_not_silence(thresholds: DetectionThresholds) -> None:
    analyzer = TemporalAnalyzer(thresholds)
    play(analyzer, seconds(0, 1))
    result = play(analyzer, seconds(1, 4), face=False)
    assert result.level is AlertLevel.WARNING
    assert AlertReason.DRIVER_NOT_VISIBLE in result.reasons


def test_critical_outranks_warning(thresholds: DetectionThresholds) -> None:
    pose = HeadPose(pitch=0, yaw=45, roll=0)
    result = play(TemporalAnalyzer(thresholds), seconds(0, 3), ear=SHUT, pose=pose)
    assert result.level is AlertLevel.CRITICAL
    assert {AlertReason.LOOKING_AWAY, AlertReason.PROLONGED_EYE_CLOSURE} <= set(result.reasons)


def test_alerts_clear_when_the_driver_recovers(thresholds: DetectionThresholds) -> None:
    analyzer = TemporalAnalyzer(thresholds)
    assert play(analyzer, seconds(0, 2), ear=SHUT).level is AlertLevel.CRITICAL
    assert play(analyzer, seconds(2, 2.1), ear=OPEN).level is AlertLevel.NORMAL


def test_camera_mounted_off_axis_is_calibrated_away() -> None:
    thresholds = DetectionThresholds(pose_calibration_s=2.0)
    analyzer = TemporalAnalyzer(thresholds)
    mounted = HeadPose(pitch=25, yaw=35, roll=0)
    calibrating = play(analyzer, seconds(0, 1.9), pose=mounted)
    settled = play(analyzer, seconds(1.9, 12), pose=mounted)
    assert not calibrating.calibrated
    assert settled.calibrated
    assert settled.level is AlertLevel.NORMAL

    dropped = HeadPose(pitch=55, yaw=35, roll=0)
    assert play(analyzer, seconds(12, 15), pose=dropped).level is AlertLevel.CRITICAL


def test_a_brief_repeated_frame_is_not_a_freeze(thresholds: DetectionThresholds) -> None:
    result = play(TemporalAnalyzer(thresholds), seconds(0, 2), unchanged=True)
    assert result.level is AlertLevel.NORMAL


def test_a_frozen_picture_warns(thresholds: DetectionThresholds) -> None:
    result = play(TemporalAnalyzer(thresholds), seconds(0, 4), unchanged=True)
    assert result.level is AlertLevel.WARNING
    assert result.reasons == (AlertReason.CAMERA_FROZEN,)


def test_a_freeze_mid_blink_is_not_prolonged_closure(thresholds: DetectionThresholds) -> None:
    """Closed eyes in a stuck picture are not closed eyes on the road."""
    analyzer = TemporalAnalyzer(thresholds)
    play(analyzer, seconds(0, 5))
    result = play(analyzer, seconds(5, 10), ear=SHUT, unchanged=True)
    assert result.reasons == (AlertReason.CAMERA_FROZEN,)
    assert result.eye_closure_s == 0


def test_a_frozen_empty_picture_blames_the_camera(thresholds: DetectionThresholds) -> None:
    result = play(TemporalAnalyzer(thresholds), seconds(0, 6), face=False, unchanged=True)
    assert result.reasons == (AlertReason.CAMERA_FROZEN,)


def test_frozen_time_does_not_count_towards_perclos(thresholds: DetectionThresholds) -> None:
    analyzer = TemporalAnalyzer(thresholds)
    play(analyzer, seconds(0, 40), ear=OPEN)
    result = play(analyzer, seconds(40, 80), ear=SHUT, unchanged=True)
    assert AlertReason.HIGH_PERCLOS not in result.reasons


def test_rules_resume_when_the_picture_returns(thresholds: DetectionThresholds) -> None:
    analyzer = TemporalAnalyzer(thresholds)
    play(analyzer, seconds(0, 5), unchanged=True)
    result = play(analyzer, seconds(5, 7), ear=SHUT)
    assert AlertReason.CAMERA_FROZEN not in result.reasons
    assert AlertReason.PROLONGED_EYE_CLOSURE in result.reasons


def test_hold_measures_continuous_time_only() -> None:
    hold = Hold()
    assert hold.update(True, 10.0) == 0
    assert hold.update(True, 12.5) == 2.5
    assert hold.update(False, 13.0) == 0
    assert hold.update(True, 14.0) == 0


def test_perclos_waits_for_enough_data() -> None:
    perclos = Perclos(window_s=60, min_coverage=0.5)
    assert perclos.update(False, 0.0) is None
    assert perclos.update(True, 10.0) is None
    assert perclos.update(False, 40.0) == pytest.approx(0.75)


def test_perclos_is_time_weighted_not_frame_weighted() -> None:
    perclos = Perclos(window_s=10, min_coverage=0.1)
    perclos.update(True, 0.0)
    # Few frames during the slow 4 s closed segment, many during the open second.
    perclos.update(False, 4.0)
    result = None
    for moment in seconds(4.1, 5, fps=100):
        result = perclos.update(False, moment)
    assert result == pytest.approx(0.8, abs=0.03)


def test_perclos_ignores_time_with_no_face() -> None:
    perclos = Perclos(window_s=100, min_coverage=0.1)
    perclos.update(False, 0.0)
    perclos.update(None, 20.0)
    perclos.update(None, 60.0)
    perclos.update(True, 60.0)
    assert perclos.update(False, 80.0) == pytest.approx(0.5)


def test_event_counter_window() -> None:
    counter = EventCounter(window_s=10)
    counter.add(0.0)
    counter.add(5.0)
    assert counter.count(9.0) == 2
    assert counter.count(12.0) == 1


def test_zero_calibration_is_passthrough() -> None:
    pose = HeadPose(5, 6, 7)
    assert PoseCalibrator(0).adjust(pose, 0.0) == pose
