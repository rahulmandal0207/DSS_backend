"""Rules that look at measurements over time.

A single frame cannot show drowsiness: a blink looks exactly like a closed eye. Every rule here
therefore depends on how long something lasted. All windows are measured in seconds rather than
frames, because the frame rate varies with lighting and CPU load.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from statistics import median

from dss.config import DetectionThresholds
from dss.domain import (
    REASON_LEVELS,
    AlertLevel,
    AlertReason,
    Assessment,
    FrameMetrics,
    HeadPose,
)


class Hold:
    """Tracks how long a condition has been continuously true."""

    def __init__(self) -> None:
        self._since: float | None = None

    def update(self, active: bool, now: float) -> float:
        """Record the condition and return how many seconds it has been true."""
        if not active:
            self._since = None
            return 0.0
        if self._since is None:
            self._since = now
        return now - self._since


class Perclos:
    """Percentage of eye closure: the time-weighted share of a window with eyes closed."""

    def __init__(self, window_s: float, min_coverage: float) -> None:
        self._window_s = window_s
        self._min_coverage = min_coverage
        self._samples: deque[tuple[float, float, bool]] = deque()
        self._last: tuple[float, bool] | None = None

    def update(self, closed: bool | None, now: float) -> float | None:
        """Record a sample, or None when no eyes were visible and the gap should not count.

        Returns the closed fraction, or None until enough of the window has been observed.
        """
        if self._last is not None:
            started, was_closed = self._last
            self._samples.append((started, now - started, was_closed))
        self._last = None if closed is None else (now, closed)

        cutoff = now - self._window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

        observed = sum(duration for _, duration, _ in self._samples)
        if observed < self._window_s * self._min_coverage:
            return None
        return sum(duration for _, duration, shut in self._samples if shut) / observed


class EventCounter:
    """Counts completed events inside a rolling time window."""

    def __init__(self, window_s: float) -> None:
        self._window_s = window_s
        self._events: deque[float] = deque()

    def add(self, now: float) -> None:
        self._events.append(now)

    def count(self, now: float) -> int:
        while self._events and self._events[0] < now - self._window_s:
            self._events.popleft()
        return len(self._events)


class PoseCalibrator:
    """Learns the driver's neutral head position.

    Cameras are rarely mounted dead ahead, so a driver looking at the road might read 15 degrees
    off-axis. Angles are reported relative to the median over the first few seconds, so rules
    measure movement away from how this driver normally sits.
    """

    def __init__(self, duration_s: float) -> None:
        self._duration_s = duration_s
        self._started: float | None = None
        self._samples: list[HeadPose] = []
        self._neutral: HeadPose | None = None if duration_s > 0 else HeadPose(0.0, 0.0, 0.0)

    @property
    def calibrated(self) -> bool:
        return self._neutral is not None

    def adjust(self, pose: HeadPose, now: float) -> HeadPose | None:
        """Return the pose relative to neutral, or None while still calibrating."""
        if self._neutral is None:
            self._started = now if self._started is None else self._started
            self._samples.append(pose)
            if now - self._started < self._duration_s:
                return None
            self._neutral = HeadPose(
                pitch=median(sample.pitch for sample in self._samples),
                yaw=median(sample.yaw for sample in self._samples),
                roll=median(sample.roll for sample in self._samples),
            )
            self._samples.clear()
        return HeadPose(
            pitch=pose.pitch - self._neutral.pitch,
            yaw=pose.yaw - self._neutral.yaw,
            roll=pose.roll - self._neutral.roll,
        )


@dataclass
class _Holds:
    eyes_closed: Hold
    mouth_open: Hold
    head_down: Hold
    head_tilted: Hold
    looking_away: Hold
    face_lost: Hold


class TemporalAnalyzer:
    """Turns per-frame metrics into an alert level. Stateful: use one per monitored driver."""

    def __init__(self, thresholds: DetectionThresholds) -> None:
        self._t = thresholds
        self._perclos = Perclos(thresholds.perclos_window_s, thresholds.perclos_min_coverage)
        self._yawns = EventCounter(thresholds.yawn_window_s)
        self._calibrator = PoseCalibrator(thresholds.pose_calibration_s)
        self._holds = _Holds(Hold(), Hold(), Hold(), Hold(), Hold(), Hold())
        self._yawn_counted = False

    def update(self, metrics: FrameMetrics) -> Assessment:
        now = metrics.timestamp
        ear, mar = metrics.ear_mean, metrics.mar
        reasons: list[AlertReason] = []

        # Averaging both eyes means a wink, or one eye hidden by a turned head, is not a closure.
        closed = None if ear is None else ear < self._t.ear_closed
        closure_s = self._holds.eyes_closed.update(bool(closed), now)
        perclos = self._perclos.update(closed, now)
        if closure_s >= self._t.eye_closure_critical_s:
            reasons.append(AlertReason.PROLONGED_EYE_CLOSURE)
        if perclos is not None and perclos >= self._t.perclos_warning:
            reasons.append(AlertReason.HIGH_PERCLOS)

        self._update_yawns(mar, now)
        yawns = self._yawns.count(now)
        if yawns >= self._t.yawn_warning_count:
            reasons.append(AlertReason.FREQUENT_YAWNING)

        reasons.extend(self._assess_head(metrics.pose, now))

        lost_s = self._holds.face_lost.update(not metrics.face_detected, now)
        if lost_s >= self._t.face_lost_hold_s:
            reasons.append(AlertReason.DRIVER_NOT_VISIBLE)

        return Assessment(
            level=highest_level(reasons),
            reasons=tuple(reasons),
            perclos=perclos,
            eye_closure_s=closure_s,
            yawns_in_window=yawns,
            calibrated=self._calibrator.calibrated,
        )

    def _update_yawns(self, mar: float | None, now: float) -> None:
        wide_open = mar is not None and mar >= self._t.mar_yawn
        open_s = self._holds.mouth_open.update(wide_open, now)
        if not wide_open:
            self._yawn_counted = False
        elif open_s >= self._t.yawn_min_duration_s and not self._yawn_counted:
            self._yawn_counted = True
            self._yawns.add(now)

    def _assess_head(self, pose: HeadPose | None, now: float) -> list[AlertReason]:
        relative = None if pose is None else self._calibrator.adjust(pose, now)
        pitch = relative.pitch if relative else 0.0
        yaw = relative.yaw if relative else 0.0
        roll = relative.roll if relative else 0.0

        rules = (
            (
                self._holds.head_down,
                pitch >= self._t.head_pitch_limit_deg,
                AlertReason.HEAD_NODDING,
            ),
            (
                self._holds.head_tilted,
                abs(roll) >= self._t.head_roll_limit_deg,
                AlertReason.HEAD_TILTED,
            ),
            (
                self._holds.looking_away,
                abs(yaw) >= self._t.head_yaw_limit_deg,
                AlertReason.LOOKING_AWAY,
            ),
        )
        return [
            reason
            for hold, active, reason in rules
            if hold.update(active, now) >= self._t.head_pose_hold_s
        ]


def highest_level(reasons: list[AlertReason] | tuple[AlertReason, ...]) -> AlertLevel:
    return max(
        (REASON_LEVELS[r] for r in reasons), key=lambda lvl: lvl.rank, default=AlertLevel.NORMAL
    )
