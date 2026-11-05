"""The monitoring loop, which owns the camera and fans results out to any number of readers."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, TypeVar

from dss.detection.pipeline import FatiguePipeline
from dss.domain import Frame, FrameAnalysis
from dss.observability import CAMERA_CONNECTED, FRAME_ERRORS, FRAME_SECONDS, FRAMES_PROCESSED
from dss.service.alerts import AlertLog, SoundAlarm
from dss.service.video import FrameSource

logger = logging.getLogger(__name__)

T = TypeVar("T")

_FPS_WINDOW_FRAMES = 30
_STALE_AFTER_S = 5.0


class LatestValue(Generic[T]):
    """Holds only the newest value and lets every reader see it.

    A queue would be wrong here: consumers would compete for items, so two video streams would
    each get half the frames, and a slow reader would watch the past. On a safety feed the only
    frame worth having is the latest one.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._value: T | None = None
        self._version = 0

    def publish(self, value: T) -> None:
        with self._condition:
            self._value = value
            self._version += 1
            self._condition.notify_all()

    def latest(self) -> T | None:
        with self._condition:
            return self._value

    def wait_newer(self, version: int, timeout: float) -> tuple[int, T] | None:
        """Block until a value newer than ``version`` exists. Returns None on timeout."""
        with self._condition:
            self._condition.wait_for(lambda: self._version > version, timeout=timeout)
            if self._version <= version or self._value is None:
                return None
            return self._version, self._value


@dataclass(frozen=True)
class ProcessedFrame:
    frame: Frame
    analysis: FrameAnalysis


class MonitorService:
    """Reads frames on a background thread, analyses them and publishes the results.

    The camera and pipeline are built on the worker thread, since neither OpenCV capture nor
    MediaPipe should be used across threads. A camera failure is retried rather than fatal, and an
    error on one frame costs that frame rather than the service.
    """

    def __init__(
        self,
        source_factory: Callable[[], FrameSource],
        pipeline_factory: Callable[[], FatiguePipeline],
        alert_log: AlertLog | None = None,
        alarm: SoundAlarm | None = None,
        retry_s: float = 2.0,
        stop_at_end: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._source_factory = source_factory
        self._pipeline_factory = pipeline_factory
        self.alerts = alert_log or AlertLog()
        self._alarm = alarm
        self._retry_s = retry_s
        self._stop_at_end = stop_at_end

        self.frames: LatestValue[ProcessedFrame] = LatestValue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._camera_connected = False
        self._last_frame_at: float | None = None
        self._frame_times: deque[float] = deque(maxlen=_FPS_WINDOW_FRAMES)
        self._fps = 0.0
        self._clock = clock

    @property
    def camera_connected(self) -> bool:
        return self._camera_connected

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def healthy(self) -> bool:
        """True while recent frames are flowing."""
        last = self._last_frame_at
        return (
            self.running
            and self._camera_connected
            and last is not None
            and self._clock() - last < _STALE_AFTER_S
        )

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="dss-monitor", daemon=True)
        self._thread.start()
        if self._alarm is not None:
            self._alarm.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._alarm is not None:
            self._alarm.stop()

    def _set_camera(self, connected: bool) -> None:
        self._camera_connected = connected
        CAMERA_CONNECTED.set(1 if connected else 0)

    def _run(self) -> None:
        try:
            pipeline = self._pipeline_factory()
        except Exception:
            logger.exception("Could not build the analysis pipeline")
            return

        try:
            while not self._stop.is_set():
                source = self._source_factory()
                if source.open():
                    logger.info("Video source connected")
                    self._set_camera(True)
                    self._read_until_failure(source, pipeline)
                    self._set_camera(False)
                source.close()
                if self._stop_at_end and not self._stop.is_set():
                    logger.info("Video source ended")
                    return
                if not self._stop.is_set():
                    logger.warning("Video source lost, retrying in %.1fs", self._retry_s)
                    self._stop.wait(self._retry_s)
        finally:
            pipeline.close()

    def _read_until_failure(self, source: FrameSource, pipeline: FatiguePipeline) -> None:
        while not self._stop.is_set():
            frame = source.read()
            if frame is None:
                return
            self._process(frame, pipeline)

    def _process(self, frame: Frame, pipeline: FatiguePipeline) -> None:
        started = self._clock()
        try:
            analysis = pipeline.process(frame, started)
        except Exception:
            FRAME_ERRORS.inc()
            logger.exception("Dropping a frame that failed analysis")
            return

        finished = self._clock()
        FRAMES_PROCESSED.inc()
        FRAME_SECONDS.observe(finished - started)
        self._last_frame_at = finished

        # Measured across a window, because two frames can land on the same clock tick.
        self._frame_times.append(finished)
        span = self._frame_times[-1] - self._frame_times[0]
        if span > 0:
            self._fps = (len(self._frame_times) - 1) / span

        self.alerts.observe(analysis.assessment)
        if self._alarm is not None:
            self._alarm.set_level(analysis.assessment.level)
        self.frames.publish(ProcessedFrame(frame, analysis))
