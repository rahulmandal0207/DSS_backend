"""Reading frames from a webcam, video file or stream URL."""

from __future__ import annotations

import logging
import time
from typing import Protocol, cast

import cv2 as cv

from dss.domain import Frame

logger = logging.getLogger(__name__)


class FrameSource(Protocol):
    def open(self) -> bool:
        """Connect to the source. Returns False if it is unavailable."""
        ...

    def read(self) -> Frame | None:
        """Return the next frame, or None if the source failed or ended."""
        ...

    def close(self) -> None: ...


class VideoSource:
    """A ``cv2.VideoCapture`` source.

    A file plays in real time rather than as fast as it can be decoded, because the fatigue rules
    measure wall-clock durations.
    """

    def __init__(self, source: str, loop: bool = False) -> None:
        self._source = source
        self._loop = loop
        self._is_camera = source.isdigit()
        self._capture: cv.VideoCapture | None = None
        self._frame_interval = 0.0
        self._next_frame_at = 0.0

    def open(self) -> bool:
        target = int(self._source) if self._is_camera else self._source
        self._capture = cv.VideoCapture(target)
        if not self._capture.isOpened():
            logger.warning("Could not open video source %r", self._source)
            self.close()
            return False

        if not self._is_camera:
            fps = self._capture.get(cv.CAP_PROP_FPS)
            self._frame_interval = 1.0 / fps if fps > 0 else 0.0
            self._next_frame_at = time.monotonic()
        return True

    def read(self) -> Frame | None:
        if self._capture is None:
            return None

        success, frame = self._capture.read()
        if not success and self._loop and not self._is_camera:
            self._capture.set(cv.CAP_PROP_POS_FRAMES, 0)
            success, frame = self._capture.read()
        if not success:
            return None

        self._pace()
        return cast(Frame, frame)

    def _pace(self) -> None:
        if self._frame_interval:
            self._next_frame_at += self._frame_interval
            delay = self._next_frame_at - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                # Running behind: do not try to catch up with a burst of frames.
                self._next_frame_at = time.monotonic()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
