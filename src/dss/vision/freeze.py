"""Noticing when a camera stops delivering new pictures."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from dss.domain import Frame

# A sparse grid of one colour channel. Pixels are sampled rather than averaged, because averaging
# would smooth away the sensor noise that this check relies on.
_STRIDE = 8
_CHANNEL = 1


class FreezeDetector:
    """Reports whether a frame shows the same picture as before.

    A frozen picture is the most dangerous camera failure, because everything downstream keeps
    working: a stalled feed showing an alert driver would read as "normal" forever. Real sensors
    always produce noise, so even a motionless scene changes between frames, while a stalled feed
    repeats them exactly.

    Each frame is compared with the last one that changed rather than simply the previous one, so
    a slow drift still counts as movement.
    """

    def __init__(self, min_change: float) -> None:
        self._min_change = min_change
        self._reference: NDArray[np.int16] | None = None

    def is_unchanged(self, frame: Frame) -> bool:
        sample = frame[::_STRIDE, ::_STRIDE, _CHANNEL].astype(np.int16)
        reference = self._reference
        if reference is None or reference.shape != sample.shape:
            self._reference = sample
            return False

        if float(np.abs(sample - reference).mean()) < self._min_change:
            return True
        self._reference = sample
        return False
