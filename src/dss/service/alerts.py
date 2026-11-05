"""Recording alert changes and sounding the in-cab alarm."""

from __future__ import annotations

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np

from dss.domain import AlertLevel, AlertReason, Assessment
from dss.observability import ALERT_RANK, ALERTS_RAISED

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 22_050
# Frequency in Hz, then beep length and the gap before the next beep in seconds.
_TONES = {AlertLevel.WARNING: (660.0, 0.25, 2.0), AlertLevel.CRITICAL: (880.0, 0.40, 0.6)}


@dataclass(frozen=True)
class AlertEvent:
    time: datetime
    level: AlertLevel
    reasons: tuple[AlertReason, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": self.time.isoformat(),
            "level": self.level.value,
            "reasons": [reason.value for reason in self.reasons],
        }


class AlertLog:
    """Keeps a bounded history of alert changes, so a long alert is one entry, not thousands."""

    def __init__(self, size: int = 200) -> None:
        self._events: deque[AlertEvent] = deque(maxlen=size)
        self._level = AlertLevel.NORMAL
        self._reasons: tuple[AlertReason, ...] = ()
        self._lock = threading.Lock()

    def observe(self, assessment: Assessment) -> AlertEvent | None:
        """Record the assessment if it differs from the last one. Returns any new event."""
        with self._lock:
            if (assessment.level, set(assessment.reasons)) == (self._level, set(self._reasons)):
                return None
            self._level, self._reasons = assessment.level, assessment.reasons
            event = AlertEvent(datetime.now(timezone.utc), assessment.level, assessment.reasons)
            self._events.append(event)

        ALERT_RANK.set(assessment.level.rank)
        if assessment.level is not AlertLevel.NORMAL:
            ALERTS_RAISED.labels(level=assessment.level.value).inc()
            logger.warning(
                "Driver %s: %s",
                assessment.level.value,
                ", ".join(reason.value for reason in assessment.reasons),
            )
        return event

    def recent(self, limit: int) -> list[AlertEvent]:
        """The newest events first."""
        with self._lock:
            return list(self._events)[-limit:][::-1]


class SoundAlarm:
    """Beeps in the background while the driver is in an alert state.

    Playback has its own thread so the audio never stalls the frame loop. A machine with no audio
    device logs one warning and carries on: a missing speaker must not stop the monitoring.
    """

    def __init__(self) -> None:
        self._level = AlertLevel.NORMAL
        self._changed = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="dss-alarm", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def set_level(self, level: AlertLevel) -> None:
        if level is not self._level:
            self._level = level
            self._changed.set()

    def stop(self) -> None:
        self._stop.set()
        self._changed.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            tone = _TONES.get(self._level)
            if tone is None:
                self._changed.wait()
                self._changed.clear()
                continue

            frequency, duration, pause = tone
            if not self._beep(frequency, duration):
                return
            self._changed.clear()
            self._changed.wait(timeout=pause)
            self._changed.clear()

    def _beep(self, frequency: float, duration: float) -> bool:
        try:
            import sounddevice

            samples = np.arange(int(_SAMPLE_RATE * duration)) / _SAMPLE_RATE
            sounddevice.play(0.4 * np.sin(2 * np.pi * frequency * samples), _SAMPLE_RATE)
            sounddevice.wait()
        except Exception:  # sounddevice raises OSError or its own PortAudioError
            logger.warning("Audio alarm disabled: no usable audio device", exc_info=True)
            return False
        return True
