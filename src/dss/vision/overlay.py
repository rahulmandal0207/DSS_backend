"""Drawing analysis results onto frames for the video streams."""

from __future__ import annotations

from enum import Enum

import cv2 as cv
import numpy as np

from dss.domain import AlertLevel, Frame, FrameAnalysis, Landmarks
from dss.vision.geometry import EYE_OUTER_CORNERS, INNER_LIPS, LEFT_EYE, MOUTH_CORNERS, RIGHT_EYE


class View(str, Enum):
    ORIGINAL = "original"
    ANNOTATED = "annotated"
    EYE_CLOSURE = "eye_closure"
    YAWN = "yawn"
    HEAD_TILT = "head_tilt"


_GREEN = (80, 200, 80)
_AMBER = (0, 170, 255)
_RED = (60, 60, 230)
_WHITE = (245, 245, 245)
_PANEL = (32, 28, 24)

_LEVEL_COLOURS = {AlertLevel.NORMAL: _GREEN, AlertLevel.WARNING: _AMBER, AlertLevel.CRITICAL: _RED}
_FONT = cv.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45
_LINE_HEIGHT = 18
_PADDING = 8


def _points(
    frame: Frame, landmarks: Landmarks, ids: tuple[int, ...], colour: tuple[int, ...]
) -> None:
    for index in ids:
        x, y = landmarks[index, :2]
        cv.circle(frame, (int(x), int(y)), 2, colour, -1, cv.LINE_AA)


def _draw_eyes(frame: Frame, landmarks: Landmarks, colour: tuple[int, ...]) -> None:
    _points(frame, landmarks, (*RIGHT_EYE, *LEFT_EYE), colour)


def _draw_mouth(frame: Frame, landmarks: Landmarks, colour: tuple[int, ...]) -> None:
    _points(frame, landmarks, (*MOUTH_CORNERS, *INNER_LIPS), colour)


def _draw_head(frame: Frame, landmarks: Landmarks, colour: tuple[int, ...]) -> None:
    left, right = (landmarks[index].astype(np.int32) for index in EYE_OUTER_CORNERS)
    cv.line(
        frame, (int(left[0]), int(left[1])), (int(right[0]), int(right[1])), colour, 2, cv.LINE_AA
    )


def _panel(frame: Frame, lines: list[str], colour: tuple[int, ...]) -> None:
    """Draw text on a translucent panel, bottom left, where it does not cover the driver's eyes."""
    text_width = max(cv.getTextSize(line, _FONT, _FONT_SCALE, 1)[0][0] for line in lines)
    panel_height = _PADDING + _LINE_HEIGHT * len(lines)
    top = max(frame.shape[0] - panel_height, 0)

    region = frame[top:, : text_width + 2 * _PADDING]
    region[:] = (region * 0.35 + np.array(_PANEL) * 0.65).astype(np.uint8)
    for row, text in enumerate(lines):
        cv.putText(
            frame,
            text,
            (_PADDING, top + _LINE_HEIGHT * (row + 1)),
            _FONT,
            _FONT_SCALE,
            colour if row == 0 else _WHITE,
            1,
            cv.LINE_AA,
        )


def _status_lines(view: View, analysis: FrameAnalysis) -> list[str]:
    metrics, assessment = analysis.metrics, analysis.assessment
    if not metrics.face_detected:
        return [assessment.level.value.upper(), "No driver detected"]

    lines = [assessment.level.value.upper()]
    ear, mar, pose = metrics.ear_mean, metrics.mar, metrics.pose
    if view in (View.ANNOTATED, View.EYE_CLOSURE) and ear is not None:
        lines.append(f"EAR {ear:.2f}  closed {assessment.eye_closure_s:.1f}s")
        if assessment.perclos is not None:
            lines.append(f"PERCLOS {assessment.perclos:.0%}")
    if view in (View.ANNOTATED, View.YAWN) and mar is not None:
        lines.append(f"MAR {mar:.2f}  yawns {assessment.yawns_in_window}")
    if view in (View.ANNOTATED, View.HEAD_TILT) and pose is not None:
        lines.append(f"Pitch {pose.pitch:+.0f}  Yaw {pose.yaw:+.0f}  Roll {pose.roll:+.0f}")
    return lines


def render(view: View, frame: Frame, analysis: FrameAnalysis) -> Frame:
    """Return a copy of ``frame`` annotated for ``view``. The input frame is left untouched."""
    if view is View.ORIGINAL:
        return frame

    canvas = frame.copy()
    colour = _LEVEL_COLOURS[analysis.assessment.level]
    landmarks = analysis.landmarks
    if landmarks is not None:
        if view in (View.ANNOTATED, View.EYE_CLOSURE):
            _draw_eyes(canvas, landmarks, colour)
        if view in (View.ANNOTATED, View.YAWN):
            _draw_mouth(canvas, landmarks, colour)
        if view in (View.ANNOTATED, View.HEAD_TILT):
            _draw_head(canvas, landmarks, colour)
    _panel(canvas, _status_lines(view, analysis), colour)
    return canvas
