"""Facial ratios computed from MediaPipe Face Mesh landmarks.

Every function takes landmarks in **pixels**. MediaPipe normalises x by width and y by height,
so a ratio taken from normalised coordinates changes with the camera's aspect ratio. Pixels keep
the measurements comparable across cameras, and between training data and live video.

The ratios use only x and y: they describe what the camera sees, as in the original EAR paper.
"""

from __future__ import annotations

import numpy as np

from dss.domain import Landmarks

# Ordered p1..p6 as in Soukupova & Cech (2016): corner, upper, upper, corner, lower, lower.
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
LEFT_EYE = (362, 385, 387, 263, 373, 380)

MOUTH_CORNERS = (61, 291)
INNER_LIPS = (13, 14)

FOREHEAD = 10
CHIN = 152
EYE_OUTER_CORNERS = (33, 263)

_EPSILON = 1e-6


def _distance(points: Landmarks, a: int, b: int) -> float:
    return float(np.linalg.norm(points[a, :2] - points[b, :2]))


def eye_aspect_ratio(points: Landmarks, eye: tuple[int, ...]) -> float:
    """Eye height over width. Roughly 0.3 when open, falling towards 0 as the eye closes."""
    p1, p2, p3, p4, p5, p6 = eye
    height = _distance(points, p2, p6) + _distance(points, p3, p5)
    return height / max(2.0 * _distance(points, p1, p4), _EPSILON)


def mouth_aspect_ratio(points: Landmarks) -> float:
    """Inner-lip gap over mouth width. Near 0 when closed, and rises as the mouth opens."""
    return _distance(points, *INNER_LIPS) / max(_distance(points, *MOUTH_CORNERS), _EPSILON)
