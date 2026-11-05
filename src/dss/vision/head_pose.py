"""3D head orientation, read from the axes of the face itself.

MediaPipe estimates depth for every landmark, so the face's orientation can be measured directly:
the line between the outer eye corners gives its x axis and the line from forehead to chin its y
axis. This needs no 3D face model and no camera calibration, and unlike a solver it has no
iterations to diverge or mirrored solution to fall into.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from dss.domain import HeadPose, Landmarks
from dss.vision.geometry import CHIN, EYE_OUTER_CORNERS, FOREHEAD

_MIN_AXIS_LENGTH = 1e-6


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float64] | None:
    length = float(np.linalg.norm(vector))
    return None if length < _MIN_AXIS_LENGTH else vector / length


def estimate_head_pose(points: Landmarks) -> HeadPose | None:
    """Estimate head orientation, or return None if the landmarks are degenerate.

    Axes follow the camera: x right, y down, z away from the lens. A face looking straight at the
    camera has an identity rotation, so the Euler angles read directly as pitch, yaw and roll.
    """
    across = _unit(points[EYE_OUTER_CORNERS[1]] - points[EYE_OUTER_CORNERS[0]])
    down = _unit(points[CHIN] - points[FOREHEAD])
    if across is None or down is None:
        return None

    forward = _unit(np.cross(across, down))
    if forward is None:
        return None
    # The two measured lines are never exactly perpendicular, so rebuild y from the other two.
    rotation = np.column_stack((across, np.cross(forward, across), forward))

    return HeadPose(
        pitch=math.degrees(math.atan2(rotation[2, 1], rotation[2, 2])),
        yaw=math.degrees(math.asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))),
        roll=math.degrees(math.atan2(rotation[1, 0], rotation[0, 0])),
    )
