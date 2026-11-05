"""Orientation is checked by rotating a face by a known angle and recovering that angle."""

import math

import numpy as np
import pytest

from conftest import synthetic_face
from dss.domain import Landmarks
from dss.vision.geometry import CHIN, FOREHEAD
from dss.vision.head_pose import estimate_head_pose


def rotation(pitch: float, yaw: float, roll: float) -> np.ndarray:
    p, y, r = (math.radians(angle) for angle in (pitch, yaw, roll))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    return rz @ ry @ rx


def rotated(points: Landmarks, pitch: float, yaw: float, roll: float) -> Landmarks:
    centre = points.mean(axis=0)
    return (points - centre) @ rotation(pitch, yaw, roll).T + centre


@pytest.mark.parametrize(
    ("pitch", "yaw", "roll"),
    [
        (0, 0, 0),
        (25, 0, 0),
        (-25, 0, 0),
        (0, 35, 0),
        (0, -35, 0),
        (0, 0, 30),
        (0, 0, -30),
        (20, 15, -10),
        (-10, -30, 20),
    ],
)
def test_recovers_known_orientation(pitch: float, yaw: float, roll: float) -> None:
    pose = estimate_head_pose(rotated(synthetic_face(), pitch, yaw, roll))
    assert pose is not None
    assert pose.pitch == pytest.approx(pitch, abs=1e-6)
    assert pose.yaw == pytest.approx(yaw, abs=1e-6)
    assert pose.roll == pytest.approx(roll, abs=1e-6)


def test_chin_dropping_towards_the_chest_is_positive_pitch() -> None:
    face = synthetic_face()
    dropped = rotated(face, 30, 0, 0)
    assert dropped[CHIN, 2] > face[CHIN, 2]  # chin moves away from the camera
    assert dropped[FOREHEAD, 2] < face[FOREHEAD, 2]  # forehead moves towards it

    pose = estimate_head_pose(dropped)
    assert pose is not None
    assert pose.pitch > 0


def test_orientation_does_not_depend_on_position_or_size() -> None:
    face = rotated(synthetic_face(), 12, -8, 5)
    moved = face * 1.7 + np.array([200.0, -50.0, 30.0])
    original, shifted = estimate_head_pose(face), estimate_head_pose(moved)
    assert original is not None
    assert shifted is not None
    assert shifted.pitch == pytest.approx(original.pitch)
    assert shifted.yaw == pytest.approx(original.yaw)
    assert shifted.roll == pytest.approx(original.roll)


def test_degenerate_landmarks_give_no_pose() -> None:
    assert estimate_head_pose(synthetic_face() * 0) is None
