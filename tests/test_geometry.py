"""Facial ratios must describe the face, not the camera."""

import pytest

from conftest import synthetic_face
from dss.vision.geometry import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio, mouth_aspect_ratio


@pytest.mark.parametrize("size", [(640, 480), (1280, 720), (480, 480), (1920, 1080)])
def test_eye_ratio_does_not_depend_on_camera_resolution(size: tuple[int, int]) -> None:
    width, height = size
    points = synthetic_face(width, height, ear=0.17)
    assert eye_aspect_ratio(points, RIGHT_EYE) == pytest.approx(0.17)
    assert eye_aspect_ratio(points, LEFT_EYE) == pytest.approx(0.17)


@pytest.mark.parametrize("size", [(640, 480), (1280, 720), (480, 480)])
def test_mouth_ratio_does_not_depend_on_camera_resolution(size: tuple[int, int]) -> None:
    width, height = size
    assert mouth_aspect_ratio(synthetic_face(width, height, mar=0.7)) == pytest.approx(0.7)


def test_mouth_ratio_rises_as_the_mouth_opens() -> None:
    closed = mouth_aspect_ratio(synthetic_face(mar=0.02))
    talking = mouth_aspect_ratio(synthetic_face(mar=0.3))
    yawning = mouth_aspect_ratio(synthetic_face(mar=0.8))
    assert closed < talking < yawning


def test_degenerate_landmarks_do_not_divide_by_zero() -> None:
    points = synthetic_face() * 0
    assert eye_aspect_ratio(points, LEFT_EYE) == 0
    assert mouth_aspect_ratio(points) == 0
