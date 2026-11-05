"""Runs the real MediaPipe model on a real photograph.

Everything else in this suite uses synthetic landmarks. These tests prove the landmark indices,
geometry and head pose match what MediaPipe actually produces.
"""

import cv2 as cv
import numpy as np
import pytest

from dss.detection.pipeline import measure
from dss.domain import Frame, FrameMetrics, Landmarks
from dss.vision.head_pose import estimate_head_pose
from dss.vision.landmarks import MediaPipeFaceMesh
from test_head_pose import rotated

pytestmark = pytest.mark.mediapipe

skimage_data = pytest.importorskip("skimage.data")

WIDE = 910  # 16:9, for the 512 pixel tall portrait


@pytest.fixture(scope="module")
def portrait() -> Frame:
    return cv.cvtColor(skimage_data.astronaut(), cv.COLOR_RGB2BGR)


def analyse(frame: Frame) -> tuple[FrameMetrics, Landmarks | None]:
    detector = MediaPipeFaceMesh(static_images=True)
    try:
        return measure(detector, frame, 0.0)
    finally:
        detector.close()


def pad_to_width(frame: Frame, width: int) -> Frame:
    """Widen the frame with black bars. The face is unchanged; only the aspect ratio differs."""
    height, current = frame.shape[:2]
    padded = np.zeros((height, width, 3), dtype=np.uint8)
    padded[:, (width - current) // 2 : (width - current) // 2 + current] = frame
    return padded


def test_measures_a_real_face(portrait: Frame) -> None:
    metrics, landmarks = analyse(portrait)
    assert metrics.face_detected
    assert landmarks is not None
    assert landmarks.shape == (478, 3)
    assert metrics.pose is not None
    assert metrics.ear_left is not None
    assert metrics.ear_right is not None
    assert metrics.mar is not None
    assert 0.2 < metrics.ear_left < 0.5
    assert 0.2 < metrics.ear_right < 0.5
    assert metrics.mar < 0.6  # smiling, not yawning


def test_a_roughly_frontal_portrait_reads_as_roughly_frontal(portrait: Frame) -> None:
    pose = analyse(portrait)[0].pose
    assert pose is not None
    assert abs(pose.pitch) < 20
    assert abs(pose.yaw) < 20
    assert abs(pose.roll) < 20


def test_no_face_in_a_blank_frame() -> None:
    assert not analyse(np.zeros((480, 640, 3), dtype=np.uint8))[0].face_detected


def test_eye_ratio_does_not_scale_with_aspect_ratio(portrait: Frame) -> None:
    """The original formula read this face 78% higher at 16:9 than in a square frame.

    MediaPipe's landmarks do move a little with the crop, so this allows for that noise while
    staying far tighter than the error it guards against.
    """
    square = analyse(portrait)[0].ear_mean
    wide = analyse(pad_to_width(portrait, WIDE))[0].ear_mean
    assert square is not None
    assert wide is not None
    assert wide == pytest.approx(square, rel=0.12)


@pytest.mark.parametrize(
    ("pitch", "yaw", "roll"), [(25, 0, 0), (-25, 0, 0), (0, 30, 0), (0, -30, 0), (0, 0, 20)]
)
def test_tracks_rotation_of_real_face_geometry(
    portrait: Frame, pitch: float, yaw: float, roll: float
) -> None:
    """Turning the measured face about one axis must move the estimate by about that angle.

    The tolerance covers the portrait's small starting pose, because Euler angles of a rotated
    pose do not add exactly.
    """
    _, landmarks = analyse(portrait)
    assert landmarks is not None
    neutral = estimate_head_pose(landmarks)
    turned = estimate_head_pose(rotated(landmarks, pitch, yaw, roll))
    assert neutral is not None
    assert turned is not None
    assert turned.pitch - neutral.pitch == pytest.approx(pitch, abs=6)
    assert turned.yaw - neutral.yaw == pytest.approx(yaw, abs=6)
    assert turned.roll - neutral.roll == pytest.approx(roll, abs=6)
