"""A frozen picture must be caught, without mistaking a motionless driver for one."""

import numpy as np

from conftest import live_frame
from dss.vision.freeze import FreezeDetector

MIN_CHANGE = 0.05


def test_repeated_frames_are_unchanged() -> None:
    detector = FreezeDetector(MIN_CHANGE)
    frame = live_frame(1)
    assert detector.is_unchanged(frame) is False  # nothing to compare with yet
    assert detector.is_unchanged(frame.copy()) is True
    assert detector.is_unchanged(frame.copy()) is True


def test_sensor_noise_counts_as_change() -> None:
    detector = FreezeDetector(MIN_CHANGE)
    results = [detector.is_unchanged(live_frame(seed)) for seed in range(5)]
    assert results == [False] * 5


def test_a_faint_noise_floor_is_still_live() -> None:
    """A still scene differs only slightly between frames, which must be enough."""
    detector = FreezeDetector(MIN_CHANGE)
    scene = np.full((480, 640, 3), 100, dtype=np.uint8)
    rng = np.random.default_rng(3)
    for _ in range(10):
        noise = rng.integers(0, 3, scene.shape, dtype=np.uint8)
        assert detector.is_unchanged(scene + noise) is False


def test_a_slow_drift_is_not_mistaken_for_a_freeze() -> None:
    """Each step is below the threshold, but the changes add up against the reference frame."""
    detector = FreezeDetector(min_change=0.5)
    frame = np.full((480, 640, 3), 100, dtype=np.uint8)
    detector.is_unchanged(frame)

    frame[:144] += 1  # 30% of the picture moves by one level
    assert detector.is_unchanged(frame) is True
    frame[:144] += 1
    assert detector.is_unchanged(frame) is False


def test_a_change_of_resolution_starts_again() -> None:
    detector = FreezeDetector(MIN_CHANGE)
    detector.is_unchanged(np.zeros((480, 640, 3), dtype=np.uint8))
    assert detector.is_unchanged(np.zeros((720, 1280, 3), dtype=np.uint8)) is False
