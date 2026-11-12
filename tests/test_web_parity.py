"""The browser demo in web/ copies the detection rules into JavaScript.

Its tests in tests/web replay the fixture below and must reproduce it exactly. This test proves the
fixture still describes the Python code, so the two implementations cannot drift apart unnoticed.

After changing a rule on purpose, regenerate the fixture:

    PYTHONPATH=src:tests python tests/test_web_parity.py --write
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from conftest import synthetic_face
from dss.config import DetectionThresholds
from dss.detection.temporal import TemporalAnalyzer
from dss.domain import FrameMetrics, HeadPose
from dss.vision.geometry import (
    CHIN,
    EYE_OUTER_CORNERS,
    FOREHEAD,
    INNER_LIPS,
    LEFT_EYE,
    MOUTH_CORNERS,
    RIGHT_EYE,
    eye_aspect_ratio,
    mouth_aspect_ratio,
)
from dss.vision.head_pose import estimate_head_pose
from test_head_pose import rotated

FIXTURE = Path(__file__).parent / "web" / "parity_fixture.json"
FPS = 10
DIGITS = 9
USED_LANDMARKS = sorted(
    {*RIGHT_EYE, *LEFT_EYE, *MOUTH_CORNERS, *INNER_LIPS, *EYE_OUTER_CORNERS, FOREHEAD, CHIN}
)


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), DIGITS)


def _geometry_cases() -> list[dict[str, Any]]:
    rng = np.random.default_rng(3)
    faces = [
        synthetic_face(),
        synthetic_face(ear=0.17),
        synthetic_face(mar=0.7, ear_right=0.12),
        rotated(synthetic_face(1280, 720), 20, -15, 10),
        rotated(synthetic_face(), -12, 25, -30) + rng.normal(0, 2, (478, 3)),
        synthetic_face() * 0,
    ]
    cases = []
    for face in faces:
        points = {str(i): [_round(c) for c in face[i]] for i in USED_LANDMARKS}
        # Measure the rounded points, which are exactly what the JavaScript test receives.
        exact = np.zeros((478, 3))
        for index, xyz in points.items():
            exact[int(index)] = xyz
        pose = estimate_head_pose(exact)
        cases.append(
            {
                "points": points,
                "ear_left": _round(eye_aspect_ratio(exact, LEFT_EYE)),
                "ear_right": _round(eye_aspect_ratio(exact, RIGHT_EYE)),
                "mar": _round(mouth_aspect_ratio(exact)),
                "pose": None
                if pose is None
                else [_round(pose.pitch), _round(pose.yaw), _round(pose.roll)],
            }
        )
    return cases


def _scenario_inputs() -> list[FrameMetrics]:
    """80 seconds that exercise every rule, with a little measurement noise."""
    rng = random.Random(7)
    neutral = (10.0, 5.0, 2.0)  # an off-centre camera, which calibration must absorb
    frames: list[FrameMetrics] = []

    def add(
        start: float,
        end: float,
        *,
        ear: float = 0.30,
        ear_right: float | None = None,
        mar: float = 0.05,
        turn: tuple[float, float, float] = (0.0, 0.0, 0.0),
        face: bool = True,
        unchanged: bool = False,
    ) -> None:
        for index in range(round(start * FPS), round(end * FPS)):
            timestamp = _round(index / FPS)
            assert timestamp is not None
            if not face:
                frames.append(FrameMetrics(timestamp, False, frame_unchanged=unchanged))
                continue
            right = ear if ear_right is None else ear_right
            pose = HeadPose(
                *(
                    _round(base + delta + rng.gauss(0, 1.0))  # type: ignore[arg-type]
                    for base, delta in zip(neutral, turn, strict=True)
                )
            )
            frames.append(
                FrameMetrics(
                    timestamp=timestamp,
                    face_detected=True,
                    ear_left=_round(ear + rng.gauss(0, 0.01)),
                    ear_right=_round(right + rng.gauss(0, 0.01)),
                    mar=_round(mar + rng.gauss(0, 0.01)),
                    pose=pose,
                    frame_unchanged=unchanged,
                )
            )

    add(0, 6)  # calibration
    add(6, 6.3, ear=0.05)  # an ordinary blink
    add(6.3, 10)
    add(10, 12.5, ear=0.05)  # eyes closed too long
    add(12.5, 15)
    add(15, 17, ear=0.32, ear_right=0.10)  # a wink
    for start in (17, 22, 27):  # three yawns
        add(start, start + 3, mar=0.8)
        add(start + 3, start + 5)
    add(32, 35, turn=(30, 0, 0))  # nodding off
    add(35, 37)
    add(37, 40, turn=(0, 40, 0))  # looking away
    add(40, 42)
    add(42, 45, turn=(0, 0, -35))  # head tilted
    add(45, 47)
    add(47, 50, face=False)  # driver out of view
    add(50, 52)
    add(52, 56.5, ear=0.05, unchanged=True)  # camera froze mid-blink
    add(56.5, 60)
    for start in range(60, 76, 2):  # repeated long blinks
        add(start, start + 1.4)
        add(start + 1.4, start + 2, ear=0.05)
    add(76, 80)
    return frames


def build_fixture() -> dict[str, Any]:
    analyzer = TemporalAnalyzer(DetectionThresholds())
    inputs, expected = [], []
    for metrics in _scenario_inputs():
        result = analyzer.update(metrics)
        pose = metrics.pose
        inputs.append(
            [
                metrics.timestamp,
                int(metrics.face_detected),
                metrics.ear_left,
                metrics.ear_right,
                metrics.mar,
                None if pose is None else [pose.pitch, pose.yaw, pose.roll],
                int(metrics.frame_unchanged),
            ]
        )
        expected.append(
            [
                result.level.value,
                [reason.value for reason in result.reasons],
                _round(result.perclos),
                _round(result.eye_closure_s),
                result.yawns_in_window,
                int(result.calibrated),
            ]
        )
    fixture = {
        "thresholds": asdict(DetectionThresholds()),
        "geometry": _geometry_cases(),
        "scenario": {"inputs": inputs, "expected": expected},
    }
    # A JSON round trip gives lists rather than tuples, as when the file is read back.
    loaded: dict[str, Any] = json.loads(json.dumps(fixture))
    return loaded


def test_fixture_matches_the_python_rules() -> None:
    stored = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert stored == build_fixture(), (
        "The Python rules changed. If that was intended, regenerate tests/web/parity_fixture.json "
        "and update web/js/fatigue.js to match."
    )


def test_scenario_exercises_every_rule() -> None:
    seen = {reason for row in build_fixture()["scenario"]["expected"] for reason in row[1]}
    assert seen == {
        "prolonged_eye_closure",
        "high_perclos",
        "frequent_yawning",
        "head_nodding",
        "looking_away",
        "head_tilted",
        "driver_not_visible",
        "camera_frozen",
    }


if __name__ == "__main__" and "--write" in sys.argv:
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(build_fixture(), separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"Wrote {FIXTURE}")
