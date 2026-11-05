"""Rendering must work for every view and every kind of analysis."""

import numpy as np
import pytest

from conftest import ScriptedDetector, synthetic_face
from dss.config import DetectionThresholds
from dss.detection.pipeline import FatiguePipeline
from dss.domain import Frame
from dss.vision.overlay import View, render


@pytest.mark.parametrize("view", list(View))
@pytest.mark.parametrize("face", [True, False])
def test_render_all_views(
    view: View, face: bool, frame: Frame, thresholds: DetectionThresholds
) -> None:
    detector = ScriptedDetector(synthetic_face() if face else None)
    analysis = FatiguePipeline(detector, thresholds).process(frame, 0.0)
    assert render(view, frame, analysis).shape == frame.shape


def test_render_leaves_the_shared_frame_untouched(
    frame: Frame, thresholds: DetectionThresholds
) -> None:
    """Several streams render the same frame, so drawing must happen on a copy."""
    analysis = FatiguePipeline(ScriptedDetector(synthetic_face()), thresholds).process(frame, 0.0)
    rendered = render(View.ANNOTATED, frame, analysis)
    assert not frame.any()
    assert rendered.any()


def test_original_view_has_no_overlay(frame: Frame, thresholds: DetectionThresholds) -> None:
    analysis = FatiguePipeline(ScriptedDetector(synthetic_face()), thresholds).process(frame, 0.0)
    assert np.array_equal(render(View.ORIGINAL, frame, analysis), frame)
