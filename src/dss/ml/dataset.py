"""Turning a folder of labelled images into a feature table for training."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2 as cv
import pandas as pd

from dss.detection.pipeline import measure
from dss.domain import Frame
from dss.ml.model import FEATURE_COLUMNS, feature_vector
from dss.vision.landmarks import LandmarkDetector

logger = logging.getLogger(__name__)

LABELS = {"drowsy": 1, "notdrowsy": 0, "alert": 0}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
UNKNOWN_SUBJECT = "unknown"


@dataclass(frozen=True)
class LabelledImage:
    path: Path
    label: int
    subject: str


def find_images(root: Path) -> Iterator[LabelledImage]:
    """Yield images laid out as ``root/<label>/[<subject>/]<image>``.

    Keeping each person in their own subject folder matters: it is what lets training test on
    people the model has never seen.
    """
    for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        label = LABELS.get(label_dir.name.lower())
        if label is None:
            logger.warning("Skipping %s: not one of %s", label_dir, sorted(LABELS))
            continue
        for path in sorted(label_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                parent = path.parent
                subject = UNKNOWN_SUBJECT if parent == label_dir else parent.name
                yield LabelledImage(path, label, subject)


def extract_features(root: Path, detector: LandmarkDetector) -> pd.DataFrame:
    """Measure every image under ``root``. Images with no usable face are skipped and counted."""
    rows: list[dict[str, object]] = []
    skipped = 0
    for image in find_images(root):
        frame = cast("Frame | None", cv.imread(str(image.path)))
        features = None
        if frame is not None:
            metrics, _ = measure(detector, frame, timestamp=0.0)
            features = feature_vector(metrics)
        if features is None:
            skipped += 1
            continue
        rows.append(
            {
                "subject": image.subject,
                "drowsy": image.label,
                **dict(zip(FEATURE_COLUMNS, features, strict=True)),
            }
        )

    logger.info("Extracted %d rows from %s, skipped %d images", len(rows), root, skipped)
    table: pd.DataFrame = pd.DataFrame(rows, columns=["subject", "drowsy", *FEATURE_COLUMNS])
    return table
