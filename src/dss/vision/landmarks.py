"""Face landmark detection, behind a small interface so the rest of the system can be tested."""

from __future__ import annotations

from typing import Protocol

import cv2 as cv
import numpy as np

from dss.domain import Frame, Landmarks


class LandmarkDetector(Protocol):
    def detect(self, frame: Frame) -> Landmarks | None:
        """Return pixel landmarks for the most prominent face, or None if there is no face."""
        ...

    def close(self) -> None: ...


class MediaPipeFaceMesh:
    """MediaPipe Face Mesh adapter. Not thread-safe: use one instance per thread."""

    def __init__(
        self,
        static_images: bool = False,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        # Imported here because MediaPipe is slow to load and only this adapter needs it.
        from mediapipe.python import solutions

        self._mesh = solutions.face_mesh.FaceMesh(
            static_image_mode=static_images,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame: Frame) -> Landmarks | None:
        results = self._mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None

        height, width = frame.shape[:2]
        landmarks = results.multi_face_landmarks[0].landmark
        # MediaPipe scales z by the image width, so all three axes end up in the same units.
        return np.array(
            [(lm.x * width, lm.y * height, lm.z * width) for lm in landmarks], dtype=np.float64
        )

    def close(self) -> None:
        self._mesh.close()
