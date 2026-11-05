"""Training and evaluating the drowsiness classifier."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dss.ml.dataset import UNKNOWN_SUBJECT
from dss.ml.model import FEATURE_COLUMNS, save_artifact

logger = logging.getLogger(__name__)

MIN_SUBJECTS_FOR_GROUP_SPLIT = 2


@dataclass(frozen=True)
class TrainingReport:
    split: str
    train_rows: int
    test_rows: int
    subjects: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]


def split_indices(
    data: pd.DataFrame, test_size: float, seed: int
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], str]:
    """Hold out whole people where possible.

    Neighbouring video frames are nearly identical, so a random split puts near-duplicates of the
    test set in the training set and overstates accuracy. Grouping by subject measures what
    actually matters: how the model does on a driver it has never seen.
    """
    known = data["subject"] != UNKNOWN_SUBJECT
    if known.all() and data["subject"].nunique() >= MIN_SUBJECTS_FOR_GROUP_SPLIT:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        train, test = next(splitter.split(data, groups=data["subject"]))
        return train, test, "by_subject"

    logger.warning(
        "No subject information: falling back to a random split. "
        "Test scores will be optimistic if rows come from the same video."
    )
    train, test = train_test_split(
        np.arange(len(data)), test_size=test_size, random_state=seed, stratify=data["drowsy"]
    )
    return train, test, "random"


def train_model(
    data: pd.DataFrame, kernel: str = "linear", test_size: float = 0.2, seed: int = 10
) -> tuple[Any, TrainingReport]:
    """Fit a scaler and SVM on the training split and score it on the held-out split."""
    clean = data.dropna(subset=list(FEATURE_COLUMNS)).reset_index(drop=True)
    if clean["drowsy"].nunique() < 2:
        raise ValueError("Training data needs both drowsy and alert examples")

    train, test, split = split_indices(clean, test_size, seed)
    x = clean[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    y = clean["drowsy"].to_numpy(dtype=np.int64)
    if len(np.unique(y[train])) < 2:
        raise ValueError("The training split holds only one class: add more subjects")

    # The scaler sits inside the pipeline, so it only ever learns from the training split.
    estimator = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(kernel=kernel, class_weight="balanced"), ensemble=False),
    )
    estimator.fit(x[train], y[train])

    predicted = estimator.predict(x[test])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y[test], predicted, average="binary", zero_division=0
    )
    report = TrainingReport(
        split=split,
        train_rows=len(train),
        test_rows=len(test),
        subjects=int(clean["subject"].nunique()),
        accuracy=float(accuracy_score(y[test], predicted)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        confusion_matrix=confusion_matrix(y[test], predicted, labels=[0, 1]).tolist(),
    )
    return estimator, report


def run_training(data_path: Path, output_path: Path, kernel: str = "linear") -> TrainingReport:
    """Train from a feature CSV, writing the model and a metrics JSON file next to it."""
    estimator, report = train_model(pd.read_csv(data_path), kernel=kernel)
    save_artifact(estimator, output_path)
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    logger.info("Saved model to %s and metrics to %s", output_path, metrics_path)
    return report
