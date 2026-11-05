"""The parts that touch the outside world: video files, image folders, audio, logs and the CLI."""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import cv2 as cv
import numpy as np
import pandas as pd
import pytest

from conftest import ScriptedDetector, synthetic_face
from dss import cli
from dss.domain import AlertLevel
from dss.ml.dataset import UNKNOWN_SUBJECT, extract_features, find_images
from dss.observability import JsonFormatter
from dss.service.alerts import SoundAlarm
from dss.service.video import VideoSource
from test_pipeline import feature_table
from test_service import wait_until


def write_video(path: Path, frames: int = 5, fps: float = 200.0) -> Path:
    writer = cv.VideoWriter(str(path), cv.VideoWriter_fourcc(*"MJPG"), fps, (64, 48))
    assert writer.isOpened()
    for _ in range(frames):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    return path


# ---- VideoSource ---------------------------------------------------------------------------


def test_file_plays_to_the_end(tmp_path: Path) -> None:
    source = VideoSource(str(write_video(tmp_path / "clip.avi")))
    assert source.open()
    read = [source.read() for _ in range(6)]
    source.close()
    assert all(frame is not None for frame in read[:5])
    assert read[5] is None


def test_file_can_loop(tmp_path: Path) -> None:
    source = VideoSource(str(write_video(tmp_path / "clip.avi", frames=3)), loop=True)
    assert source.open()
    frames = [source.read() for _ in range(10)]
    source.close()
    assert all(frame is not None for frame in frames)


def test_missing_source_reports_failure(tmp_path: Path) -> None:
    source = VideoSource(str(tmp_path / "missing.avi"))
    assert not source.open()
    assert source.read() is None


# ---- Dataset extraction --------------------------------------------------------------------


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv.imwrite(str(path), np.zeros((48, 64, 3), dtype=np.uint8))


def test_finds_labels_and_subjects(tmp_path: Path) -> None:
    write_image(tmp_path / "drowsy" / "anna" / "1.png")
    write_image(tmp_path / "notdrowsy" / "ben" / "1.png")
    write_image(tmp_path / "alert" / "2.png")
    write_image(tmp_path / "unrelated" / "3.png")

    found = {(item.label, item.subject) for item in find_images(tmp_path)}
    assert found == {(1, "anna"), (0, "ben"), (0, UNKNOWN_SUBJECT)}


def test_extraction_skips_unusable_images(tmp_path: Path) -> None:
    write_image(tmp_path / "drowsy" / "anna" / "good.png")
    (tmp_path / "drowsy" / "anna" / "corrupt.png").write_bytes(b"not an image")

    table = extract_features(tmp_path, ScriptedDetector(synthetic_face()))
    assert len(table) == 1
    assert table.iloc[0]["subject"] == "anna"
    assert table.iloc[0]["ear_left"] == pytest.approx(0.30)


def test_extraction_skips_frames_with_no_face(tmp_path: Path) -> None:
    write_image(tmp_path / "drowsy" / "anna" / "1.png")
    assert extract_features(tmp_path, ScriptedDetector(None)).empty


# ---- Audio alarm ---------------------------------------------------------------------------


@pytest.fixture
def fake_audio(monkeypatch: pytest.MonkeyPatch) -> list[np.ndarray]:
    played: list[np.ndarray] = []
    module = types.ModuleType("sounddevice")
    module.play = lambda data, rate: played.append(data)  # type: ignore[attr-defined]
    module.wait = lambda: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return played


def test_alarm_is_silent_while_normal(fake_audio: list[np.ndarray]) -> None:
    alarm = SoundAlarm()
    alarm.start()
    alarm.set_level(AlertLevel.NORMAL)
    alarm.stop()
    assert fake_audio == []


def test_alarm_beeps_until_the_driver_recovers(fake_audio: list[np.ndarray]) -> None:
    alarm = SoundAlarm()
    alarm.start()
    alarm.set_level(AlertLevel.CRITICAL)
    wait_until(lambda: len(fake_audio) >= 1)
    alarm.set_level(AlertLevel.NORMAL)
    alarm.stop()
    assert not alarm._thread.is_alive()


def test_alarm_survives_a_machine_with_no_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("sounddevice")

    def fail(*_: object) -> None:
        raise OSError("PortAudio library not found")

    module.play = fail  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sounddevice", module)

    alarm = SoundAlarm()
    alarm.start()
    alarm.set_level(AlertLevel.WARNING)
    wait_until(lambda: not alarm._thread.is_alive())
    alarm.set_level(AlertLevel.CRITICAL)  # still safe to call
    alarm.stop()


# ---- Logging -------------------------------------------------------------------------------


def test_json_logs_are_one_object_per_line() -> None:
    record = logging.LogRecord("dss", logging.INFO, "x", 1, "hello %s", ("driver",), None)
    entry = json.loads(JsonFormatter().format(record))
    assert entry["message"] == "hello driver"
    assert entry["level"] == "INFO"


def test_json_logs_include_exceptions() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = logging.LogRecord("dss", logging.ERROR, "x", 1, "failed", (), sys.exc_info())
    assert "ValueError: boom" in json.loads(JsonFormatter().format(record))["exception"]


# ---- Command line --------------------------------------------------------------------------


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DSS_MODEL_PATH", raising=False)


@pytest.mark.usefixtures("clean_env")
def test_train_command_writes_a_model_and_metrics(tmp_path: Path) -> None:
    data = tmp_path / "features.csv"
    feature_table().to_csv(data, index=False)
    model = tmp_path / "models" / "model.joblib"

    assert cli.main(["train", "--data", str(data), "--output", str(model)]) == 0

    assert model.exists()
    report = json.loads(model.with_suffix(".metrics.json").read_text(encoding="utf-8"))
    assert report["split"] == "by_subject"


@pytest.mark.usefixtures("clean_env")
def test_extract_command_writes_a_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dss.vision.landmarks.MediaPipeFaceMesh",
        lambda **_: ScriptedDetector(synthetic_face()),
    )
    write_image(tmp_path / "raw" / "drowsy" / "anna" / "1.png")
    output = tmp_path / "out" / "features.csv"

    assert cli.main(["extract", "--input", str(tmp_path / "raw"), "--output", str(output)]) == 0
    assert len(pd.read_csv(output)) == 1


@pytest.mark.usefixtures("clean_env")
def test_extract_command_fails_when_nothing_is_usable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "dss.vision.landmarks.MediaPipeFaceMesh", lambda **_: ScriptedDetector(None)
    )
    write_image(tmp_path / "raw" / "drowsy" / "1.png")
    code = cli.main(
        ["extract", "--input", str(tmp_path / "raw"), "--output", str(tmp_path / "x.csv")]
    )
    assert code == 1
    assert not (tmp_path / "x.csv").exists()


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        cli.build_parser().parse_args(["--version"])
    assert stopped.value.code == 0
    assert "dss" in capsys.readouterr().out


@pytest.mark.usefixtures("clean_env")
def test_unsafe_configuration_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DSS_HOST", "0.0.0.0")
    monkeypatch.delenv("DSS_API_KEY", raising=False)
    assert cli.main(["serve"]) == 2
    assert "DSS_API_KEY" in capsys.readouterr().err
