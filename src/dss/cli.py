"""The ``dss`` command line."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ValidationError

from dss import __version__
from dss.config import DetectionThresholds, Settings
from dss.observability import configure_logging

if TYPE_CHECKING:
    from dss.service.monitor import MonitorService

logger = logging.getLogger(__name__)


def _build_monitor(settings: Settings) -> MonitorService:
    # Heavy imports stay inside the commands so `dss --help` answers instantly.
    from dss.detection.pipeline import FatiguePipeline
    from dss.ml.model import DrowsinessModel
    from dss.service.alerts import AlertLog, SoundAlarm
    from dss.service.monitor import MonitorService
    from dss.service.video import VideoSource
    from dss.vision.landmarks import MediaPipeFaceMesh

    model = DrowsinessModel.load(settings.model_path) if settings.model_path else None
    thresholds = DetectionThresholds()

    def build_pipeline() -> FatiguePipeline:
        return FatiguePipeline(
            MediaPipeFaceMesh(),
            thresholds,
            model=model,
            model_threshold=settings.model_drowsy_probability,
            model_smoothing_s=settings.model_smoothing_s,
        )

    is_file = settings.camera_index is None
    return MonitorService(
        lambda: VideoSource(settings.video_source, settings.video_loop),
        build_pipeline,
        AlertLog(settings.alert_history_size),
        SoundAlarm() if settings.sound_enabled else None,
        retry_s=settings.camera_retry_s,
        stop_at_end=is_file and not settings.video_loop and "://" not in settings.video_source,
    )


def serve(settings: Settings) -> int:
    """Run the monitor behind the HTTP API until interrupted."""
    from waitress import serve as waitress_serve

    from dss.service.api import create_app

    monitor = _build_monitor(settings)
    app = create_app(settings, monitor)
    monitor.start()

    def shut_down(*_: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, shut_down)
    logger.info("Serving on http://%s:%d", settings.host, settings.port)
    try:
        waitress_serve(
            app,
            host=settings.host,
            port=settings.port,
            threads=settings.server_threads,
            ident="dss",
        )
    except KeyboardInterrupt:
        logger.info("Shutting down")
    finally:
        monitor.stop()
    return 0


def monitor_window(settings: Settings) -> int:
    """Run the monitor with a local preview window. Press q to quit."""
    import cv2 as cv

    from dss.vision.overlay import View, render

    monitor = _build_monitor(settings)
    monitor.start()
    version = 0
    try:
        while monitor.running:
            update = monitor.frames.wait_newer(version, timeout=0.2)
            if update is not None:
                version, processed = update
                image = render(View.ANNOTATED, processed.frame, processed.analysis)
                cv.imshow("Driver Surveillance System", image)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop()
        cv.destroyAllWindows()
    return 0


def extract(input_dir: Path, output: Path) -> int:
    """Build a training CSV from a labelled image folder."""
    from dss.ml.dataset import extract_features
    from dss.vision.landmarks import MediaPipeFaceMesh

    detector = MediaPipeFaceMesh(static_images=True)
    try:
        table = extract_features(input_dir, detector)
    finally:
        detector.close()
    if table.empty:
        logger.error("No usable faces found under %s", input_dir)
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"Wrote {len(table)} rows to {output}")
    return 0


def train(data: Path, output: Path, kernel: str) -> int:
    """Train the classifier and print its held-out scores."""
    from dss.ml.train import run_training

    report = run_training(data, output, kernel=kernel)
    print(json.dumps(asdict(report), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dss", description=__doc__)
    parser.add_argument("--version", action="version", version=f"dss {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("serve", help="run the HTTP API and dashboard")
    commands.add_parser("monitor", help="run with a local preview window")

    extract_cmd = commands.add_parser("extract", help="build a training CSV from labelled images")
    extract_cmd.add_argument(
        "--input", type=Path, required=True, help="<input>/<label>/<subject>/*"
    )
    extract_cmd.add_argument("--output", type=Path, required=True)

    train_cmd = commands.add_parser("train", help="train the classifier from a feature CSV")
    train_cmd.add_argument("--data", type=Path, required=True)
    train_cmd.add_argument("--output", type=Path, required=True)
    train_cmd.add_argument("--kernel", default="linear", choices=["linear", "rbf"])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings()
    except ValidationError as error:
        for problem in error.errors():
            print(f"Configuration error: {problem['msg']}", file=sys.stderr)
        return 2
    configure_logging(settings.log_level, settings.log_format)

    if args.command == "serve":
        return serve(settings)
    if args.command == "monitor":
        return monitor_window(settings)
    if args.command == "extract":
        return extract(args.input, args.output)
    return train(args.data, args.output, args.kernel)
